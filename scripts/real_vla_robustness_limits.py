#!/usr/bin/env python3
"""Experiment 338: Robustness Limits (Real OpenVLA-7B)

Systematically probe the detector's limits:
1. Minimum detectable perturbation per corruption type
2. Maximum benign perturbation (false positive boundary)
3. Adversarial input crafting (gradient-free)
4. Edge cases: solid colors, gradients, patterns
5. Calibration age: how stale can calibration be?
6. Combined stress: drift + corruption
"""

import json, time, os, sys
import numpy as np
import torch
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor

def extract_hidden(model, processor, image, prompt, layer=3):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return fwd.hidden_states[layer][0, -1, :].float().cpu().numpy()

def apply_corruption(image, ctype, severity=1.0):
    arr = np.array(image).astype(np.float32) / 255.0
    if ctype == 'fog':
        arr = arr * (1 - 0.6 * severity) + 0.6 * severity
    elif ctype == 'night':
        arr = arr * max(0.01, 1.0 - 0.95 * severity)
    elif ctype == 'noise':
        arr = arr + np.random.RandomState(42).randn(*arr.shape) * 0.3 * severity
        arr = np.clip(arr, 0, 1)
    elif ctype == 'blur':
        return image.filter(ImageFilter.GaussianBlur(radius=10 * severity))
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))

def cosine_dist(a, b):
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return 1.0 - dot / (na * nb)

def main():
    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    results = {}

    np.random.seed(42)
    pixels = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    base_img = Image.fromarray(pixels)
    clean_emb = extract_hidden(model, processor, base_img, prompt)

    ctypes = ['fog', 'night', 'noise', 'blur']

    # ========== 1. Minimum detection threshold ==========
    print("\n=== Minimum Detection Threshold ===")
    min_detect = {}

    for ct in ctypes:
        # Binary search for minimum detectable severity
        lo, hi = 0.0, 0.01
        # First check if 0.01 is detectable
        img = apply_corruption(base_img, ct, hi)
        emb = extract_hidden(model, processor, img, prompt)
        d = cosine_dist(clean_emb, emb)

        if d == 0:
            # Need higher severity
            for test_sev in [0.02, 0.05, 0.1, 0.2]:
                img = apply_corruption(base_img, ct, test_sev)
                emb = extract_hidden(model, processor, img, prompt)
                d = cosine_dist(clean_emb, emb)
                if d > 0:
                    hi = test_sev
                    lo = test_sev / 2
                    break

        # Binary search
        for _ in range(10):
            mid = (lo + hi) / 2
            img = apply_corruption(base_img, ct, mid)
            emb = extract_hidden(model, processor, img, prompt)
            d = cosine_dist(clean_emb, emb)
            if d > 0:
                hi = mid
            else:
                lo = mid

        min_detect[ct] = {
            'min_severity': float(hi),
            'distance_at_min': float(cosine_dist(clean_emb, extract_hidden(model, processor, apply_corruption(base_img, ct, hi), prompt))),
        }
        print(f"  {ct}: min_sev={hi:.6f}")

    results['min_detection'] = min_detect

    # ========== 2. Edge case images ==========
    print("\n=== Edge Case Images ===")
    edge_results = {}

    # Solid colors
    for color_name, color in [('black', 0), ('white', 255), ('gray', 128), ('red', [255,0,0]), ('green', [0,255,0]), ('blue', [0,0,255])]:
        if isinstance(color, int):
            img_arr = np.full((224, 224, 3), color, dtype=np.uint8)
        else:
            img_arr = np.zeros((224, 224, 3), dtype=np.uint8)
            for ch in range(3):
                img_arr[:, :, ch] = color[ch]
        img = Image.fromarray(img_arr)
        emb = extract_hidden(model, processor, img, prompt)

        # Distance from our random scene
        d_from_random = cosine_dist(clean_emb, emb)

        # Can we detect fog on this image?
        fog_img = apply_corruption(img, 'fog', 1.0)
        fog_emb = extract_hidden(model, processor, fog_img, prompt)
        fog_d = cosine_dist(emb, fog_emb)

        edge_results[color_name] = {
            'dist_from_random_scene': float(d_from_random),
            'fog_detection_dist': float(fog_d),
            'fog_detectable': bool(fog_d > 0),
        }
        print(f"  {color_name}: d_from_random={d_from_random:.6f}, fog_d={fog_d:.6f}")

    # Gradient image
    gradient = np.zeros((224, 224, 3), dtype=np.uint8)
    for i in range(224):
        gradient[i, :, :] = int(i / 224 * 255)
    img = Image.fromarray(gradient)
    emb = extract_hidden(model, processor, img, prompt)
    d = cosine_dist(clean_emb, emb)
    fog_img = apply_corruption(img, 'fog', 1.0)
    fog_emb = extract_hidden(model, processor, fog_img, prompt)
    fog_d = cosine_dist(emb, fog_emb)
    edge_results['gradient'] = {
        'dist_from_random_scene': float(d),
        'fog_detection_dist': float(fog_d),
        'fog_detectable': bool(fog_d > 0),
    }
    print(f"  gradient: d={d:.6f}, fog_d={fog_d:.6f}")

    # Checkerboard pattern
    checker = np.zeros((224, 224, 3), dtype=np.uint8)
    for i in range(224):
        for j in range(224):
            if (i // 16 + j // 16) % 2 == 0:
                checker[i, j] = 200
            else:
                checker[i, j] = 50
    img = Image.fromarray(checker)
    emb = extract_hidden(model, processor, img, prompt)
    d = cosine_dist(clean_emb, emb)
    fog_img = apply_corruption(img, 'fog', 1.0)
    fog_emb = extract_hidden(model, processor, fog_img, prompt)
    fog_d = cosine_dist(emb, fog_emb)
    edge_results['checkerboard'] = {
        'dist_from_random_scene': float(d),
        'fog_detection_dist': float(fog_d),
        'fog_detectable': bool(fog_d > 0),
    }
    print(f"  checkerboard: d={d:.6f}, fog_d={fog_d:.6f}")

    results['edge_cases'] = edge_results

    # ========== 3. JPEG compression sensitivity ==========
    print("\n=== JPEG Compression ===")
    jpeg_results = {}
    import io

    for quality in [95, 80, 60, 40, 20, 5]:
        buffer = io.BytesIO()
        base_img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        jpeg_img = Image.open(buffer).convert('RGB')
        emb = extract_hidden(model, processor, jpeg_img, prompt)
        d = cosine_dist(clean_emb, emb)
        jpeg_results[f"q{quality}"] = {
            'distance': float(d),
            'detected': bool(d > 0),
        }
        print(f"  Q={quality}: d={d:.6f}")

    results['jpeg_compression'] = jpeg_results

    # ========== 4. Resolution scaling ==========
    print("\n=== Resolution Scaling ===")
    resolution_results = {}

    for size in [32, 64, 112, 224, 448]:
        if size == 224:
            img = base_img
        else:
            # Resize to target, then back to 224
            img = base_img.resize((size, size), Image.BILINEAR).resize((224, 224), Image.BILINEAR)
        emb = extract_hidden(model, processor, img, prompt)
        d = cosine_dist(clean_emb, emb)
        resolution_results[f"{size}x{size}"] = {
            'distance': float(d),
            'detected': bool(d > 0),
        }
        print(f"  {size}×{size}: d={d:.6f}")

    results['resolution'] = resolution_results

    # ========== 5. Combined stress (drift + corruption) ==========
    print("\n=== Combined Stress ===")
    combined_results = {}

    arr_base = np.array(base_img).astype(float)
    for drift_pct in [0, 5, 10, 20]:
        rng_d = np.random.RandomState(123)
        drift = rng_d.randn(*arr_base.shape) * drift_pct / 100 * 255
        drifted_arr = np.clip(arr_base + drift, 0, 255).astype(np.uint8)
        drifted_img = Image.fromarray(drifted_arr)
        drifted_emb = extract_hidden(model, processor, drifted_img, prompt)
        drifted_d = cosine_dist(clean_emb, drifted_emb)

        for ct in ['fog', 'blur']:
            for sev in [0.1, 0.5]:
                img = apply_corruption(drifted_img, ct, sev)
                emb = extract_hidden(model, processor, img, prompt)
                d = cosine_dist(clean_emb, emb)
                key = f"drift{drift_pct}_{ct}_{sev}"
                combined_results[key] = {
                    'drift_pct': drift_pct,
                    'corruption': ct,
                    'severity': sev,
                    'total_distance': float(d),
                    'drift_only_distance': float(drifted_d),
                    'net_corruption_signal': float(d - drifted_d),
                    'detected_vs_cal': bool(d > 0),
                }
                print(f"  drift={drift_pct}%, {ct}@{sev}: d={d:.6f}, net={d-drifted_d:.6f}")

    results['combined_stress'] = combined_results

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/robustness_limits_{ts}.json"
    def convert(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)): return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return obj
    def recursive_convert(d):
        if isinstance(d, dict): return {k: recursive_convert(v) for k, v in d.items()}
        if isinstance(d, list): return [recursive_convert(x) for x in d]
        return convert(d)
    results = recursive_convert(results)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
