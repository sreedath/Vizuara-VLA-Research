#!/usr/bin/env python3
"""Experiment 324: Pixel-Level Sensitivity Probing (Real OpenVLA-7B)

Tests the detector's sensitivity at the pixel level:
1. Single-pixel perturbation map: which pixels cause max embedding shift?
2. Structured patterns: stripes, checkerboard, gradients — do they evade detection?
3. Semantic regions: sky vs road vs objects — where is the model most sensitive?
4. Frequency-band sensitivity: low/mid/high frequency perturbations
5. Color channel sensitivity: R vs G vs B perturbations
6. Detection boundary: minimum number of pixels to trigger detection
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

    np.random.seed(42)
    pixels = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    base_img = Image.fromarray(pixels)
    prompt = "In: What action should the robot take to pick up the object?\nOut:"

    clean_emb = extract_hidden(model, processor, base_img, prompt)

    results = {}

    # ========== 1. Structured pattern perturbations ==========
    print("\n=== Structured Pattern Perturbations ===")
    patterns = {}

    # Horizontal stripes
    arr = np.array(base_img).copy()
    for row in range(0, 224, 2):
        arr[row, :, :] = np.clip(arr[row, :, :].astype(int) + 50, 0, 255)
    stripe_h = Image.fromarray(arr.astype(np.uint8))
    d = cosine_dist(clean_emb, extract_hidden(model, processor, stripe_h, prompt))
    patterns['horizontal_stripes'] = float(d)
    print(f"  Horizontal stripes: d={d:.8f}")

    # Vertical stripes
    arr = np.array(base_img).copy()
    for col in range(0, 224, 2):
        arr[:, col, :] = np.clip(arr[:, col, :].astype(int) + 50, 0, 255)
    stripe_v = Image.fromarray(arr.astype(np.uint8))
    d = cosine_dist(clean_emb, extract_hidden(model, processor, stripe_v, prompt))
    patterns['vertical_stripes'] = float(d)
    print(f"  Vertical stripes: d={d:.8f}")

    # Checkerboard
    arr = np.array(base_img).copy()
    for r in range(224):
        for c in range(224):
            if (r + c) % 2 == 0:
                arr[r, c, :] = np.clip(arr[r, c, :].astype(int) + 50, 0, 255)
    checker = Image.fromarray(arr.astype(np.uint8))
    d = cosine_dist(clean_emb, extract_hidden(model, processor, checker, prompt))
    patterns['checkerboard'] = float(d)
    print(f"  Checkerboard: d={d:.8f}")

    # Gradient overlay (left-right brightness gradient)
    arr = np.array(base_img).astype(float)
    gradient = np.linspace(0, 30, 224).reshape(1, 224, 1)
    arr = np.clip(arr + gradient, 0, 255)
    grad_img = Image.fromarray(arr.astype(np.uint8))
    d = cosine_dist(clean_emb, extract_hidden(model, processor, grad_img, prompt))
    patterns['gradient_overlay'] = float(d)
    print(f"  Gradient overlay: d={d:.8f}")

    # Single-color tint (slightly red)
    arr = np.array(base_img).copy().astype(float)
    arr[:, :, 0] = np.clip(arr[:, :, 0] + 20, 0, 255)  # add red
    tint_img = Image.fromarray(arr.astype(np.uint8))
    d = cosine_dist(clean_emb, extract_hidden(model, processor, tint_img, prompt))
    patterns['red_tint'] = float(d)
    print(f"  Red tint: d={d:.8f}")

    # Block pattern (16x16 blocks, every other block brighter)
    arr = np.array(base_img).copy()
    for br in range(0, 224, 32):
        for bc in range(0, 224, 32):
            if ((br // 32) + (bc // 32)) % 2 == 0:
                arr[br:br+16, bc:bc+16, :] = np.clip(arr[br:br+16, bc:bc+16, :].astype(int) + 40, 0, 255)
    block_img = Image.fromarray(arr.astype(np.uint8))
    d = cosine_dist(clean_emb, extract_hidden(model, processor, block_img, prompt))
    patterns['block_pattern'] = float(d)
    print(f"  Block pattern: d={d:.8f}")

    results['structured_patterns'] = patterns

    # ========== 2. Spatial region sensitivity ==========
    print("\n=== Spatial Region Sensitivity ===")
    regions = {}
    region_defs = {
        'top_quarter': (0, 0, 224, 56),
        'bottom_quarter': (0, 168, 224, 224),
        'left_quarter': (0, 0, 56, 224),
        'right_quarter': (168, 0, 224, 224),
        'center': (56, 56, 168, 168),
        'top_left': (0, 0, 112, 112),
        'top_right': (112, 0, 224, 112),
        'bottom_left': (0, 112, 112, 224),
        'bottom_right': (112, 112, 224, 224),
    }

    for rname, (x1, y1, x2, y2) in region_defs.items():
        arr = np.array(base_img).copy().astype(float)
        # Apply fog only to this region
        arr[y1:y2, x1:x2, :] = arr[y1:y2, x1:x2, :] * 0.4 + 0.6 * 255
        region_img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        d = cosine_dist(clean_emb, extract_hidden(model, processor, region_img, prompt))
        n_pixels = (x2 - x1) * (y2 - y1)
        regions[rname] = {'distance': float(d), 'n_pixels': n_pixels, 'd_per_pixel': float(d / n_pixels) if n_pixels > 0 else 0}
        print(f"  {rname}: d={d:.8f}, pixels={n_pixels}")

    results['spatial_regions'] = regions

    # ========== 3. Color channel sensitivity ==========
    print("\n=== Color Channel Sensitivity ===")
    channels = {}
    for ch_idx, ch_name in enumerate(['red', 'green', 'blue']):
        for delta in [10, 25, 50, 100]:
            arr = np.array(base_img).copy().astype(float)
            arr[:, :, ch_idx] = np.clip(arr[:, :, ch_idx] + delta, 0, 255)
            ch_img = Image.fromarray(arr.astype(np.uint8))
            d = cosine_dist(clean_emb, extract_hidden(model, processor, ch_img, prompt))
            key = f"{ch_name}_+{delta}"
            channels[key] = float(d)
            print(f"  {key}: d={d:.8f}")

    results['color_channels'] = channels

    # ========== 4. Frequency-band perturbations ==========
    print("\n=== Frequency Band Sensitivity ===")
    freq_results = {}

    # Create frequency-band noise using DCT-like approach
    rng = np.random.RandomState(42)

    # Low frequency: large smooth blobs
    for freq_name, kernel_size in [('very_low', 31), ('low', 15), ('medium', 7), ('high', 3), ('very_high', 1)]:
        noise = rng.randn(224, 224, 3) * 30
        if kernel_size > 1:
            from scipy.ndimage import gaussian_filter
            noise = gaussian_filter(noise, sigma=kernel_size)
            # Normalize to same energy
            noise = noise / (np.std(noise) + 1e-10) * 30

        arr = np.array(base_img).astype(float) + noise
        freq_img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        d = cosine_dist(clean_emb, extract_hidden(model, processor, freq_img, prompt))
        freq_results[freq_name] = {'distance': float(d), 'kernel_size': kernel_size}
        print(f"  {freq_name} (k={kernel_size}): d={d:.8f}")

    results['frequency_bands'] = freq_results

    # ========== 5. Minimum pixels to trigger detection ==========
    print("\n=== Minimum Pixels for Detection ===")
    min_pixel_results = {}

    for n_pixels in [1, 10, 50, 100, 500, 1000, 5000, 10000, 50000]:
        rng_px = np.random.RandomState(42)
        arr = np.array(base_img).copy()

        # Randomly select pixels and set to white (maximum perturbation)
        total_px = 224 * 224
        if n_pixels > total_px:
            n_pixels = total_px
        indices = rng_px.choice(total_px, size=min(n_pixels, total_px), replace=False)
        rows = indices // 224
        cols = indices % 224
        arr[rows, cols, :] = 255  # white pixels

        perturbed = Image.fromarray(arr)
        d = cosine_dist(clean_emb, extract_hidden(model, processor, perturbed, prompt))
        pct = n_pixels / total_px * 100
        min_pixel_results[str(n_pixels)] = {'distance': float(d), 'percentage': float(pct), 'detected': bool(d > 0)}
        print(f"  {n_pixels} pixels ({pct:.2f}%): d={d:.8f}")

    results['min_pixels'] = min_pixel_results

    # ========== 6. Perturbation magnitude scaling ==========
    print("\n=== Perturbation Magnitude Scaling ===")
    mag_results = {}
    for mag in [1, 2, 5, 10, 20, 50, 100, 200]:
        rng_m = np.random.RandomState(42)
        noise = rng_m.randn(224, 224, 3) * mag
        arr = np.array(base_img).astype(float) + noise
        mag_img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        d = cosine_dist(clean_emb, extract_hidden(model, processor, mag_img, prompt))
        mag_results[str(mag)] = {'distance': float(d), 'detected': bool(d > 0)}
        print(f"  magnitude={mag}: d={d:.8f}")

    results['magnitude_scaling'] = mag_results

    # ========== 7. Adversarial pixel arrangements ==========
    print("\n=== Adversarial Arrangements (trying to evade) ===")
    evasion_results = {}

    # Uniform brightness shift
    for shift in [1, 5, 10, 20]:
        arr = np.clip(np.array(base_img).astype(int) + shift, 0, 255).astype(np.uint8)
        d = cosine_dist(clean_emb, extract_hidden(model, processor, Image.fromarray(arr), prompt))
        evasion_results[f'uniform_shift_{shift}'] = float(d)
        print(f"  Uniform shift +{shift}: d={d:.8f}")

    # Gamma correction
    for gamma in [0.9, 0.95, 1.05, 1.1]:
        arr = np.array(base_img).astype(float) / 255.0
        arr = np.clip(arr ** gamma, 0, 1) * 255
        d = cosine_dist(clean_emb, extract_hidden(model, processor, Image.fromarray(arr.astype(np.uint8)), prompt))
        evasion_results[f'gamma_{gamma}'] = float(d)
        print(f"  Gamma={gamma}: d={d:.8f}")

    # Contrast adjustment
    for factor in [0.9, 0.95, 1.05, 1.1]:
        arr = np.array(base_img).astype(float)
        mean = arr.mean()
        arr = (arr - mean) * factor + mean
        d = cosine_dist(clean_emb, extract_hidden(model, processor, Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)), prompt))
        evasion_results[f'contrast_{factor}'] = float(d)
        print(f"  Contrast={factor}: d={d:.8f}")

    # Salt and pepper (sparse extreme perturbation)
    for pct in [0.001, 0.01, 0.05, 0.1]:
        rng_sp = np.random.RandomState(42)
        arr = np.array(base_img).copy()
        n = int(224 * 224 * pct)
        idx = rng_sp.choice(224 * 224, size=n, replace=False)
        rows, cols = idx // 224, idx % 224
        vals = rng_sp.choice([0, 255], size=n)
        arr[rows, cols, :] = vals.reshape(-1, 1)
        d = cosine_dist(clean_emb, extract_hidden(model, processor, Image.fromarray(arr), prompt))
        evasion_results[f'salt_pepper_{pct}'] = float(d)
        print(f"  Salt&pepper {pct*100:.1f}%: d={d:.8f}")

    results['evasion_attempts'] = evasion_results

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/pixel_probe_{ts}.json"

    def convert(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    def recursive_convert(d):
        if isinstance(d, dict):
            return {k: recursive_convert(v) for k, v in d.items()}
        if isinstance(d, list):
            return [recursive_convert(x) for x in d]
        return convert(d)

    results = recursive_convert(results)

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
