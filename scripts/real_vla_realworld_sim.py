#!/usr/bin/env python3
"""Experiment 327: Real-World Corruption Simulation (Real OpenVLA-7B)

Tests with more realistic corruption patterns:
1. Rain simulation (diagonal streaks + darkening)
2. Sun glare (bright spot + overexposure)
3. Dust/dirt on lens (partial occlusion)
4. Motion blur (directional)
5. Shadow overlay (partial darkening)
6. Lens flare (bright artifacts)
7. Frost/ice on camera
8. Partial occlusion (hand, insect on lens)
9. Color temperature shift (warm/cool)
10. HDR failure (clipped highlights/shadows)
"""

import json, time, os, sys
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageDraw
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

def compute_auroc(id_scores, ood_scores):
    id_s = np.asarray(id_scores)
    ood_s = np.asarray(ood_scores)
    n_id, n_ood = len(id_s), len(ood_s)
    if n_id == 0 or n_ood == 0:
        return 0.5
    count = sum(float(np.sum(o > id_s) + 0.5 * np.sum(o == id_s)) for o in ood_s)
    return count / (n_id * n_ood)

def apply_rain(image, severity=1.0):
    """Simulate rain: diagonal streaks + slight darkening."""
    arr = np.array(image).astype(float)
    # Darken slightly
    arr = arr * (1 - 0.15 * severity)
    # Add diagonal streaks
    rng = np.random.RandomState(42)
    n_drops = int(500 * severity)
    for _ in range(n_drops):
        x = rng.randint(0, 224)
        y = rng.randint(0, 224)
        length = rng.randint(5, 20)
        for i in range(length):
            ny, nx = y + i, x + i // 2
            if 0 <= ny < 224 and 0 <= nx < 224:
                arr[ny, nx, :] = np.clip(arr[ny, nx, :] + 60 * severity, 0, 255)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def apply_glare(image, severity=1.0):
    """Simulate sun glare: bright spot with radial falloff."""
    arr = np.array(image).astype(float)
    cy, cx = 80, 140  # slightly off-center
    radius = 60 * severity
    Y, X = np.ogrid[:224, :224]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    mask = np.clip(1 - dist / max(radius, 1), 0, 1) ** 2
    arr += mask[:, :, None] * 200 * severity
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def apply_dust(image, severity=1.0):
    """Simulate dust/dirt spots on lens."""
    arr = np.array(image).astype(float)
    rng = np.random.RandomState(42)
    n_spots = int(20 * severity)
    for _ in range(n_spots):
        cx, cy = rng.randint(20, 204), rng.randint(20, 204)
        r = rng.randint(5, 15)
        Y, X = np.ogrid[:224, :224]
        mask = ((X - cx)**2 + (Y - cy)**2) < r**2
        arr[mask] = arr[mask] * (1 - 0.5 * severity) + 80 * severity
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def apply_motion_blur(image, severity=1.0):
    """Simulate horizontal motion blur."""
    arr = np.array(image).astype(float)
    kernel_size = max(1, int(15 * severity))
    # Simple horizontal averaging
    result = np.zeros_like(arr)
    for offset in range(kernel_size):
        shifted = np.roll(arr, offset, axis=1)
        result += shifted
    result /= kernel_size
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

def apply_shadow(image, severity=1.0):
    """Simulate shadow overlay (dark region)."""
    arr = np.array(image).astype(float)
    # Diagonal shadow
    for y in range(224):
        for x in range(224):
            if x + y < 224 * severity:
                arr[y, x, :] *= max(0.2, 1 - 0.6 * severity)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def apply_frost(image, severity=1.0):
    """Simulate frost/ice on camera (white overlay + blur)."""
    arr = np.array(image).astype(float)
    rng = np.random.RandomState(42)
    # White crystalline overlay
    frost = rng.rand(224, 224) ** (1 / max(severity, 0.1))
    frost_mask = (frost > 0.7).astype(float) * severity
    arr = arr * (1 - frost_mask[:, :, None]) + 240 * frost_mask[:, :, None]
    # Slight blur
    result = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return result.filter(ImageFilter.GaussianBlur(radius=2 * severity))

def apply_occlusion(image, severity=1.0):
    """Simulate partial occlusion (dark object covering part of image)."""
    arr = np.array(image).copy()
    # Black rectangle covering a fraction of the image
    h = int(224 * 0.3 * severity)
    w = int(224 * 0.4 * severity)
    y_start = 112 - h // 2
    x_start = 80
    arr[y_start:y_start+h, x_start:x_start+w, :] = 20  # near-black
    return Image.fromarray(arr)

def apply_warm_temp(image, severity=1.0):
    """Simulate warm color temperature (more red/yellow, less blue)."""
    arr = np.array(image).astype(float)
    arr[:, :, 0] = np.clip(arr[:, :, 0] + 20 * severity, 0, 255)  # more red
    arr[:, :, 1] = np.clip(arr[:, :, 1] + 10 * severity, 0, 255)  # slightly more green
    arr[:, :, 2] = np.clip(arr[:, :, 2] - 20 * severity, 0, 255)  # less blue
    return Image.fromarray(arr.astype(np.uint8))

def apply_cool_temp(image, severity=1.0):
    """Simulate cool color temperature (more blue, less red)."""
    arr = np.array(image).astype(float)
    arr[:, :, 0] = np.clip(arr[:, :, 0] - 20 * severity, 0, 255)  # less red
    arr[:, :, 2] = np.clip(arr[:, :, 2] + 20 * severity, 0, 255)  # more blue
    return Image.fromarray(arr.astype(np.uint8))

def apply_hdr_clip(image, severity=1.0):
    """Simulate HDR failure: clip highlights and crush shadows."""
    arr = np.array(image).astype(float)
    thresh = 128 * (1 - severity * 0.5)
    arr[arr > thresh + 50] = 255
    arr[arr < thresh - 50] = 0
    return Image.fromarray(arr.astype(np.uint8))

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

    corruption_fns = {
        'rain': apply_rain,
        'glare': apply_glare,
        'dust': apply_dust,
        'motion_blur': apply_motion_blur,
        'shadow': apply_shadow,
        'frost': apply_frost,
        'occlusion': apply_occlusion,
        'warm_temp': apply_warm_temp,
        'cool_temp': apply_cool_temp,
        'hdr_clip': apply_hdr_clip,
    }

    # ========== Test each corruption ==========
    print("\n=== Real-World Corruptions ===")
    for cname, cfn in corruption_fns.items():
        severities = [0.25, 0.5, 0.75, 1.0]
        distances = []
        for sev in severities:
            try:
                img = cfn(base_img, sev)
                emb = extract_hidden(model, processor, img, prompt)
                d = cosine_dist(clean_emb, emb)
                distances.append(float(d))
            except Exception as e:
                distances.append(-1.0)
                print(f"  {cname} sev={sev} ERROR: {e}")

        # AUROC with 5 clean variants
        clean_dists = [0.0] * 5  # all zero for same scene
        ood_dists = [d for d in distances if d >= 0]
        auroc = compute_auroc(clean_dists, ood_dists)

        results[cname] = {
            'distances': {str(s): d for s, d in zip(severities, distances)},
            'auroc': float(auroc),
            'detected_all': all(d > 0 for d in distances if d >= 0),
            'max_distance': float(max(d for d in distances if d >= 0)),
            'min_distance': float(min(d for d in distances if d >= 0)),
        }
        print(f"  {cname}: AUROC={auroc:.3f}, d=[{', '.join(f'{d:.6f}' for d in distances)}]")

    # ========== Test multiple scenes ==========
    print("\n=== Multi-Scene Detection ===")
    scene_results = {}
    for seed in [0, 42, 99, 123, 777]:
        rng = np.random.RandomState(seed)
        px = rng.randint(50, 200, (224, 224, 3)).astype(np.uint8)
        scene_img = Image.fromarray(px)
        scene_clean = extract_hidden(model, processor, scene_img, prompt)

        scene_aurocs = {}
        for cname in ['rain', 'glare', 'frost', 'motion_blur', 'shadow']:
            cfn = corruption_fns[cname]
            img = cfn(scene_img, 0.5)
            emb = extract_hidden(model, processor, img, prompt)
            d = cosine_dist(scene_clean, emb)
            scene_aurocs[cname] = float(d)

        scene_results[f"seed_{seed}"] = scene_aurocs
        print(f"  Scene {seed}: {scene_aurocs}")

    results['multi_scene'] = scene_results

    # ========== Composite real-world ==========
    print("\n=== Composite Real-World Scenarios ===")
    composite_results = {}

    # Scenario 1: Rain + night (reduced visibility)
    img = apply_rain(base_img, 0.5)
    arr = np.array(img).astype(float) * 0.3  # darken significantly
    comp1 = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    d1 = cosine_dist(clean_emb, extract_hidden(model, processor, comp1, prompt))
    composite_results['rain_night'] = float(d1)

    # Scenario 2: Glare + motion blur (driving into sun)
    img = apply_glare(base_img, 0.7)
    img = apply_motion_blur(img, 0.3)
    d2 = cosine_dist(clean_emb, extract_hidden(model, processor, img, prompt))
    composite_results['glare_motion'] = float(d2)

    # Scenario 3: Frost + shadow (early morning)
    img = apply_frost(base_img, 0.5)
    img = apply_shadow(img, 0.3)
    d3 = cosine_dist(clean_emb, extract_hidden(model, processor, img, prompt))
    composite_results['frost_shadow'] = float(d3)

    # Scenario 4: Dust + warm (desert conditions)
    img = apply_dust(base_img, 0.5)
    img = apply_warm_temp(img, 0.5)
    d4 = cosine_dist(clean_emb, extract_hidden(model, processor, img, prompt))
    composite_results['dust_warm'] = float(d4)

    results['composite'] = composite_results
    print(f"  rain_night: d={d1:.6f}")
    print(f"  glare_motion: d={d2:.6f}")
    print(f"  frost_shadow: d={d3:.6f}")
    print(f"  dust_warm: d={d4:.6f}")

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/realworld_sim_{ts}.json"

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
