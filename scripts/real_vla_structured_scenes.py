#!/usr/bin/env python3
"""Experiment 354: Structured Scene Robustness

Test detection on structured images that resemble real-world scenes:
1. Gradient backgrounds (sky-ground transitions)
2. Checkerboard patterns (texture-rich)
3. Face-like patterns (eyes, nose, mouth structures)
4. Grid/line patterns (urban environments)
5. Color patches (object segmentation boundaries)
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

def compute_auroc(id_scores, ood_scores):
    id_s = np.asarray(id_scores)
    ood_s = np.asarray(ood_scores)
    n_id, n_ood = len(id_s), len(ood_s)
    if n_id == 0 or n_ood == 0:
        return 0.5
    count = sum(float(np.sum(o > id_s) + 0.5 * np.sum(o == id_s)) for o in ood_s)
    return count / (n_id * n_ood)

def create_gradient_scene(seed):
    """Sky-ground gradient with horizon line."""
    rng = np.random.RandomState(seed)
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    horizon = 80 + rng.randint(-20, 20)
    # Sky gradient
    for y in range(horizon):
        t = y / horizon
        img[y] = [int(135 + 50*t), int(206 - 50*t), int(235 - 30*t)]
    # Ground
    ground_color = rng.randint(40, 120, 3)
    img[horizon:] = ground_color
    # Random objects
    for _ in range(3):
        x, y = rng.randint(0, 200), rng.randint(horizon, 200)
        c = rng.randint(50, 255, 3)
        img[y:y+20, x:x+20] = c
    return Image.fromarray(img)

def create_checkerboard(seed):
    """Checkerboard pattern at various scales."""
    rng = np.random.RandomState(seed)
    size = rng.choice([8, 16, 32])
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    c1 = rng.randint(20, 120, 3)
    c2 = rng.randint(130, 255, 3)
    for y in range(224):
        for x in range(224):
            if ((y // size) + (x // size)) % 2:
                img[y, x] = c1
            else:
                img[y, x] = c2
    return Image.fromarray(img)

def create_urban_grid(seed):
    """Grid lines on solid background (road/building pattern)."""
    rng = np.random.RandomState(seed)
    bg = rng.randint(40, 100, 3)
    img = np.full((224, 224, 3), bg, dtype=np.uint8)
    line_color = rng.randint(150, 255, 3)
    # Vertical lines
    for x in range(0, 224, rng.randint(20, 50)):
        img[:, max(0,x-1):x+2] = line_color
    # Horizontal lines
    for y in range(0, 224, rng.randint(20, 50)):
        img[max(0,y-1):y+2, :] = line_color
    return Image.fromarray(img)

def create_color_patches(seed):
    """Random color patches (like segmented objects)."""
    rng = np.random.RandomState(seed)
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    n_patches = rng.randint(5, 15)
    for _ in range(n_patches):
        x1, y1 = rng.randint(0, 200), rng.randint(0, 200)
        w, h = rng.randint(20, 80), rng.randint(20, 80)
        color = rng.randint(30, 255, 3)
        img[y1:y1+h, x1:x1+w] = color
    return Image.fromarray(img)

def create_natural_texture(seed):
    """Noise with spatial correlation (natural texture approximation)."""
    rng = np.random.RandomState(seed)
    # Start with coarse noise, then upscale
    coarse = rng.randint(50, 200, (28, 28, 3), dtype=np.uint8)
    img = Image.fromarray(coarse).resize((224, 224), Image.BILINEAR)
    return img

def main():
    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    results = {}
    ctypes = ['fog', 'night', 'noise', 'blur']

    scene_types = {
        'gradient': create_gradient_scene,
        'checkerboard': create_checkerboard,
        'urban_grid': create_urban_grid,
        'color_patches': create_color_patches,
        'natural_texture': create_natural_texture,
    }

    # Also include random pixels for comparison
    def create_random(seed):
        rng = np.random.RandomState(seed)
        px = rng.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        return Image.fromarray(px)
    scene_types['random_pixels'] = create_random

    n_scenes = 5  # per scene type

    # ========== 1. Per-scene-type detection ==========
    print("\n=== Per-Scene-Type Detection ===")

    all_results = {}
    for stype, create_fn in scene_types.items():
        print(f"\n  Scene type: {stype}")

        id_dists = []
        ood_dists = {ct: [] for ct in ctypes}
        emb_dists = {ct: [] for ct in ctypes}

        for i in range(n_scenes):
            seed = 42 + i * 100
            img = create_fn(seed)
            cal = extract_hidden(model, processor, img, prompt)

            # ID: re-embed same image
            emb = extract_hidden(model, processor, img, prompt)
            id_dists.append(float(cosine_dist(cal, emb)))

            # OOD
            for ct in ctypes:
                corrupted = apply_corruption(img, ct, 0.5)
                emb = extract_hidden(model, processor, corrupted, prompt)
                d = float(cosine_dist(cal, emb))
                ood_dists[ct].append(d)
                emb_dists[ct].append(d)

        per_type = {}
        for ct in ctypes:
            auroc = compute_auroc(id_dists, ood_dists[ct])
            per_type[ct] = {
                'auroc': float(auroc),
                'mean_dist': float(np.mean(ood_dists[ct])),
                'min_dist': float(min(ood_dists[ct])),
                'all_detected': all(d > max(id_dists) for d in ood_dists[ct]),
            }

        all_results[stype] = {
            'id_mean': float(np.mean(id_dists)),
            'id_max': float(max(id_dists)),
            'per_type': per_type,
            'all_perfect': all(per_type[ct]['auroc'] == 1.0 for ct in ctypes),
        }

        auroc_str = ', '.join(ct + '=' + format(per_type[ct]['auroc'], '.3f') for ct in ctypes)
        print(f"    AUROCs: {auroc_str}")

    results['per_scene_type'] = all_results

    # ========== 2. Cross-scene-type robustness ==========
    print("\n=== Cross-Scene-Type Analysis ===")

    # How does distance magnitude vary across scene types?
    cross_type = {}
    for ct in ctypes:
        means = {stype: all_results[stype]['per_type'][ct]['mean_dist'] for stype in scene_types}
        max_ratio = max(means.values()) / min(means.values()) if min(means.values()) > 0 else float('inf')
        cross_type[ct] = {
            'per_scene_type': means,
            'max_min_ratio': float(max_ratio) if not np.isinf(max_ratio) else 'infinite',
            'most_sensitive': max(means, key=means.get),
            'least_sensitive': min(means, key=means.get),
        }
        print(f"  {ct}: most_sensitive={max(means, key=means.get)}, "
              f"least={min(means, key=means.get)}, ratio={max_ratio:.2f}x")

    results['cross_type'] = cross_type

    # ========== 3. Severity curves per scene type ==========
    print("\n=== Severity Curves ===")

    severity_curves = {}
    for stype in ['gradient', 'random_pixels']:
        create_fn = scene_types[stype]
        img = create_fn(42)
        cal = extract_hidden(model, processor, img, prompt)

        for ct in ['fog', 'blur']:
            curve = []
            for sev in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
                corrupted = apply_corruption(img, ct, sev)
                emb = extract_hidden(model, processor, corrupted, prompt)
                d = float(cosine_dist(cal, emb))
                curve.append({'severity': float(sev), 'distance': d})

            key = f"{stype}_{ct}"
            severity_curves[key] = curve
            print(f"  {key}: min_d={curve[0]['distance']:.6f}, max_d={curve[-1]['distance']:.6f}")

    results['severity_curves'] = severity_curves

    # ========== 4. Summary statistics ==========
    print("\n=== Summary ===")

    n_perfect = sum(1 for s in all_results.values() if s['all_perfect'])
    n_total = len(all_results)
    total_aurocs = []
    for s in all_results.values():
        for ct in ctypes:
            total_aurocs.append(s['per_type'][ct]['auroc'])

    summary = {
        'n_scene_types': n_total,
        'n_perfect': n_perfect,
        'total_tests': len(total_aurocs),
        'mean_auroc': float(np.mean(total_aurocs)),
        'min_auroc': float(min(total_aurocs)),
        'all_perfect': n_perfect == n_total,
    }
    print(f"  {n_perfect}/{n_total} scene types all-perfect")
    print(f"  Mean AUROC: {np.mean(total_aurocs):.4f}")
    print(f"  Min AUROC: {min(total_aurocs):.4f}")

    results['summary'] = summary

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/structured_scenes_{ts}.json"
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
