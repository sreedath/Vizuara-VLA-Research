#!/usr/bin/env python3
"""Experiment 411: Feature Attribution Analysis

Investigates which input regions and token positions contribute most to the
corruption detection signal. Uses gradient-free attribution methods to
understand what the detector "looks at."

Tests:
1. Spatial attribution: occlude image regions and measure detection signal change
2. Frequency-band attribution: which spatial frequencies carry corruption signal?
3. Channel attribution: RGB channel importance for detection
4. Cumulative patch occlusion: how does detection scale with clean area fraction?
5. Fine-grained 8x8 spatial attribution for fog
"""

import json, time, os, sys
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageDraw
from transformers import AutoModelForVision2Seq, AutoProcessor

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

def extract_hidden(model, processor, image, prompt, layer=3):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return fwd.hidden_states[layer][0, -1, :].float().cpu().numpy()

def cosine_dist(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    return 1.0 - np.dot(a, b) / (na * nb)

def main():
    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    img = Image.fromarray(np.random.RandomState(42).randint(0, 255, (224, 224, 3), dtype=np.uint8))

    # Get clean and corrupted baselines
    print("Computing baselines...")
    clean_emb = extract_hidden(model, processor, img, prompt)
    corruptions = ['fog', 'night', 'noise', 'blur']
    corrupt_embs = {}
    base_dists = {}
    for c in corruptions:
        corrupt_embs[c] = extract_hidden(model, processor, apply_corruption(img, c), prompt)
        base_dists[c] = cosine_dist(corrupt_embs[c], clean_emb)
        print(f"  {c} baseline dist: {base_dists[c]:.6f}")

    results = {"base_dists": {k: float(v) for k, v in base_dists.items()}}

    # === Test 1: Spatial Occlusion Attribution (4x4) ===
    print("\n=== Spatial Occlusion (4x4 grid) ===")
    grid_size = 4
    patch_h = 224 // grid_size
    patch_w = 224 // grid_size

    spatial_attr = {}
    for c in corruptions:
        corrupted = apply_corruption(img, c)
        attr_grid = np.zeros((grid_size, grid_size))

        for gi in range(grid_size):
            for gj in range(grid_size):
                occluded = np.array(corrupted).copy()
                clean_arr = np.array(img)
                y0, y1 = gi * patch_h, (gi + 1) * patch_h
                x0, x1 = gj * patch_w, (gj + 1) * patch_w
                occluded[y0:y1, x0:x1] = clean_arr[y0:y1, x0:x1]

                occluded_img = Image.fromarray(occluded)
                occluded_emb = extract_hidden(model, processor, occluded_img, prompt)
                occluded_dist = cosine_dist(occluded_emb, clean_emb)
                attr_grid[gi, gj] = base_dists[c] - occluded_dist

        spatial_attr[c] = {
            "grid": attr_grid.tolist(),
            "max_patch": [int(np.unravel_index(np.argmax(attr_grid), attr_grid.shape)[0]),
                         int(np.unravel_index(np.argmax(attr_grid), attr_grid.shape)[1])],
            "mean_attr": float(np.mean(attr_grid)),
            "std_attr": float(np.std(attr_grid)),
            "max_attr": float(np.max(attr_grid)),
            "uniformity": float(np.std(attr_grid) / max(np.mean(np.abs(attr_grid)), 1e-12))
        }
        print(f"  {c}: mean={np.mean(attr_grid):.6f}, std={np.std(attr_grid):.6f}, uniformity={spatial_attr[c]['uniformity']:.4f}")

    results["spatial_occlusion"] = spatial_attr

    # === Test 2: Fine-grained spatial (8x8) for fog ===
    print("\n=== Fine-grained Spatial (8x8, fog) ===")
    grid_size_8 = 8
    patch_h_8 = 224 // grid_size_8
    patch_w_8 = 224 // grid_size_8
    corrupted_fog = apply_corruption(img, 'fog')
    attr_grid_8 = np.zeros((grid_size_8, grid_size_8))

    for gi in range(grid_size_8):
        for gj in range(grid_size_8):
            occluded = np.array(corrupted_fog).copy()
            clean_arr = np.array(img)
            y0, y1 = gi * patch_h_8, (gi + 1) * patch_h_8
            x0, x1 = gj * patch_w_8, (gj + 1) * patch_w_8
            occluded[y0:y1, x0:x1] = clean_arr[y0:y1, x0:x1]
            occluded_img = Image.fromarray(occluded)
            occluded_emb = extract_hidden(model, processor, occluded_img, prompt)
            occluded_dist = cosine_dist(occluded_emb, clean_emb)
            attr_grid_8[gi, gj] = base_dists['fog'] - occluded_dist

    results["spatial_8x8_fog"] = {
        "grid": attr_grid_8.tolist(),
        "mean": float(np.mean(attr_grid_8)),
        "std": float(np.std(attr_grid_8)),
        "uniformity": float(np.std(attr_grid_8) / max(np.mean(np.abs(attr_grid_8)), 1e-12))
    }
    print(f"  8x8 fog: mean={np.mean(attr_grid_8):.6f}, std={np.std(attr_grid_8):.6f}")

    # === Test 3: Frequency Band Attribution ===
    print("\n=== Frequency Band Attribution ===")
    freq_results = {}
    for c in corruptions:
        corrupted = apply_corruption(img, c)
        corrupt_arr = np.array(corrupted).astype(np.float32)
        clean_arr = np.array(img).astype(np.float32)
        diff = corrupt_arr - clean_arr

        freq_bands = {}
        for ch in range(3):
            fft = np.fft.fft2(diff[:, :, ch])
            fft_shift = np.fft.fftshift(fft)
            h, w = fft_shift.shape
            cy, cx = h // 2, w // 2

            bands = {
                "low": (0, min(h, w) // 6),
                "mid": (min(h, w) // 6, min(h, w) // 3),
                "high": (min(h, w) // 3, min(h, w) // 2)
            }

            for band_name, (r_min, r_max) in bands.items():
                mask = np.zeros_like(fft_shift)
                y_coords, x_coords = np.ogrid[:h, :w]
                r = np.sqrt((y_coords - cy)**2 + (x_coords - cx)**2)
                mask[(r >= r_min) & (r < r_max)] = 1

                band_power = np.sum(np.abs(fft_shift * mask)**2)
                if band_name not in freq_bands:
                    freq_bands[band_name] = 0
                freq_bands[band_name] += band_power

        total_power = sum(freq_bands.values())
        freq_results[c] = {band: float(p / total_power) for band, p in freq_bands.items()}
        print(f"  {c}: low={freq_results[c]['low']:.3f}, mid={freq_results[c]['mid']:.3f}, high={freq_results[c]['high']:.3f}")

    results["frequency_attribution"] = freq_results

    # === Test 4: Channel Attribution ===
    print("\n=== Channel Attribution ===")
    channel_results = {}
    for c in corruptions:
        corrupted = apply_corruption(img, c)
        channel_attr = {}

        for ch_idx, ch_name in enumerate(['R', 'G', 'B']):
            hybrid = np.array(corrupted).copy()
            clean_arr = np.array(img)
            hybrid[:, :, ch_idx] = clean_arr[:, :, ch_idx]
            hybrid_img = Image.fromarray(hybrid)
            hybrid_emb = extract_hidden(model, processor, hybrid_img, prompt)
            hybrid_dist = cosine_dist(hybrid_emb, clean_emb)
            channel_attr[ch_name] = float(base_dists[c] - hybrid_dist)

        channel_results[c] = channel_attr
        print(f"  {c}: R={channel_attr['R']:.6f}, G={channel_attr['G']:.6f}, B={channel_attr['B']:.6f}")

    results["channel_attribution"] = channel_results

    # === Test 5: Cumulative Patch Occlusion ===
    print("\n=== Cumulative Patch Occlusion (fog) ===")
    cumulative = []
    for frac in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        corrupted = apply_corruption(img, 'fog')
        if frac > 0:
            occluded = np.array(corrupted).copy()
            clean_arr = np.array(img)
            side = int(224 * np.sqrt(frac))
            y0 = (224 - side) // 2
            x0 = (224 - side) // 2
            occluded[y0:y0+side, x0:x0+side] = clean_arr[y0:y0+side, x0:x0+side]
            corrupted = Image.fromarray(occluded)

        emb = extract_hidden(model, processor, corrupted, prompt)
        dist = cosine_dist(emb, clean_emb)
        cumulative.append({
            "fraction_clean": float(frac),
            "distance": float(dist),
            "relative_to_base": float(dist / base_dists['fog']) if base_dists['fog'] > 0 else 0
        })
        print(f"  {frac:.0%} clean: dist={dist:.6f} ({dist/base_dists['fog']*100:.1f}% of base)")

    results["cumulative_occlusion"] = cumulative

    # Save
    out_path = "/workspace/Vizuara-VLA-Research/experiments/feature_attribution_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
