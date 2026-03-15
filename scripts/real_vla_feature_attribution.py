#!/usr/bin/env python3
"""Experiment 379: Feature Attribution for Corruption Detection

Which input features does the detector rely on most?
1. Occlusion sensitivity: mask regions and measure detection change
2. Channel importance: R, G, B individual contributions
3. Frequency band analysis: low vs high frequency components
4. Spatial uniformity: half-image corruption detection
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
    ctypes = ['fog', 'night', 'noise', 'blur']

    print("Generating images...")
    seeds = list(range(0, 500, 100))[:5]
    images = {}
    clean_embs = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        px = rng.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        images[seed] = Image.fromarray(px)
        clean_embs[seed] = extract_hidden(model, processor, images[seed], prompt)

    centroid = np.mean(list(clean_embs.values()), axis=0)

    # ========== 1. Occlusion Sensitivity ==========
    print("\n=== Occlusion Sensitivity ===")

    occlusion = {}
    patch_size = 32
    stride = 32

    for ct in ctypes:
        seed = seeds[0]
        corrupt_img = apply_corruption(images[seed], ct, 0.5)
        corrupt_arr = np.array(corrupt_img)
        base_emb = extract_hidden(model, processor, corrupt_img, prompt)
        base_dist = cosine_dist(base_emb, centroid)

        heatmap = np.zeros((224 // stride, 224 // stride))
        for gi, y in enumerate(range(0, 224 - patch_size + 1, stride)):
            for gj, x in enumerate(range(0, 224 - patch_size + 1, stride)):
                masked = corrupt_arr.copy()
                masked[y:y+patch_size, x:x+patch_size] = 128
                masked_img = Image.fromarray(masked)
                emb = extract_hidden(model, processor, masked_img, prompt)
                d = cosine_dist(emb, centroid)
                heatmap[gi, gj] = base_dist - d

        occlusion[ct] = {
            'heatmap': heatmap.tolist(),
            'base_dist': float(base_dist),
            'max_importance': float(np.max(heatmap)),
            'min_importance': float(np.min(heatmap)),
            'center_importance': float(heatmap[3, 3]),
            'corner_importance': float(np.mean([heatmap[0,0], heatmap[0,-1], heatmap[-1,0], heatmap[-1,-1]])),
        }
        print(f"  {ct}: max_imp={np.max(heatmap):.6f}, center={heatmap[3,3]:.6f}")

    results['occlusion'] = occlusion

    # ========== 2. Channel Importance ==========
    print("\n=== Channel Importance ===")

    channel_imp = {}
    channel_names = ['red', 'green', 'blue']

    for ct in ctypes:
        seed = seeds[0]
        corrupt_img = apply_corruption(images[seed], ct, 0.5)
        corrupt_arr = np.array(corrupt_img).copy()
        base_emb = extract_hidden(model, processor, corrupt_img, prompt)
        base_dist = cosine_dist(base_emb, centroid)

        per_channel = {}
        for c, cname in enumerate(channel_names):
            zeroed = corrupt_arr.copy()
            zeroed[:, :, c] = 128
            zeroed_img = Image.fromarray(zeroed)
            emb = extract_hidden(model, processor, zeroed_img, prompt)
            d = cosine_dist(emb, centroid)

            per_channel[cname] = {
                'dist_without': float(d),
                'dist_change': float(d - base_dist),
                'relative_change': float((d - base_dist) / max(base_dist, 1e-10)),
            }

        channel_imp[ct] = per_channel
        print(f"  {ct}: R={per_channel['red']['relative_change']:.4f}, "
              f"G={per_channel['green']['relative_change']:.4f}, "
              f"B={per_channel['blue']['relative_change']:.4f}")

    results['channel_importance'] = channel_imp

    # ========== 3. Frequency Analysis ==========
    print("\n=== Frequency Analysis ===")

    freq_analysis = {}
    for ct in ctypes:
        seed = seeds[0]
        corrupt_arr = np.array(apply_corruption(images[seed], ct, 0.5)).astype(np.float32)
        clean_arr = np.array(images[seed]).astype(np.float32)

        diff = corrupt_arr - clean_arr
        diff_fft = np.fft.fft2(diff[:,:,0])
        diff_magnitude = np.abs(np.fft.fftshift(diff_fft))

        cy, cx = 112, 112
        y_grid, x_grid = np.ogrid[:224, :224]
        dist_from_center = np.sqrt((y_grid - cy)**2 + (x_grid - cx)**2)

        low_freq_mask = dist_from_center < 30
        mid_freq_mask = (dist_from_center >= 30) & (dist_from_center < 80)
        high_freq_mask = dist_from_center >= 80

        total_energy = float(np.sum(diff_magnitude**2))
        low_freq_energy = float(np.sum(diff_magnitude[low_freq_mask]**2)) / max(total_energy, 1e-10)
        mid_freq_energy = float(np.sum(diff_magnitude[mid_freq_mask]**2)) / max(total_energy, 1e-10)
        high_freq_energy = float(np.sum(diff_magnitude[high_freq_mask]**2)) / max(total_energy, 1e-10)

        freq_analysis[ct] = {
            'low_freq_energy': low_freq_energy,
            'mid_freq_energy': mid_freq_energy,
            'high_freq_energy': high_freq_energy,
        }
        print(f"  {ct}: low={low_freq_energy:.4f}, mid={mid_freq_energy:.4f}, "
              f"high={high_freq_energy:.4f}")

    results['frequency'] = freq_analysis

    # ========== 4. Spatial Uniformity ==========
    print("\n=== Spatial Uniformity ===")

    spatial = {}
    for ct in ctypes:
        seed = seeds[0]
        arr = np.array(images[seed]).copy()
        corrupt_arr = np.array(apply_corruption(images[seed], ct, 0.5))

        halves = {}
        for name, slc in [('left', (slice(None), slice(None, 112))),
                          ('right', (slice(None), slice(112, None))),
                          ('top', (slice(None, 112), slice(None))),
                          ('bottom', (slice(112, None), slice(None)))]:
            half = arr.copy()
            half[slc] = corrupt_arr[slc]
            emb = extract_hidden(model, processor, Image.fromarray(half), prompt)
            halves[name] = cosine_dist(emb, centroid)

        full_dist = cosine_dist(extract_hidden(model, processor, Image.fromarray(corrupt_arr), prompt), centroid)

        spatial[ct] = {
            **{f'{k}_dist': float(v) for k, v in halves.items()},
            'full_dist': float(full_dist),
            'half_to_full_ratio': float(np.mean(list(halves.values())) / max(full_dist, 1e-10)),
            'lr_asymmetry': float(abs(halves['left'] - halves['right']) / max(halves['left'] + halves['right'], 1e-10)),
            'tb_asymmetry': float(abs(halves['top'] - halves['bottom']) / max(halves['top'] + halves['bottom'], 1e-10)),
        }
        print(f"  {ct}: L={halves['left']:.6f}, R={halves['right']:.6f}, "
              f"T={halves['top']:.6f}, B={halves['bottom']:.6f}, full={full_dist:.6f}")

    results['spatial'] = spatial

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/feature_attribution_{ts}.json"
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
