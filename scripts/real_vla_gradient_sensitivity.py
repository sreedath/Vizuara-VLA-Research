#!/usr/bin/env python3
"""Experiment 364: Embedding Gradient Sensitivity Analysis

How sensitive are embeddings to pixel-level changes?
1. Finite-difference Jacobian norm per corruption direction
2. Directional sensitivity: which pixel perturbations move embeddings most?
3. Embedding stability under random pixel flips
4. Channel (R/G/B) sensitivity comparison
5. Spatial sensitivity map (which image regions matter most?)
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

    # Generate test images
    print("Generating images...")
    seeds = list(range(0, 500, 100))[:5]
    images = {}
    clean_embs = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        px = rng.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        images[seed] = Image.fromarray(px)
        clean_embs[seed] = extract_hidden(model, processor, images[seed], prompt)

    # ========== 1. Finite-Difference Jacobian Norm ==========
    print("\n=== Finite-Difference Jacobian Norm ===")

    eps_values = [1, 2, 5, 10, 20, 50]
    jacobian_norms = {}

    for seed in seeds[:3]:
        arr = np.array(images[seed]).copy()
        base_emb = clean_embs[seed]

        for eps in eps_values:
            perturbed = np.clip(arr.astype(np.int16) + eps, 0, 255).astype(np.uint8)
            pert_img = Image.fromarray(perturbed)
            pert_emb = extract_hidden(model, processor, pert_img, prompt)

            emb_diff = np.linalg.norm(pert_emb - base_emb)
            cos_d = cosine_dist(pert_emb, base_emb)

            key = f"seed{seed}_eps{eps}"
            jacobian_norms[key] = {
                'eps': eps,
                'l2_diff': float(emb_diff),
                'cosine_dist': float(cos_d),
                'ratio_l2_per_eps': float(emb_diff / eps),
            }

        print(f"  seed={seed}: L2@eps=[1,5,20,50] = "
              f"[{jacobian_norms[f'seed{seed}_eps1']['l2_diff']:.4f}, "
              f"{jacobian_norms[f'seed{seed}_eps5']['l2_diff']:.4f}, "
              f"{jacobian_norms[f'seed{seed}_eps20']['l2_diff']:.4f}, "
              f"{jacobian_norms[f'seed{seed}_eps50']['l2_diff']:.4f}]")

    results['jacobian_norms'] = jacobian_norms

    # ========== 2. Channel Sensitivity ==========
    print("\n=== Channel Sensitivity (R/G/B) ===")

    channel_sens = {}
    for seed in seeds[:3]:
        arr = np.array(images[seed]).copy()
        base_emb = clean_embs[seed]

        for ch_idx, ch_name in enumerate(['R', 'G', 'B']):
            perturbed = arr.copy()
            perturbed[:, :, ch_idx] = np.clip(perturbed[:, :, ch_idx].astype(np.int16) + 30, 0, 255).astype(np.uint8)
            pert_img = Image.fromarray(perturbed)
            pert_emb = extract_hidden(model, processor, pert_img, prompt)

            channel_sens[f"seed{seed}_{ch_name}"] = {
                'channel': ch_name,
                'l2_diff': float(np.linalg.norm(pert_emb - base_emb)),
                'cosine_dist': float(cosine_dist(pert_emb, base_emb)),
            }

        r = channel_sens[f"seed{seed}_R"]['cosine_dist']
        g = channel_sens[f"seed{seed}_G"]['cosine_dist']
        b = channel_sens[f"seed{seed}_B"]['cosine_dist']
        print(f"  seed={seed}: R={r:.6f}, G={g:.6f}, B={b:.6f}")

    results['channel_sensitivity'] = channel_sens

    # ========== 3. Spatial Sensitivity Map ==========
    print("\n=== Spatial Sensitivity Map ===")

    spatial = {}
    seed = seeds[0]
    arr = np.array(images[seed]).copy()
    base_emb = clean_embs[seed]

    grid_size = 7
    patch_size = 32
    for gy in range(grid_size):
        for gx in range(grid_size):
            y0, y1 = gy * patch_size, min((gy + 1) * patch_size, 224)
            x0, x1 = gx * patch_size, min((gx + 1) * patch_size, 224)

            perturbed = arr.copy()
            perturbed[y0:y1, x0:x1] = np.clip(
                perturbed[y0:y1, x0:x1].astype(np.int16) + 50, 0, 255
            ).astype(np.uint8)
            pert_img = Image.fromarray(perturbed)
            pert_emb = extract_hidden(model, processor, pert_img, prompt)

            key = f"{gy}_{gx}"
            spatial[key] = {
                'row': gy, 'col': gx,
                'l2_diff': float(np.linalg.norm(pert_emb - base_emb)),
                'cosine_dist': float(cosine_dist(pert_emb, base_emb)),
            }

    most_sens = max(spatial.items(), key=lambda x: x[1]['cosine_dist'])
    least_sens = min(spatial.items(), key=lambda x: x[1]['cosine_dist'])
    all_dists = [v['cosine_dist'] for v in spatial.values()]
    print(f"  Most sensitive patch: ({most_sens[1]['row']},{most_sens[1]['col']}) "
          f"cos_dist={most_sens[1]['cosine_dist']:.6f}")
    print(f"  Least sensitive patch: ({least_sens[1]['row']},{least_sens[1]['col']}) "
          f"cos_dist={least_sens[1]['cosine_dist']:.6f}")
    print(f"  Range: {min(all_dists):.6f} to {max(all_dists):.6f}, "
          f"ratio={max(all_dists)/(min(all_dists)+1e-12):.1f}x")

    results['spatial_sensitivity'] = spatial

    # ========== 4. Random Pixel Flip Stability ==========
    print("\n=== Random Pixel Flip Stability ===")

    flip_stability = {}
    for seed in seeds[:3]:
        arr = np.array(images[seed]).copy()
        base_emb = clean_embs[seed]

        for n_flips_frac in [0.001, 0.005, 0.01, 0.05, 0.1, 0.25]:
            n_pixels = arr.shape[0] * arr.shape[1]
            n_flips = int(n_pixels * n_flips_frac)

            rng = np.random.RandomState(seed + 999)
            perturbed = arr.copy()
            for _ in range(n_flips):
                y = rng.randint(0, 224)
                x = rng.randint(0, 224)
                perturbed[y, x] = rng.randint(0, 256, 3, dtype=np.uint8)

            pert_img = Image.fromarray(perturbed)
            pert_emb = extract_hidden(model, processor, pert_img, prompt)

            key = f"seed{seed}_frac{n_flips_frac}"
            flip_stability[key] = {
                'fraction': n_flips_frac,
                'n_flips': n_flips,
                'l2_diff': float(np.linalg.norm(pert_emb - base_emb)),
                'cosine_dist': float(cosine_dist(pert_emb, base_emb)),
            }

        fracs = [0.001, 0.01, 0.1, 0.25]
        dists = [flip_stability[f"seed{seed}_frac{f}"]['cosine_dist'] for f in fracs]
        print(f"  seed={seed}: cos_dist@[0.1%,1%,10%,25%] = "
              f"[{', '.join(f'{d:.6f}' for d in dists)}]")

    results['pixel_flip_stability'] = flip_stability

    # ========== 5. Corruption Direction Sensitivity ==========
    print("\n=== Corruption Direction Sensitivity ===")

    direction_sens = {}
    ctypes = ['fog', 'night', 'noise', 'blur']

    for seed in seeds[:3]:
        base_emb = clean_embs[seed]

        for ct in ctypes:
            for sev in [0.01, 0.02, 0.05]:
                corrupt_img = apply_corruption(images[seed], ct, sev)
                corrupt_emb = extract_hidden(model, processor, corrupt_img, prompt)

                l2 = float(np.linalg.norm(corrupt_emb - base_emb))
                cos_d = float(cosine_dist(corrupt_emb, base_emb))

                clean_arr = np.array(images[seed]).astype(np.float32)
                corrupt_arr = np.array(corrupt_img).astype(np.float32)
                pixel_l2 = float(np.linalg.norm(corrupt_arr - clean_arr))

                key = f"seed{seed}_{ct}_sev{sev}"
                direction_sens[key] = {
                    'corruption': ct,
                    'severity': sev,
                    'emb_l2': l2,
                    'emb_cosine': cos_d,
                    'pixel_l2': pixel_l2,
                    'sensitivity': float(l2 / (pixel_l2 + 1e-10)),
                }

        for ct in ctypes:
            k = f"seed{seed}_{ct}_sev0.05"
            if k in direction_sens:
                s = direction_sens[k]
                print(f"  seed={seed}, {ct}: emb_L2={s['emb_l2']:.4f}, "
                      f"pixel_L2={s['pixel_l2']:.1f}, sensitivity={s['sensitivity']:.6f}")

    results['direction_sensitivity'] = direction_sens

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/gradient_sensitivity_{ts}.json"
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
