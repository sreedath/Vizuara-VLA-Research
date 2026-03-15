#!/usr/bin/env python3
"""Experiment 369: Corruption Interpolation Analysis

How do embeddings behave when smoothly transitioning between corruptions?
1. Linear interpolation between clean and corrupted images
2. Embedding path linearity (geodesic vs straight-line)
3. Detection threshold crossing point
4. Inter-corruption interpolation paths
5. Convexity of the detection boundary
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

def interpolate_images(img1, img2, alpha):
    """Pixel-level linear interpolation between two images."""
    arr1 = np.array(img1).astype(np.float32)
    arr2 = np.array(img2).astype(np.float32)
    blended = (1 - alpha) * arr1 + alpha * arr2
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

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

    centroid = np.mean(list(clean_embs.values()), axis=0)

    # ========== 1. Clean-to-Corrupt Interpolation ==========
    print("\n=== Clean-to-Corrupt Interpolation ===")

    interp_results = {}
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for ct in ctypes:
        per_alpha = {}
        for alpha in alphas:
            dists = []
            for seed in seeds[:3]:
                corrupt_img = apply_corruption(images[seed], ct, 1.0)
                interp_img = interpolate_images(images[seed], corrupt_img, alpha)
                emb = extract_hidden(model, processor, interp_img, prompt)
                d = cosine_dist(emb, centroid)
                dists.append(d)

            per_alpha[str(alpha)] = {
                'mean_dist': float(np.mean(dists)),
                'std_dist': float(np.std(dists)),
            }

        interp_results[ct] = per_alpha

        # Report linearity
        dists_at = [per_alpha[str(a)]['mean_dist'] for a in alphas]
        d0, d1 = dists_at[0], dists_at[-1]
        linear_pred = [d0 + (d1 - d0) * a for a in alphas]
        residuals = [abs(dists_at[i] - linear_pred[i]) for i in range(len(alphas))]
        max_residual = max(residuals)
        linearity = 1 - max_residual / (abs(d1 - d0) + 1e-10)
        print(f"  {ct}: dist@0={d0:.6f}, dist@1={d1:.6f}, linearity={linearity:.4f}")

    results['clean_corrupt_interp'] = interp_results

    # ========== 2. Embedding Path Linearity ==========
    print("\n=== Embedding Path Linearity ===")

    linearity_results = {}
    for ct in ctypes:
        path_lengths = []
        straight_lengths = []

        for seed in seeds[:3]:
            # Get embeddings along interpolation path
            path_embs = []
            for alpha in alphas:
                corrupt_img = apply_corruption(images[seed], ct, 1.0)
                interp_img = interpolate_images(images[seed], corrupt_img, alpha)
                emb = extract_hidden(model, processor, interp_img, prompt)
                path_embs.append(emb)

            # Path length (sum of consecutive L2 distances)
            path_len = sum(np.linalg.norm(path_embs[i+1] - path_embs[i])
                          for i in range(len(path_embs)-1))

            # Straight-line distance
            straight_len = float(np.linalg.norm(path_embs[-1] - path_embs[0]))

            path_lengths.append(path_len)
            straight_lengths.append(straight_len)

        ratio = np.mean(path_lengths) / (np.mean(straight_lengths) + 1e-10)
        linearity_results[ct] = {
            'mean_path_length': float(np.mean(path_lengths)),
            'mean_straight_length': float(np.mean(straight_lengths)),
            'path_to_straight_ratio': float(ratio),
            'is_nearly_linear': float(ratio) < 1.05,
        }
        print(f"  {ct}: path/straight={ratio:.4f} ({'linear' if ratio < 1.05 else 'curved'})")

    results['path_linearity'] = linearity_results

    # ========== 3. Detection Threshold Crossing ==========
    print("\n=== Detection Threshold Crossing ===")

    # Threshold = max clean distance to centroid
    clean_dists = [cosine_dist(clean_embs[s], centroid) for s in seeds]
    threshold = max(clean_dists)

    crossing_results = {}
    for ct in ctypes:
        crossing_alphas = []
        for seed in seeds[:3]:
            corrupt_img = apply_corruption(images[seed], ct, 1.0)
            prev_below = True
            for alpha in np.linspace(0, 1, 21):
                interp_img = interpolate_images(images[seed], corrupt_img, alpha)
                emb = extract_hidden(model, processor, interp_img, prompt)
                d = cosine_dist(emb, centroid)
                if d > threshold and prev_below:
                    crossing_alphas.append(float(alpha))
                    break
                prev_below = d <= threshold

            if len(crossing_alphas) < len(seeds[:3]):
                # Might not cross for noise
                pass

        crossing_results[ct] = {
            'mean_crossing_alpha': float(np.mean(crossing_alphas)) if crossing_alphas else None,
            'min_crossing_alpha': float(min(crossing_alphas)) if crossing_alphas else None,
            'n_crossed': len(crossing_alphas),
            'threshold': float(threshold),
        }
        if crossing_alphas:
            print(f"  {ct}: crosses at alpha={np.mean(crossing_alphas):.3f} "
                  f"(min={min(crossing_alphas):.3f})")
        else:
            print(f"  {ct}: never crosses threshold")

    results['threshold_crossing'] = crossing_results

    # ========== 4. Inter-Corruption Interpolation ==========
    print("\n=== Inter-Corruption Interpolation ===")

    inter_corrupt = {}
    seed = seeds[0]
    for i, ct1 in enumerate(ctypes):
        for ct2 in ctypes[i+1:]:
            corrupt1 = apply_corruption(images[seed], ct1, 0.5)
            corrupt2 = apply_corruption(images[seed], ct2, 0.5)
            emb1 = extract_hidden(model, processor, corrupt1, prompt)
            emb2 = extract_hidden(model, processor, corrupt2, prompt)

            interp_dists = []
            for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
                interp_img = interpolate_images(corrupt1, corrupt2, alpha)
                emb = extract_hidden(model, processor, interp_img, prompt)
                d = cosine_dist(emb, centroid)
                interp_dists.append(float(d))

            # Check if midpoint is closer or farther than endpoints
            endpoint_mean = (interp_dists[0] + interp_dists[-1]) / 2
            midpoint = interp_dists[2]
            convexity = midpoint - endpoint_mean

            key = f"{ct1}_to_{ct2}"
            inter_corrupt[key] = {
                'distances': interp_dists,
                'midpoint_dist': midpoint,
                'endpoint_mean': endpoint_mean,
                'convexity': float(convexity),
            }
            print(f"  {ct1}↔{ct2}: midpoint={midpoint:.6f}, "
                  f"endpoint_mean={endpoint_mean:.6f}, "
                  f"{'concave' if convexity < 0 else 'convex'}")

    results['inter_corruption_interp'] = inter_corrupt

    # ========== 5. Monotonicity of Severity ==========
    print("\n=== Severity Monotonicity ===")

    monotonicity = {}
    for ct in ctypes:
        for seed in seeds[:3]:
            sevs = np.linspace(0, 1, 21)
            dists = []
            for sev in sevs:
                corrupt_img = apply_corruption(images[seed], ct, sev)
                emb = extract_hidden(model, processor, corrupt_img, prompt)
                d = cosine_dist(emb, centroid)
                dists.append(float(d))

            # Check monotonicity
            n_increases = sum(1 for i in range(len(dists)-1) if dists[i+1] > dists[i])
            n_decreases = sum(1 for i in range(len(dists)-1) if dists[i+1] < dists[i])
            is_monotone = n_decreases == 0

            key = f"{ct}_seed{seed}"
            monotonicity[key] = {
                'n_increases': n_increases,
                'n_decreases': n_decreases,
                'is_monotone': is_monotone,
                'dists': dists,
            }

        mono_count = sum(1 for s in seeds[:3]
                        if monotonicity[f"{ct}_seed{s}"]['is_monotone'])
        print(f"  {ct}: {mono_count}/3 seeds monotonically increasing")

    results['monotonicity'] = monotonicity

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/corruption_interpolation_{ts}.json"
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
