#!/usr/bin/env python3
"""Experiment 367: Embedding Manifold Geometry

Characterize the geometric structure of clean vs corrupted embeddings:
1. Pairwise distance distributions (within-clean, within-corrupt, between)
2. Convex hull containment test
3. Corruption-specific embedding "direction" consistency
4. Angular distribution of corruption shifts
5. Embedding norm analysis
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

def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

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

    # Generate embeddings
    print("Generating embeddings...")
    seeds = list(range(0, 2000, 100))[:20]
    images = {}
    clean_embs = {}
    corrupt_embs = {ct: {} for ct in ctypes}

    for seed in seeds:
        rng = np.random.RandomState(seed)
        px = rng.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        images[seed] = Image.fromarray(px)
        clean_embs[seed] = extract_hidden(model, processor, images[seed], prompt)
        for ct in ctypes:
            corrupted = apply_corruption(images[seed], ct, 0.5)
            corrupt_embs[ct][seed] = extract_hidden(model, processor, corrupted, prompt)

    print(f"  {len(seeds)} scenes, embedding dim={clean_embs[seeds[0]].shape[0]}")

    # ========== 1. Pairwise Distance Distributions ==========
    print("\n=== Pairwise Distance Distributions ===")

    dist_distributions = {}

    # Within-clean distances
    clean_dists = []
    for i, s1 in enumerate(seeds):
        for s2 in seeds[i+1:]:
            clean_dists.append(cosine_dist(clean_embs[s1], clean_embs[s2]))

    dist_distributions['within_clean'] = {
        'mean': float(np.mean(clean_dists)),
        'std': float(np.std(clean_dists)),
        'min': float(min(clean_dists)),
        'max': float(max(clean_dists)),
    }
    print(f"  Within-clean: mean={np.mean(clean_dists):.6f}, range=[{min(clean_dists):.6f}, {max(clean_dists):.6f}]")

    for ct in ctypes:
        # Within-corrupt
        c_dists = []
        for i, s1 in enumerate(seeds):
            for s2 in seeds[i+1:]:
                c_dists.append(cosine_dist(corrupt_embs[ct][s1], corrupt_embs[ct][s2]))

        # Between clean and corrupt
        between_dists = []
        for s in seeds:
            between_dists.append(cosine_dist(clean_embs[s], corrupt_embs[ct][s]))

        dist_distributions[ct] = {
            'within_corrupt_mean': float(np.mean(c_dists)),
            'within_corrupt_std': float(np.std(c_dists)),
            'between_mean': float(np.mean(between_dists)),
            'between_std': float(np.std(between_dists)),
            'between_min': float(min(between_dists)),
            'between_max': float(max(between_dists)),
            'separation_ratio': float(np.mean(between_dists) / (np.mean(clean_dists) + 1e-12)),
        }
        print(f"  {ct}: within_corrupt={np.mean(c_dists):.6f}, "
              f"between={np.mean(between_dists):.6f}, "
              f"separation={dist_distributions[ct]['separation_ratio']:.1f}x")

    results['distance_distributions'] = dist_distributions

    # ========== 2. Corruption Shift Direction Consistency ==========
    print("\n=== Corruption Shift Direction Consistency ===")

    shift_consistency = {}
    for ct in ctypes:
        # Compute shift vectors
        shifts = []
        for s in seeds:
            shift = corrupt_embs[ct][s] - clean_embs[s]
            shifts.append(shift)

        # Pairwise cosine similarity of shift vectors
        shift_sims = []
        for i in range(len(shifts)):
            for j in range(i+1, len(shifts)):
                shift_sims.append(cosine_sim(shifts[i], shifts[j]))

        # Mean shift direction
        mean_shift = np.mean(shifts, axis=0)
        mean_shift_norm = np.linalg.norm(mean_shift)
        alignment_to_mean = [cosine_sim(s, mean_shift) for s in shifts]

        shift_consistency[ct] = {
            'mean_pairwise_sim': float(np.mean(shift_sims)),
            'min_pairwise_sim': float(min(shift_sims)),
            'std_pairwise_sim': float(np.std(shift_sims)),
            'mean_alignment': float(np.mean(alignment_to_mean)),
            'min_alignment': float(min(alignment_to_mean)),
            'mean_shift_norm': float(mean_shift_norm),
        }
        print(f"  {ct}: pairwise_sim={np.mean(shift_sims):.4f}, "
              f"alignment={np.mean(alignment_to_mean):.4f}")

    results['shift_consistency'] = shift_consistency

    # ========== 3. Cross-Corruption Angular Relationships ==========
    print("\n=== Cross-Corruption Angular Relationships ===")

    cross_angular = {}
    for i, ct1 in enumerate(ctypes):
        for ct2 in ctypes[i+1:]:
            sims = []
            for s in seeds:
                shift1 = corrupt_embs[ct1][s] - clean_embs[s]
                shift2 = corrupt_embs[ct2][s] - clean_embs[s]
                sims.append(cosine_sim(shift1, shift2))

            cross_angular[f"{ct1}_vs_{ct2}"] = {
                'mean_sim': float(np.mean(sims)),
                'std_sim': float(np.std(sims)),
            }
            print(f"  {ct1} vs {ct2}: shift_sim={np.mean(sims):.4f}")

    results['cross_angular'] = cross_angular

    # ========== 4. Embedding Norm Analysis ==========
    print("\n=== Embedding Norm Analysis ===")

    norm_analysis = {}
    clean_norms = [float(np.linalg.norm(clean_embs[s])) for s in seeds]
    norm_analysis['clean'] = {
        'mean_norm': float(np.mean(clean_norms)),
        'std_norm': float(np.std(clean_norms)),
        'min_norm': float(min(clean_norms)),
        'max_norm': float(max(clean_norms)),
    }
    print(f"  Clean: norm={np.mean(clean_norms):.4f} +/- {np.std(clean_norms):.4f}")

    for ct in ctypes:
        corrupt_norms = [float(np.linalg.norm(corrupt_embs[ct][s])) for s in seeds]
        norm_change = [(corrupt_norms[i] - clean_norms[i]) / clean_norms[i] * 100
                       for i in range(len(seeds))]
        norm_analysis[ct] = {
            'mean_norm': float(np.mean(corrupt_norms)),
            'std_norm': float(np.std(corrupt_norms)),
            'mean_change_pct': float(np.mean(norm_change)),
            'max_change_pct': float(max(abs(c) for c in norm_change)),
        }
        print(f"  {ct}: norm={np.mean(corrupt_norms):.4f}, "
              f"change={np.mean(norm_change):.2f}%")

    results['norm_analysis'] = norm_analysis

    # ========== 5. Projection onto Corruption Axes ==========
    print("\n=== Projection onto Corruption Axes ===")

    # Define corruption axes as mean shift directions
    axes_dict = {}
    for ct in ctypes:
        shifts = [corrupt_embs[ct][s] - clean_embs[s] for s in seeds]
        mean_shift = np.mean(shifts, axis=0)
        axes_dict[ct] = mean_shift / (np.linalg.norm(mean_shift) + 1e-12)

    projection = {}
    for ct in ctypes:
        # Project each corrupted embedding's shift onto each axis
        per_axis = {}
        for axis_ct in ctypes:
            projs = []
            for s in seeds:
                shift = corrupt_embs[ct][s] - clean_embs[s]
                proj = float(np.dot(shift, axes_dict[axis_ct]))
                projs.append(proj)
            per_axis[axis_ct] = {
                'mean_proj': float(np.mean(projs)),
                'std_proj': float(np.std(projs)),
            }

        projection[ct] = per_axis

    # Print as matrix
    for ct in ctypes:
        vals = [projection[ct][act]['mean_proj'] for act in ctypes]
        print(f"  {ct}: " + ', '.join(f'{act}={v:.4f}' for act, v in zip(ctypes, vals)))

    results['projection'] = projection

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/manifold_geometry_{ts}.json"
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
