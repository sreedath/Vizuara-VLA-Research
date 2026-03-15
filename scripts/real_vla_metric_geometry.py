#!/usr/bin/env python3
"""Experiment 353: Metric Space Geometry

Analyze geometric properties of the embedding space:
1. Triangle inequality verification for cosine distance
2. Metric space structure (symmetry, identity)
3. Embedding norm distribution (clean vs corrupted)
4. Angular distribution of corruption directions
5. Volume ratio: corruption hyperball vs clean region
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

    # Generate embeddings
    print("Generating embeddings...")
    seeds = list(range(0, 2000, 100))[:20]
    clean_embs = {}
    corrupt_embs = {ct: {} for ct in ctypes}

    for seed in seeds:
        rng = np.random.RandomState(seed)
        px = rng.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(px)
        clean_embs[seed] = extract_hidden(model, processor, img, prompt)

        for ct in ctypes:
            corrupted = apply_corruption(img, ct, 0.5)
            corrupt_embs[ct][seed] = extract_hidden(model, processor, corrupted, prompt)

    # ========== 1. Triangle Inequality ==========
    print("\n=== Triangle Inequality ===")

    triangle_tests = []
    # Test d(a,c) <= d(a,b) + d(b,c) for all triples
    test_seeds = seeds[:10]
    violations = 0
    total_tests = 0

    for s1 in test_seeds:
        for s2 in test_seeds:
            for s3 in test_seeds:
                if s1 >= s2 or s2 >= s3:
                    continue
                a = clean_embs[s1]
                b = clean_embs[s2]
                c = clean_embs[s3]

                d_ab = cosine_dist(a, b)
                d_bc = cosine_dist(b, c)
                d_ac = cosine_dist(a, c)

                holds = d_ac <= d_ab + d_bc + 1e-10
                if not holds:
                    violations += 1
                total_tests += 1

    # Also test with corrupted embeddings
    for s1 in test_seeds[:5]:
        for ct1 in ctypes:
            for ct2 in ctypes:
                if ct1 >= ct2:
                    continue
                a = clean_embs[s1]
                b = corrupt_embs[ct1][s1]
                c = corrupt_embs[ct2][s1]

                d_ab = cosine_dist(a, b)
                d_bc = cosine_dist(b, c)
                d_ac = cosine_dist(a, c)

                holds = d_ac <= d_ab + d_bc + 1e-10
                if not holds:
                    violations += 1
                total_tests += 1

    triangle_results = {
        'total_tests': total_tests,
        'violations': violations,
        'holds': violations == 0,
    }
    print(f"  {violations}/{total_tests} violations — triangle inequality {'HOLDS' if violations == 0 else 'VIOLATED'}")

    results['triangle_inequality'] = triangle_results

    # ========== 2. Embedding Norms ==========
    print("\n=== Embedding Norms ===")

    clean_norms = [float(np.linalg.norm(clean_embs[s])) for s in seeds]
    corrupt_norms = {}
    for ct in ctypes:
        norms = [float(np.linalg.norm(corrupt_embs[ct][s])) for s in seeds]
        corrupt_norms[ct] = {
            'mean': float(np.mean(norms)),
            'std': float(np.std(norms)),
            'min': float(min(norms)),
            'max': float(max(norms)),
            'ratio_to_clean': float(np.mean(norms) / np.mean(clean_norms)),
        }

    norm_results = {
        'clean': {
            'mean': float(np.mean(clean_norms)),
            'std': float(np.std(clean_norms)),
            'min': float(min(clean_norms)),
            'max': float(max(clean_norms)),
        },
        'corrupt': corrupt_norms,
    }
    print(f"  Clean: mean={np.mean(clean_norms):.4f}, std={np.std(clean_norms):.4f}")
    for ct in ctypes:
        cn = corrupt_norms[ct]
        print(f"  {ct}: mean={cn['mean']:.4f}, ratio={cn['ratio_to_clean']:.4f}")

    results['norms'] = norm_results

    # ========== 3. Corruption Direction Analysis ==========
    print("\n=== Corruption Directions ===")

    direction_results = {}
    for seed in seeds[:10]:
        clean = clean_embs[seed]
        directions = {}

        for ct in ctypes:
            corrupt = corrupt_embs[ct][seed]
            delta = corrupt - clean
            delta_norm = np.linalg.norm(delta)
            if delta_norm > 0:
                direction = delta / delta_norm
            else:
                direction = np.zeros_like(delta)
            directions[ct] = direction

        # Pairwise angles between corruption directions
        pairwise_angles = {}
        for i, ct1 in enumerate(ctypes):
            for j, ct2 in enumerate(ctypes):
                if i >= j:
                    continue
                cos_angle = np.dot(directions[ct1], directions[ct2])
                angle_deg = float(np.degrees(np.arccos(np.clip(cos_angle, -1, 1))))
                pairwise_angles[f"{ct1}_vs_{ct2}"] = angle_deg

        direction_results[str(seed)] = pairwise_angles

    # Average angles across scenes
    avg_angles = {}
    for pair in direction_results[str(seeds[0])]:
        angles = [direction_results[str(s)][pair] for s in seeds[:10]]
        avg_angles[pair] = {
            'mean_angle': float(np.mean(angles)),
            'std_angle': float(np.std(angles)),
        }
        print(f"  {pair}: mean={np.mean(angles):.1f}° ± {np.std(angles):.1f}°")

    results['corruption_directions'] = {
        'per_scene': direction_results,
        'average': avg_angles,
    }

    # ========== 4. Euclidean vs Cosine comparison ==========
    print("\n=== Euclidean vs Cosine ===")

    metric_comparison = {}
    for ct in ctypes:
        cos_dists = []
        euc_dists = []
        for seed in seeds:
            clean = clean_embs[seed]
            corrupt = corrupt_embs[ct][seed]
            cos_dists.append(float(cosine_dist(clean, corrupt)))
            euc_dists.append(float(np.linalg.norm(corrupt - clean)))

        # Rank correlation between metrics
        cos_ranks = np.argsort(np.argsort(cos_dists)).astype(float)
        euc_ranks = np.argsort(np.argsort(euc_dists)).astype(float)
        n = len(cos_ranks)
        spearman = 1 - 6 * np.sum((cos_ranks - euc_ranks)**2) / (n * (n**2 - 1))

        metric_comparison[ct] = {
            'cosine_mean': float(np.mean(cos_dists)),
            'euclidean_mean': float(np.mean(euc_dists)),
            'spearman_correlation': float(spearman),
            'cosine_cv': float(np.std(cos_dists) / np.mean(cos_dists)) if np.mean(cos_dists) > 0 else 0,
            'euclidean_cv': float(np.std(euc_dists) / np.mean(euc_dists)) if np.mean(euc_dists) > 0 else 0,
        }
        print(f"  {ct}: cos_mean={np.mean(cos_dists):.6f}, euc_mean={np.mean(euc_dists):.4f}, "
              f"spearman={spearman:.4f}")

    results['metric_comparison'] = metric_comparison

    # ========== 5. Clean region geometry ==========
    print("\n=== Clean Region Geometry ===")

    # How do clean embeddings from different scenes relate?
    clean_arr = np.array([clean_embs[s] for s in seeds])

    # Pairwise distances between clean embeddings
    n = len(seeds)
    pairwise_clean = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = cosine_dist(clean_arr[i], clean_arr[j])
            pairwise_clean[i, j] = d
            pairwise_clean[j, i] = d

    # Diameter of clean set
    clean_diameter = float(np.max(pairwise_clean))
    clean_mean_dist = float(np.mean(pairwise_clean[np.triu_indices(n, k=1)]))

    # Centroid
    centroid = clean_arr.mean(axis=0)
    centroid_dists = [float(cosine_dist(centroid, clean_arr[i])) for i in range(n)]

    # Compare to corruption distances
    min_corruption = {}
    for ct in ctypes:
        dists = [float(cosine_dist(clean_embs[s], corrupt_embs[ct][s])) for s in seeds]
        min_corruption[ct] = float(min(dists))

    clean_geometry = {
        'diameter': clean_diameter,
        'mean_pairwise_dist': clean_mean_dist,
        'centroid_max_dist': float(max(centroid_dists)),
        'centroid_mean_dist': float(np.mean(centroid_dists)),
        'min_corruption_dist': min_corruption,
        'diameter_to_corruption_ratio': {
            ct: clean_diameter / min_corruption[ct] if min_corruption[ct] > 0 else 'infinite'
            for ct in ctypes
        },
    }
    print(f"  Clean diameter: {clean_diameter:.6f}")
    print(f"  Clean mean pairwise: {clean_mean_dist:.6f}")
    for ct in ctypes:
        ratio = clean_diameter / min_corruption[ct] if min_corruption[ct] > 0 else float('inf')
        print(f"  diameter/min_{ct}: {ratio:.2f}")

    results['clean_geometry'] = clean_geometry

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/metric_geometry_{ts}.json"
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
