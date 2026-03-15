#!/usr/bin/env python3
"""Experiment 331: Embedding Geometry Deep Dive (Real OpenVLA-7B)

Deep analysis of embedding space geometry:
1. PCA projection of all corruption embeddings
2. Distance distribution analysis
3. Convex hull properties
4. Nearest-neighbor analysis
5. Manifold curvature
6. Cluster separability metrics
7. Decision boundary characterization
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

    # ========== Collect comprehensive embeddings ==========
    print("\n=== Collecting Embeddings ===")

    # 5 scenes
    scene_imgs = {}
    scene_embs = {}
    for seed in [42, 99, 123, 777, 2000]:
        rng = np.random.RandomState(seed)
        px = rng.randint(50, 200, (224, 224, 3)).astype(np.uint8)
        scene_imgs[seed] = Image.fromarray(px)
        scene_embs[seed] = extract_hidden(model, processor, scene_imgs[seed], prompt)

    # 4 corruptions × 5 severities × 5 scenes = 100 embeddings
    ctypes = ['fog', 'night', 'noise', 'blur']
    sevs = [0.1, 0.25, 0.5, 0.75, 1.0]

    all_embs = []
    all_labels = []
    all_meta = []

    for seed in scene_embs:
        # Clean
        all_embs.append(scene_embs[seed])
        all_labels.append('clean')
        all_meta.append({'scene': seed, 'corruption': 'clean', 'severity': 0})

        # Corrupted
        for ct in ctypes:
            for sev in sevs:
                img = apply_corruption(scene_imgs[seed], ct, sev)
                emb = extract_hidden(model, processor, img, prompt)
                all_embs.append(emb)
                all_labels.append(ct)
                all_meta.append({'scene': seed, 'corruption': ct, 'severity': sev})

    all_embs = np.array(all_embs)  # 105 × 4096
    print(f"  Total embeddings: {len(all_embs)}")

    # ========== 1. PCA Analysis ==========
    print("\n=== PCA Analysis ===")
    mean_emb = all_embs.mean(axis=0)
    centered = all_embs - mean_emb
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    explained_var = S**2 / np.sum(S**2)
    cum_var = np.cumsum(explained_var)

    # Project to 3D
    pca_3d = centered @ Vt[:3].T

    pca_results = {
        'explained_variance_top10': [float(v) for v in explained_var[:10]],
        'cumulative_variance_top10': [float(v) for v in cum_var[:10]],
        'dims_for_80pct': int(np.searchsorted(cum_var, 0.8) + 1),
        'dims_for_90pct': int(np.searchsorted(cum_var, 0.9) + 1),
        'dims_for_95pct': int(np.searchsorted(cum_var, 0.95) + 1),
    }
    print(f"  80%: {pca_results['dims_for_80pct']}D, 90%: {pca_results['dims_for_90pct']}D, 95%: {pca_results['dims_for_95pct']}D")
    print(f"  Top-3 variance: {explained_var[:3]}")

    results['pca'] = pca_results

    # ========== 2. Cluster Analysis ==========
    print("\n=== Cluster Analysis ===")

    # Separate clean and corrupt embeddings
    clean_idx = [i for i, l in enumerate(all_labels) if l == 'clean']
    corrupt_idx = [i for i, l in enumerate(all_labels) if l != 'clean']

    clean_embs = all_embs[clean_idx]
    corrupt_embs_all = all_embs[corrupt_idx]

    # Intra-class distances
    clean_dists = []
    for i in range(len(clean_embs)):
        for j in range(i+1, len(clean_embs)):
            clean_dists.append(cosine_dist(clean_embs[i], clean_embs[j]))

    # Inter-class distances (clean vs corrupt)
    inter_dists = []
    for c in clean_embs:
        for o in corrupt_embs_all[:20]:  # sample
            inter_dists.append(cosine_dist(c, o))

    # Silhouette-like score
    avg_intra = np.mean(clean_dists) if clean_dists else 0
    avg_inter = np.mean(inter_dists) if inter_dists else 0
    silhouette = (avg_inter - avg_intra) / max(avg_inter, avg_intra, 1e-10)

    cluster_results = {
        'n_clean': len(clean_idx),
        'n_corrupt': len(corrupt_idx),
        'avg_clean_intra_dist': float(avg_intra),
        'max_clean_intra_dist': float(max(clean_dists)) if clean_dists else 0,
        'avg_inter_dist': float(avg_inter),
        'min_inter_dist': float(min(inter_dists)) if inter_dists else 0,
        'silhouette_score': float(silhouette),
    }
    print(f"  Clean intra: mean={avg_intra:.6f}, max={max(clean_dists):.6f}")
    print(f"  Inter: mean={avg_inter:.6f}, min={min(inter_dists):.6f}")
    print(f"  Silhouette: {silhouette:.4f}")

    results['clusters'] = cluster_results

    # ========== 3. Per-corruption geometry ==========
    print("\n=== Per-Corruption Geometry ===")
    corruption_geometry = {}

    for ct in ctypes:
        ct_idx = [i for i, l in enumerate(all_labels) if l == ct]
        ct_embs = all_embs[ct_idx]

        # Mean direction from clean centroid
        clean_mean = np.mean(clean_embs, axis=0)
        ct_mean = np.mean(ct_embs, axis=0)
        direction = ct_mean - clean_mean
        dir_norm = np.linalg.norm(direction)

        # Spread (std of distances from mean)
        ct_dists_from_mean = [cosine_dist(ct_mean, e) for e in ct_embs]

        # Distance to clean
        ct_to_clean = [cosine_dist(clean_mean, e) for e in ct_embs]

        corruption_geometry[ct] = {
            'n_samples': len(ct_idx),
            'mean_dist_to_clean': float(np.mean(ct_to_clean)),
            'std_dist_to_clean': float(np.std(ct_to_clean)),
            'spread': float(np.mean(ct_dists_from_mean)),
            'direction_norm': float(dir_norm),
            'min_dist_to_clean': float(min(ct_to_clean)),
            'max_dist_to_clean': float(max(ct_to_clean)),
        }
        print(f"  {ct}: mean_d={np.mean(ct_to_clean):.6f}±{np.std(ct_to_clean):.6f}, spread={np.mean(ct_dists_from_mean):.6f}")

    results['corruption_geometry'] = corruption_geometry

    # ========== 4. Nearest-neighbor analysis ==========
    print("\n=== Nearest Neighbor ===")
    # For each corrupt embedding, find nearest clean embedding
    nn_results = {}
    for ct in ctypes:
        ct_idx = [i for i, l in enumerate(all_labels) if l == ct]
        correct_nn = 0
        total = 0
        for idx in ct_idx:
            # Find nearest among all embeddings
            dists = [cosine_dist(all_embs[idx], all_embs[j]) for j in range(len(all_embs)) if j != idx]
            labels_other = [all_labels[j] for j in range(len(all_embs)) if j != idx]
            nn_label = labels_other[np.argmin(dists)]
            if nn_label == ct:
                correct_nn += 1
            total += 1

        nn_results[ct] = {
            'nn_accuracy': float(correct_nn / total) if total > 0 else 0,
            'n_samples': total,
        }
        print(f"  {ct}: 1-NN accuracy = {correct_nn}/{total} = {correct_nn/total:.3f}")

    results['nearest_neighbor'] = nn_results

    # ========== 5. Distance matrix summary ==========
    print("\n=== Distance Matrix ===")
    # Average distance between corruption types
    type_means = {}
    for ct in ['clean'] + ctypes:
        idx = [i for i, l in enumerate(all_labels) if l == ct]
        if idx:
            type_means[ct] = np.mean(all_embs[idx], axis=0)

    dist_matrix = {}
    for t1 in type_means:
        dist_matrix[t1] = {}
        for t2 in type_means:
            dist_matrix[t1][t2] = float(cosine_dist(type_means[t1], type_means[t2]))

    results['distance_matrix'] = dist_matrix
    print("  Type-type distances (cosine):")
    for t1 in dist_matrix:
        dists = [f"{dist_matrix[t1][t2]:.6f}" for t2 in dist_matrix[t1]]
        print(f"    {t1}: {dists}")

    # ========== 6. Convexity analysis ==========
    print("\n=== Convexity Analysis ===")
    # Test if midpoint between two corrupt embeddings is closer to or farther from clean
    convex_results = {}
    for ct in ctypes:
        ct_idx = [i for i, l in enumerate(all_labels) if l == ct]
        if len(ct_idx) < 2:
            continue
        midpoint_tests = []
        for i in range(min(10, len(ct_idx))):
            for j in range(i+1, min(10, len(ct_idx))):
                midpoint = (all_embs[ct_idx[i]] + all_embs[ct_idx[j]]) / 2
                d_mid = cosine_dist(np.mean(clean_embs, axis=0), midpoint)
                d_i = cosine_dist(np.mean(clean_embs, axis=0), all_embs[ct_idx[i]])
                d_j = cosine_dist(np.mean(clean_embs, axis=0), all_embs[ct_idx[j]])
                avg_endpoint = (d_i + d_j) / 2
                midpoint_tests.append({
                    'd_midpoint': float(d_mid),
                    'd_avg_endpoints': float(avg_endpoint),
                    'convex': bool(d_mid <= avg_endpoint),
                })

        n_convex = sum(1 for t in midpoint_tests if t['convex'])
        convex_results[ct] = {
            'n_tests': len(midpoint_tests),
            'n_convex': n_convex,
            'convexity_rate': float(n_convex / len(midpoint_tests)) if midpoint_tests else 0,
        }
        print(f"  {ct}: {n_convex}/{len(midpoint_tests)} convex ({n_convex/len(midpoint_tests)*100:.1f}%)")

    results['convexity'] = convex_results

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/geometry_deep_{ts}.json"
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
