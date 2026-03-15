#!/usr/bin/env python3
"""Experiment 374: Embedding Space Topology and Cluster Geometry

Detailed analysis of the embedding space structure:
1. Clean embedding cluster shape: eigenspectrum, effective dimension
2. Corruption cluster separation: inter-cluster vs intra-cluster distances
3. Decision boundary geometry: margin, confidence intervals
4. Nearest-neighbor structure: k-NN consistency
5. Mahalanobis distance comparison: covariance-aware detection
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

def compute_auroc(id_scores, ood_scores):
    id_s = np.asarray(id_scores)
    ood_s = np.asarray(ood_scores)
    n_id, n_ood = len(id_s), len(ood_s)
    if n_id == 0 or n_ood == 0:
        return 0.5
    count = sum(float(np.sum(o > id_s) + 0.5 * np.sum(o == id_s)) for o in ood_s)
    return count / (n_id * n_ood)

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

    # Generate images with more seeds for better statistics
    print("Generating images...")
    seeds = list(range(0, 2000, 100))[:20]
    images = {}
    clean_embs = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        px = rng.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        images[seed] = Image.fromarray(px)
        clean_embs[seed] = extract_hidden(model, processor, images[seed], prompt)

    # Corrupt embeddings at multiple severities
    corrupt_embs = {ct: {} for ct in ctypes}
    for ct in ctypes:
        for seed in seeds:
            corrupt_img = apply_corruption(images[seed], ct, 0.5)
            corrupt_embs[ct][seed] = extract_hidden(model, processor, corrupt_img, prompt)

    centroid = np.mean(list(clean_embs.values()), axis=0)
    print(f"  {len(seeds)} scenes, centroid computed")

    # ========== 1. Clean Cluster Shape ==========
    print("\n=== Clean Cluster Shape ===")

    clean_arr = np.array(list(clean_embs.values()))
    clean_centered = clean_arr - centroid

    # Check if all clean embeddings are identical
    pairwise_dists = []
    for i in range(len(seeds)):
        for j in range(i+1, len(seeds)):
            pairwise_dists.append(cosine_dist(clean_arr[i], clean_arr[j]))

    clean_std = np.std(clean_arr, axis=0)
    nonzero_dims = np.sum(clean_std > 1e-10)

    # Eigenspectrum (if variance exists)
    if nonzero_dims > 0:
        cov = np.cov(clean_centered.T)
        eigenvalues = np.linalg.eigvalsh(cov)[::-1]
        total_var = np.sum(eigenvalues[eigenvalues > 0])
        cumvar = np.cumsum(eigenvalues) / max(total_var, 1e-30)
        effective_dim = np.searchsorted(cumvar, 0.95) + 1
    else:
        eigenvalues = np.zeros(10)
        effective_dim = 0
        cumvar = np.zeros(10)

    results['clean_cluster'] = {
        'n_scenes': len(seeds),
        'embedding_dim': int(clean_arr.shape[1]),
        'mean_pairwise_cosine_dist': float(np.mean(pairwise_dists)),
        'max_pairwise_cosine_dist': float(np.max(pairwise_dists)) if pairwise_dists else 0,
        'std_pairwise_cosine_dist': float(np.std(pairwise_dists)) if pairwise_dists else 0,
        'nonzero_variance_dims': int(nonzero_dims),
        'effective_dim_95': int(effective_dim),
        'all_identical': float(np.max(pairwise_dists)) < 1e-10 if pairwise_dists else True,
    }
    print(f"  Pairwise cosine dist: mean={np.mean(pairwise_dists):.8f}, "
          f"max={np.max(pairwise_dists):.8f}")
    print(f"  Nonzero dims: {nonzero_dims}, effective_dim: {effective_dim}")

    # ========== 2. Inter-Cluster Distances ==========
    print("\n=== Inter-Cluster Distances ===")

    cluster_dists = {}
    for ct in ctypes:
        corrupt_arr = np.array([corrupt_embs[ct][s] for s in seeds])
        corrupt_centroid = np.mean(corrupt_arr, axis=0)

        # Inter-cluster: clean centroid to corrupt centroid
        inter_dist = cosine_dist(centroid, corrupt_centroid)

        # Intra-cluster corrupt: pairwise within corrupt
        intra_dists = []
        for i in range(len(seeds)):
            for j in range(i+1, len(seeds)):
                intra_dists.append(cosine_dist(corrupt_arr[i], corrupt_arr[j]))

        # Clean-to-corrupt distances per scene
        scene_dists = [cosine_dist(clean_embs[s], corrupt_embs[ct][s]) for s in seeds]

        # Separation ratio
        intra_mean = np.mean(intra_dists) if intra_dists else 0
        sep_ratio = inter_dist / max(intra_mean, 1e-10)

        cluster_dists[ct] = {
            'inter_cluster_dist': float(inter_dist),
            'intra_corrupt_mean': float(intra_mean),
            'intra_corrupt_max': float(np.max(intra_dists)) if intra_dists else 0,
            'intra_corrupt_std': float(np.std(intra_dists)) if intra_dists else 0,
            'scene_dist_mean': float(np.mean(scene_dists)),
            'scene_dist_std': float(np.std(scene_dists)),
            'separation_ratio': float(sep_ratio),
        }
        print(f"  {ct}: inter={inter_dist:.6f}, intra={intra_mean:.6f}, "
              f"sep_ratio={sep_ratio:.2f}x")

    results['cluster_distances'] = cluster_dists

    # ========== 3. Decision Boundary Margin ==========
    print("\n=== Decision Boundary Margin ===")

    boundary = {}
    clean_to_centroid = [cosine_dist(centroid, clean_embs[s]) for s in seeds]
    threshold = max(clean_to_centroid)

    for ct in ctypes:
        corrupt_to_centroid = [cosine_dist(centroid, corrupt_embs[ct][s]) for s in seeds]

        gap = min(corrupt_to_centroid) - threshold
        margin = gap / threshold if threshold > 0 else float('inf')

        # Confidence interval: how many std devs is the gap?
        if np.std(corrupt_to_centroid) > 0:
            z_score = gap / np.std(corrupt_to_centroid)
        else:
            z_score = float('inf')

        boundary[ct] = {
            'threshold': float(threshold),
            'min_corrupt_dist': float(min(corrupt_to_centroid)),
            'max_clean_dist': float(threshold),
            'gap': float(gap),
            'relative_margin': float(margin),
            'z_score': float(z_score),
            'auroc': float(compute_auroc(clean_to_centroid, corrupt_to_centroid)),
        }
        print(f"  {ct}: gap={gap:.6f}, margin={margin:.2f}x, z={z_score:.2f}")

    results['decision_boundary'] = boundary

    # ========== 4. k-NN Analysis ==========
    print("\n=== k-NN Consistency ===")

    all_embs = []
    all_labels = []
    for s in seeds:
        all_embs.append(clean_embs[s])
        all_labels.append('clean')
    for ct in ctypes:
        for s in seeds:
            all_embs.append(corrupt_embs[ct][s])
            all_labels.append(ct)
    all_embs = np.array(all_embs)

    knn_results = {}
    for k in [1, 3, 5, 10]:
        correct = 0
        total = len(all_embs)
        per_class_acc = {c: {'correct': 0, 'total': 0} for c in ['clean'] + ctypes}

        for i in range(total):
            dists = [cosine_dist(all_embs[i], all_embs[j]) for j in range(total) if j != i]
            labels_other = [all_labels[j] for j in range(total) if j != i]
            sorted_idx = np.argsort(dists)[:k]
            neighbor_labels = [labels_other[idx] for idx in sorted_idx]

            # Majority vote
            from collections import Counter
            vote = Counter(neighbor_labels).most_common(1)[0][0]
            true_label = all_labels[i]

            per_class_acc[true_label]['total'] += 1
            if vote == true_label:
                correct += 1
                per_class_acc[true_label]['correct'] += 1

        acc = correct / total
        knn_results[str(k)] = {
            'accuracy': float(acc),
            'per_class': {c: float(v['correct'] / max(v['total'], 1))
                         for c, v in per_class_acc.items()},
        }
        print(f"  k={k}: accuracy={acc:.4f}")

    results['knn'] = knn_results

    # ========== 5. Distance Metric Comparison ==========
    print("\n=== Distance Metric Comparison ===")

    metrics = {}
    for ct in ctypes:
        clean_cos = [cosine_dist(centroid, clean_embs[s]) for s in seeds]
        corrupt_cos = [cosine_dist(centroid, corrupt_embs[ct][s]) for s in seeds]

        # Euclidean
        clean_euc = [float(np.linalg.norm(clean_embs[s] - centroid)) for s in seeds]
        corrupt_euc = [float(np.linalg.norm(corrupt_embs[ct][s] - centroid)) for s in seeds]

        # L1 (Manhattan)
        clean_l1 = [float(np.sum(np.abs(clean_embs[s] - centroid))) for s in seeds]
        corrupt_l1 = [float(np.sum(np.abs(corrupt_embs[ct][s] - centroid))) for s in seeds]

        # Chebyshev (L-inf)
        clean_linf = [float(np.max(np.abs(clean_embs[s] - centroid))) for s in seeds]
        corrupt_linf = [float(np.max(np.abs(corrupt_embs[ct][s] - centroid))) for s in seeds]

        metrics[ct] = {
            'cosine_auroc': float(compute_auroc(clean_cos, corrupt_cos)),
            'euclidean_auroc': float(compute_auroc(clean_euc, corrupt_euc)),
            'manhattan_auroc': float(compute_auroc(clean_l1, corrupt_l1)),
            'chebyshev_auroc': float(compute_auroc(clean_linf, corrupt_linf)),
            'cosine_gap': float(min(corrupt_cos) - max(clean_cos)),
            'euclidean_gap': float(min(corrupt_euc) - max(clean_euc)),
        }
        print(f"  {ct}: cos={metrics[ct]['cosine_auroc']:.4f}, "
              f"euc={metrics[ct]['euclidean_auroc']:.4f}, "
              f"man={metrics[ct]['manhattan_auroc']:.4f}, "
              f"cheb={metrics[ct]['chebyshev_auroc']:.4f}")

    results['metric_comparison'] = metrics

    # ========== 6. Corruption Severity Sensitivity Curve ==========
    print("\n=== Severity Sensitivity Curve ===")

    severity_curve = {}
    for ct in ctypes:
        sevs = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
        sev_data = {}
        for sev in sevs:
            sev_dists = []
            for seed in seeds[:10]:
                corrupt_img = apply_corruption(images[seed], ct, sev)
                emb = extract_hidden(model, processor, corrupt_img, prompt)
                d = cosine_dist(emb, centroid)
                sev_dists.append(d)

            # Detection rate at this severity
            det_rate = sum(1 for d in sev_dists if d > threshold) / len(sev_dists)
            sev_data[str(sev)] = {
                'mean_dist': float(np.mean(sev_dists)),
                'min_dist': float(min(sev_dists)),
                'detection_rate': float(det_rate),
            }

        severity_curve[ct] = sev_data
        # Find minimum detection severity
        min_det = None
        for sev in sevs:
            if sev_data[str(sev)]['detection_rate'] == 1.0:
                min_det = sev
                break
        print(f"  {ct}: 100% detection at severity={min_det}")

    results['severity_curve'] = severity_curve

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/embedding_topology_{ts}.json"
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
