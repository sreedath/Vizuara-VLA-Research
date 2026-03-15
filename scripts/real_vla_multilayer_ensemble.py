#!/usr/bin/env python3
"""Experiment 371: Multi-Layer Ensemble Detection

Can combining detectors from multiple hidden layers improve detection?
1. Individual layer AUROC comparison (layers 1-32)
2. Ensemble methods: max, mean, weighted combination
3. Layer voting: majority vote across layers
4. Worst-case layer vs ensemble comparison
5. Optimal layer selection for each corruption
"""

import json, time, os, sys
import numpy as np
import torch
from PIL import Image, ImageFilter
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

def extract_all_layers(model, processor, image, prompt):
    """Extract hidden states from all layers."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return [h[0, -1, :].float().cpu().numpy() for h in fwd.hidden_states]

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

    # Generate images
    print("Generating images...")
    seeds = list(range(0, 1500, 100))[:15]
    images = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        px = rng.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        images[seed] = Image.fromarray(px)

    # Extract all-layer embeddings
    print("Extracting all-layer embeddings...")
    clean_all = {}  # {seed: [layer0_emb, layer1_emb, ...]}
    corrupt_all = {ct: {} for ct in ctypes}

    for seed in seeds:
        clean_all[seed] = extract_all_layers(model, processor, images[seed], prompt)
        for ct in ctypes:
            corrupt_img = apply_corruption(images[seed], ct, 0.5)
            corrupt_all[ct][seed] = extract_all_layers(model, processor, corrupt_img, prompt)

    n_layers = len(clean_all[seeds[0]])
    print(f"  {n_layers} layers, {len(seeds)} scenes")

    # Compute per-layer centroids
    centroids = []
    for l in range(n_layers):
        c = np.mean([clean_all[s][l] for s in seeds], axis=0)
        centroids.append(c)

    # ========== 1. Per-Layer AUROC ==========
    print("\n=== Per-Layer AUROC ===")

    per_layer_auroc = {}
    for l in range(n_layers):
        id_dists = [cosine_dist(centroids[l], clean_all[s][l]) for s in seeds]
        per_ct = {}
        for ct in ctypes:
            ood_dists = [cosine_dist(centroids[l], corrupt_all[ct][s][l]) for s in seeds]
            per_ct[ct] = float(compute_auroc(id_dists, ood_dists))
        per_layer_auroc[str(l)] = per_ct

    # Find best/worst layer per corruption
    for ct in ctypes:
        aurocs = [(l, per_layer_auroc[str(l)][ct]) for l in range(n_layers)]
        best = max(aurocs, key=lambda x: x[1])
        worst = min(aurocs, key=lambda x: x[1])
        print(f"  {ct}: best=L{best[0]} ({best[1]:.4f}), worst=L{worst[0]} ({worst[1]:.4f})")

    results['per_layer_auroc'] = per_layer_auroc

    # ========== 2. Ensemble Methods ==========
    print("\n=== Ensemble Methods ===")

    # Sampled layers for ensemble
    ensemble_layers = [1, 3, 7, 15, 23, 31]  # early, mid, late

    ensemble_results = {}
    for ct in ctypes:
        # Per-layer distances
        per_layer_id = {}
        per_layer_ood = {}
        for l in ensemble_layers:
            per_layer_id[l] = [cosine_dist(centroids[l], clean_all[s][l]) for s in seeds]
            per_layer_ood[l] = [cosine_dist(centroids[l], corrupt_all[ct][s][l]) for s in seeds]

        # Method 1: Max distance across layers
        max_id = [max(per_layer_id[l][i] for l in ensemble_layers) for i in range(len(seeds))]
        max_ood = [max(per_layer_ood[l][i] for l in ensemble_layers) for i in range(len(seeds))]
        auroc_max = compute_auroc(max_id, max_ood)

        # Method 2: Mean distance
        mean_id = [np.mean([per_layer_id[l][i] for l in ensemble_layers]) for i in range(len(seeds))]
        mean_ood = [np.mean([per_layer_ood[l][i] for l in ensemble_layers]) for i in range(len(seeds))]
        auroc_mean = compute_auroc(mean_id, mean_ood)

        # Method 3: Sum of distances
        sum_id = [sum(per_layer_id[l][i] for l in ensemble_layers) for i in range(len(seeds))]
        sum_ood = [sum(per_layer_ood[l][i] for l in ensemble_layers) for i in range(len(seeds))]
        auroc_sum = compute_auroc(sum_id, sum_ood)

        # Single best layer
        single_best = max(ensemble_layers,
                         key=lambda l: compute_auroc(per_layer_id[l], per_layer_ood[l]))
        auroc_single = compute_auroc(per_layer_id[single_best], per_layer_ood[single_best])

        ensemble_results[ct] = {
            'max_auroc': float(auroc_max),
            'mean_auroc': float(auroc_mean),
            'sum_auroc': float(auroc_sum),
            'single_best_layer': single_best,
            'single_best_auroc': float(auroc_single),
        }
        print(f"  {ct}: max={auroc_max:.4f}, mean={auroc_mean:.4f}, "
              f"single_best=L{single_best}({auroc_single:.4f})")

    results['ensemble'] = ensemble_results

    # ========== 3. Layer Voting ==========
    print("\n=== Layer Voting ===")

    voting = {}
    for ct in ctypes:
        # For each scene, how many layers detect it as OOD?
        clean_votes = []
        corrupt_votes = []

        for seed in seeds:
            clean_count = 0
            corrupt_count = 0
            for l in range(n_layers):
                id_dists = [cosine_dist(centroids[l], clean_all[s][l]) for s in seeds]
                thresh = max(id_dists)

                clean_dist = cosine_dist(centroids[l], clean_all[seed][l])
                corrupt_dist = cosine_dist(centroids[l], corrupt_all[ct][seed][l])

                if clean_dist > thresh:
                    clean_count += 1
                if corrupt_dist > thresh:
                    corrupt_count += 1

            clean_votes.append(clean_count)
            corrupt_votes.append(corrupt_count)

        voting[ct] = {
            'clean_mean_votes': float(np.mean(clean_votes)),
            'corrupt_mean_votes': float(np.mean(corrupt_votes)),
            'clean_max_votes': int(max(clean_votes)),
            'corrupt_min_votes': int(min(corrupt_votes)),
            'unanimous_detection': int(min(corrupt_votes)) == n_layers,
        }
        print(f"  {ct}: clean_votes={np.mean(clean_votes):.1f}, "
              f"corrupt_votes={np.mean(corrupt_votes):.1f}/{n_layers}")

    results['voting'] = voting

    # ========== 4. Layer Correlation ==========
    print("\n=== Layer Distance Correlation ===")

    # How correlated are distances across layers?
    layer_corr = {}
    sample_layers = [1, 3, 7, 15, 23, 31]
    for ct in ctypes:
        all_dists = {}
        for l in sample_layers:
            dists = [cosine_dist(centroids[l], corrupt_all[ct][s][l]) for s in seeds]
            all_dists[l] = dists

        corr_pairs = {}
        for i, l1 in enumerate(sample_layers):
            for l2 in sample_layers[i+1:]:
                r = float(np.corrcoef(all_dists[l1], all_dists[l2])[0, 1])
                corr_pairs[f"L{l1}_L{l2}"] = r

        layer_corr[ct] = {
            'mean_correlation': float(np.mean(list(corr_pairs.values()))),
            'min_correlation': float(min(corr_pairs.values())),
            'pairs': corr_pairs,
        }
        print(f"  {ct}: mean_corr={np.mean(list(corr_pairs.values())):.4f}")

    results['layer_correlation'] = layer_corr

    # ========== 5. Noise-Specific Multi-Layer Analysis ==========
    print("\n=== Noise Detection: Multi-Layer Advantage ===")

    noise_analysis = {}
    for l in range(n_layers):
        id_dists = [cosine_dist(centroids[l], clean_all[s][l]) for s in seeds]
        ood_dists = [cosine_dist(centroids[l], corrupt_all['noise'][s][l]) for s in seeds]
        auroc = compute_auroc(id_dists, ood_dists)
        gap = min(ood_dists) - max(id_dists)
        noise_analysis[str(l)] = {
            'auroc': float(auroc),
            'gap': float(gap),
        }

    # Best layer for noise
    best_noise_layer = max(range(n_layers), key=lambda l: noise_analysis[str(l)]['auroc'])
    results['noise_best_layer'] = {
        'layer': best_noise_layer,
        'auroc': noise_analysis[str(best_noise_layer)]['auroc'],
        'gap': noise_analysis[str(best_noise_layer)]['gap'],
    }
    print(f"  Best noise layer: L{best_noise_layer} "
          f"(AUROC={noise_analysis[str(best_noise_layer)]['auroc']:.4f}, "
          f"gap={noise_analysis[str(best_noise_layer)]['gap']:.6f})")

    results['noise_per_layer'] = noise_analysis

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/multilayer_ensemble_{ts}.json"
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
