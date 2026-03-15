#!/usr/bin/env python3
"""Experiment 428: Hidden State Distribution Analysis

Deep statistical analysis of the 4096-dimensional hidden state vectors.
Investigates sparsity, activation patterns, per-dimension statistics,
and which dimensions carry the most discriminative information.

Tests:
1. Per-dimension mean/std for clean vs corrupted
2. Sparsity analysis (near-zero dimensions)
3. Top discriminative dimensions (by effect size)
4. Activation histogram comparison
5. Few-dimension detection: how few dims suffice?
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

def compute_auroc(id_scores, ood_scores):
    id_s = np.asarray(id_scores, dtype=np.float64)
    ood_s = np.asarray(ood_scores, dtype=np.float64)
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
    corruptions = ['fog', 'night', 'noise', 'blur']

    seeds = [42, 123, 456, 789, 999, 1234, 5678, 9999]
    scenes = [Image.fromarray(np.random.RandomState(s).randint(0, 255, (224, 224, 3), dtype=np.uint8)) for s in seeds]

    print(f"Extracting embeddings for {len(scenes)} scenes...")
    clean_embs = np.array([extract_hidden(model, processor, s, prompt) for s in scenes])
    centroid = np.mean(clean_embs, axis=0)
    hidden_dim = clean_embs.shape[1]
    print(f"  Hidden dim: {hidden_dim}")

    corrupt_embs = {}
    for c in corruptions:
        corrupt_embs[c] = np.array([extract_hidden(model, processor, apply_corruption(s, c), prompt) for s in scenes])
        print(f"  {c} extracted")

    results = {"n_scenes": len(scenes), "hidden_dim": hidden_dim}

    # === Test 1: Per-dimension statistics ===
    print("\n=== Per-Dimension Statistics ===")
    clean_mean = np.mean(clean_embs, axis=0)
    clean_std = np.std(clean_embs, axis=0)

    dim_stats = {
        "clean_mean_of_means": float(np.mean(clean_mean)),
        "clean_std_of_means": float(np.std(clean_mean)),
        "clean_mean_of_stds": float(np.mean(clean_std)),
        "clean_max_std_dim": int(np.argmax(clean_std)),
        "clean_max_std_val": float(np.max(clean_std)),
        "clean_min_std_dim": int(np.argmin(clean_std)),
        "clean_min_std_val": float(np.min(clean_std)),
    }

    for c in corruptions:
        corr_mean = np.mean(corrupt_embs[c], axis=0)
        shift = corr_mean - clean_mean
        dim_stats[f"{c}_mean_shift"] = float(np.mean(np.abs(shift)))
        dim_stats[f"{c}_max_shift_dim"] = int(np.argmax(np.abs(shift)))
        dim_stats[f"{c}_max_shift_val"] = float(np.max(np.abs(shift)))

    print(f"  Clean: mean={dim_stats['clean_mean_of_means']:.4f}, std={dim_stats['clean_mean_of_stds']:.6f}")
    for c in corruptions:
        print(f"  {c}: mean_shift={dim_stats[f'{c}_mean_shift']:.6f}, max at dim {dim_stats[f'{c}_max_shift_dim']}")
    results["dim_stats"] = dim_stats

    # === Test 2: Sparsity analysis ===
    print("\n=== Sparsity Analysis ===")
    thresholds = [0.001, 0.01, 0.1, 1.0]
    sparsity = {}
    for thresh in thresholds:
        n_zero_clean = int(np.sum(np.abs(clean_mean) < thresh))
        sparsity[f"clean_below_{thresh}"] = n_zero_clean
        for c in corruptions:
            corr_mean = np.mean(corrupt_embs[c], axis=0)
            n_zero = int(np.sum(np.abs(corr_mean) < thresh))
            sparsity[f"{c}_below_{thresh}"] = n_zero

    # Activation range
    sparsity["clean_min_activation"] = float(np.min(clean_mean))
    sparsity["clean_max_activation"] = float(np.max(clean_mean))
    sparsity["clean_positive_dims"] = int(np.sum(clean_mean > 0))
    sparsity["clean_negative_dims"] = int(np.sum(clean_mean < 0))
    print(f"  Clean: {sparsity['clean_positive_dims']} positive, {sparsity['clean_negative_dims']} negative dims")
    print(f"  Activation range: [{sparsity['clean_min_activation']:.4f}, {sparsity['clean_max_activation']:.4f}]")
    results["sparsity"] = sparsity

    # === Test 3: Top discriminative dimensions ===
    print("\n=== Top Discriminative Dimensions ===")
    discriminative = {}
    for c in corruptions:
        # Effect size per dimension: |mean_shift| / pooled_std
        corr_mean = np.mean(corrupt_embs[c], axis=0)
        corr_std = np.std(corrupt_embs[c], axis=0)
        pooled_std = np.sqrt((clean_std**2 + corr_std**2) / 2 + 1e-20)
        effect_size = np.abs(corr_mean - clean_mean) / pooled_std

        ranked = np.argsort(effect_size)[::-1]
        top20 = [(int(ranked[i]), float(effect_size[ranked[i]])) for i in range(20)]
        discriminative[c] = {
            "top_20_dims": top20,
            "mean_effect_size": float(np.mean(effect_size)),
            "max_effect_size": float(np.max(effect_size)),
            "n_large_effect": int(np.sum(effect_size > 1.0)),
        }
        print(f"  {c}: max_effect={np.max(effect_size):.4f} at dim {ranked[0]}, {discriminative[c]['n_large_effect']} dims >1.0")
    results["discriminative"] = discriminative

    # === Test 4: Few-dimension detection ===
    print("\n=== Few-Dimension Detection ===")
    few_dim = {}
    for n_dims in [1, 2, 5, 10, 20, 50, 100, 500, 1000]:
        per_corr = {}
        for c in corruptions:
            # Get top-N discriminative dimensions for this corruption
            corr_mean = np.mean(corrupt_embs[c], axis=0)
            corr_std = np.std(corrupt_embs[c], axis=0)
            pooled_std = np.sqrt((clean_std**2 + corr_std**2) / 2 + 1e-20)
            effect_size = np.abs(corr_mean - clean_mean) / pooled_std
            top_dims = np.argsort(effect_size)[::-1][:n_dims]

            # Compute detection using only these dimensions
            clean_proj = clean_embs[:, top_dims]
            corr_proj = corrupt_embs[c][:, top_dims]
            cent_proj = centroid[top_dims]

            id_dists = [float(np.linalg.norm(e - cent_proj)) for e in clean_proj]
            ood_dists = [float(np.linalg.norm(e - cent_proj)) for e in corr_proj]
            auroc = float(compute_auroc(id_dists, ood_dists))
            per_corr[c] = auroc

        # Also compute with random dimensions
        rng = np.random.RandomState(42)
        rand_dims = rng.choice(hidden_dim, min(n_dims, hidden_dim), replace=False)
        rand_per_corr = {}
        for c in corruptions:
            clean_proj = clean_embs[:, rand_dims]
            corr_proj = corrupt_embs[c][:, rand_dims]
            cent_proj = centroid[rand_dims]
            id_dists = [float(np.linalg.norm(e - cent_proj)) for e in clean_proj]
            ood_dists = [float(np.linalg.norm(e - cent_proj)) for e in corr_proj]
            rand_per_corr[c] = float(compute_auroc(id_dists, ood_dists))

        few_dim[str(n_dims)] = {
            "top_dims": per_corr,
            "random_dims": rand_per_corr,
        }
        overall_top = float(np.mean(list(per_corr.values())))
        overall_rand = float(np.mean(list(rand_per_corr.values())))
        print(f"  {n_dims}D: top={overall_top:.4f}, random={overall_rand:.4f}")
    results["few_dim_detection"] = few_dim

    # === Test 5: Activation distribution ===
    print("\n=== Activation Distribution ===")
    activation_dist = {}
    # Histogram bins for clean vs corrupt
    for c in ['clean'] + corruptions:
        if c == 'clean':
            vals = clean_embs.flatten()
        else:
            vals = corrupt_embs[c].flatten()
        activation_dist[c] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "skewness": float(np.mean(((vals - np.mean(vals)) / np.std(vals))**3)),
            "kurtosis": float(np.mean(((vals - np.mean(vals)) / np.std(vals))**4) - 3),
            "pct_positive": float(np.mean(vals > 0) * 100),
            "pct_near_zero": float(np.mean(np.abs(vals) < 0.1) * 100),
        }
        print(f"  {c}: mean={activation_dist[c]['mean']:.4f}, std={activation_dist[c]['std']:.4f}, skew={activation_dist[c]['skewness']:.4f}")
    results["activation_distribution"] = activation_dist

    out_path = "/workspace/Vizuara-VLA-Research/experiments/hidden_state_stats_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
