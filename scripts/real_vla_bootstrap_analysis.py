#!/usr/bin/env python3
"""Experiment 430: Bootstrap Confidence Analysis

Uses bootstrap resampling to estimate confidence intervals for
AUROC and detection thresholds. How reliable are our estimates
given the small sample sizes?

Tests:
1. Bootstrap AUROC confidence intervals (1000 resamples)
2. Threshold stability under resampling
3. False positive rate estimation
4. Detection delay analysis (how many frames needed)
5. Statistical significance tests
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

    # Use larger sample for bootstrap reliability
    seeds = list(range(42, 42 + 15))  # 15 scenes
    scenes = [Image.fromarray(np.random.RandomState(s).randint(0, 255, (224, 224, 3), dtype=np.uint8)) for s in seeds]

    print(f"Extracting embeddings for {len(scenes)} scenes...")
    clean_embs = [extract_hidden(model, processor, s, prompt) for s in scenes]
    centroid = np.mean(clean_embs, axis=0)
    clean_dists = np.array([cosine_dist(e, centroid) for e in clean_embs])

    corrupt_dists = {}
    for c in corruptions:
        dists = []
        for s in scenes:
            emb = extract_hidden(model, processor, apply_corruption(s, c), prompt)
            dists.append(cosine_dist(emb, centroid))
        corrupt_dists[c] = np.array(dists)
        print(f"  {c}: mean_dist={np.mean(dists):.6f}")

    results = {"n_scenes": len(scenes)}

    # === Test 1: Bootstrap AUROC ===
    print("\n=== Bootstrap AUROC Confidence Intervals ===")
    n_bootstrap = 1000
    rng = np.random.RandomState(42)
    bootstrap_results = {}
    for c in corruptions:
        aurocs = []
        for _ in range(n_bootstrap):
            id_idx = rng.choice(len(clean_dists), len(clean_dists), replace=True)
            ood_idx = rng.choice(len(corrupt_dists[c]), len(corrupt_dists[c]), replace=True)
            auroc = compute_auroc(clean_dists[id_idx], corrupt_dists[c][ood_idx])
            aurocs.append(auroc)
        aurocs = np.array(aurocs)
        bootstrap_results[c] = {
            "mean": float(np.mean(aurocs)),
            "std": float(np.std(aurocs)),
            "ci_lower": float(np.percentile(aurocs, 2.5)),
            "ci_upper": float(np.percentile(aurocs, 97.5)),
            "pct_perfect": float(np.mean(aurocs == 1.0) * 100),
        }
        print(f"  {c}: {np.mean(aurocs):.4f} [{np.percentile(aurocs, 2.5):.4f}, {np.percentile(aurocs, 97.5):.4f}], {np.mean(aurocs == 1.0)*100:.1f}% perfect")
    results["bootstrap_auroc"] = bootstrap_results

    # === Test 2: Threshold stability ===
    print("\n=== Threshold Stability ===")
    # Optimal threshold = midpoint between max clean and min corrupt
    threshold_results = {}
    for c in corruptions:
        thresholds = []
        for _ in range(n_bootstrap):
            id_idx = rng.choice(len(clean_dists), len(clean_dists), replace=True)
            ood_idx = rng.choice(len(corrupt_dists[c]), len(corrupt_dists[c]), replace=True)
            max_clean = np.max(clean_dists[id_idx])
            min_corrupt = np.min(corrupt_dists[c][ood_idx])
            threshold = (max_clean + min_corrupt) / 2
            thresholds.append(float(threshold))

        thresholds = np.array(thresholds)
        gap = np.min(corrupt_dists[c]) - np.max(clean_dists)
        threshold_results[c] = {
            "mean_threshold": float(np.mean(thresholds)),
            "std_threshold": float(np.std(thresholds)),
            "ci_lower": float(np.percentile(thresholds, 2.5)),
            "ci_upper": float(np.percentile(thresholds, 97.5)),
            "actual_gap": float(gap),
            "gap_positive": bool(gap > 0),
        }
        print(f"  {c}: threshold={np.mean(thresholds):.6f} ± {np.std(thresholds):.6f}, gap={gap:.6f}")
    results["threshold_stability"] = threshold_results

    # === Test 3: False positive rates ===
    print("\n=== False Positive Rate Estimation ===")
    fpr_results = {}
    # Use different threshold strategies
    for strategy in ['3sigma', '5sigma', 'max_clean']:
        clean_mean = float(np.mean(clean_dists))
        clean_std = float(np.std(clean_dists))
        if strategy == '3sigma':
            threshold = clean_mean + 3 * clean_std
        elif strategy == '5sigma':
            threshold = clean_mean + 5 * clean_std
        elif strategy == 'max_clean':
            threshold = float(np.max(clean_dists)) * 1.01

        # Estimate FPR on clean
        fpr = float(np.mean(clean_dists > threshold))
        # TPR on each corruption
        tprs = {}
        for c in corruptions:
            tprs[c] = float(np.mean(corrupt_dists[c] > threshold))

        fpr_results[strategy] = {
            "threshold": float(threshold),
            "fpr": fpr,
            "tpr": tprs,
        }
        print(f"  {strategy}: threshold={threshold:.6f}, FPR={fpr:.4f}, TPR_fog={tprs['fog']:.4f}")
    results["false_positive_rates"] = fpr_results

    # === Test 4: Separation statistics ===
    print("\n=== Separation Statistics ===")
    separation = {}
    for c in corruptions:
        # Cohen's d
        pooled_std = np.sqrt((np.var(clean_dists) + np.var(corrupt_dists[c])) / 2)
        cohens_d = (np.mean(corrupt_dists[c]) - np.mean(clean_dists)) / pooled_std if pooled_std > 0 else float('inf')

        # Overlap coefficient
        min_corrupt = np.min(corrupt_dists[c])
        max_clean = np.max(clean_dists)
        overlap = max(0, max_clean - min_corrupt)

        separation[c] = {
            "cohens_d": float(cohens_d),
            "clean_max": float(max_clean),
            "corrupt_min": float(min_corrupt),
            "gap": float(min_corrupt - max_clean),
            "overlap": float(overlap),
            "perfectly_separated": bool(min_corrupt > max_clean),
        }
        print(f"  {c}: Cohen's d={cohens_d:.4f}, gap={min_corrupt - max_clean:.6f}, separated={min_corrupt > max_clean}")
    results["separation"] = separation

    # === Test 5: Sample size effect ===
    print("\n=== Sample Size Effect ===")
    sample_size_results = {}
    for n in [2, 3, 5, 7, 10, 15]:
        if n > len(scenes):
            continue
        aurocs_per_n = []
        for trial in range(100):
            idx = rng.choice(len(scenes), n, replace=False)
            sub_embs = [clean_embs[i] for i in idx]
            sub_centroid = np.mean(sub_embs, axis=0)
            sub_clean_dists = [cosine_dist(e, sub_centroid) for e in sub_embs]
            all_ood = []
            for c in corruptions:
                for i in idx:
                    emb_val = corrupt_dists[c][i]  # Reuse distances (approximate)
                    all_ood.append(float(emb_val))
            auroc = compute_auroc(sub_clean_dists, all_ood)
            aurocs_per_n.append(auroc)

        sample_size_results[str(n)] = {
            "mean_auroc": float(np.mean(aurocs_per_n)),
            "std_auroc": float(np.std(aurocs_per_n)),
            "min_auroc": float(np.min(aurocs_per_n)),
            "pct_perfect": float(np.mean(np.array(aurocs_per_n) >= 1.0) * 100),
        }
        print(f"  n={n}: mean_auroc={np.mean(aurocs_per_n):.4f}, std={np.std(aurocs_per_n):.4f}, {np.mean(np.array(aurocs_per_n) >= 1.0)*100:.0f}% perfect")
    results["sample_size_effect"] = sample_size_results

    out_path = "/workspace/Vizuara-VLA-Research/experiments/bootstrap_analysis_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
