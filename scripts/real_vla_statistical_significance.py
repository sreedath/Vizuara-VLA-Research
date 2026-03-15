#!/usr/bin/env python3
"""Experiment 396: Statistical Significance and Confidence Intervals

Rigorous statistical analysis of key findings with bootstrap confidence
intervals, permutation tests, and Bonferroni-corrected p-values.

Tests:
1. Bootstrap CI for detection AUROC (per corruption)
2. Permutation test for clean vs corrupt separation
3. Multi-hypothesis testing with Bonferroni correction
4. Effect size confidence intervals
5. Sample size power analysis
6. Cross-validation of detection threshold
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
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    return 1.0 - np.dot(a, b) / (na * nb)

def compute_auroc(id_scores, ood_scores):
    id_s = np.asarray(id_scores)
    ood_s = np.asarray(ood_scores)
    n_id, n_ood = len(id_s), len(ood_s)
    if n_id == 0 or n_ood == 0:
        return 0.5
    count = sum(float(np.sum(o > id_s) + 0.5 * np.sum(o == id_s)) for o in ood_s)
    return count / (n_id * n_ood)

def bootstrap_auroc(id_scores, ood_scores, n_bootstrap=1000, seed=42):
    """Bootstrap confidence interval for AUROC."""
    rng = np.random.RandomState(seed)
    id_arr = np.array(id_scores)
    ood_arr = np.array(ood_scores)
    aurocs = []
    for _ in range(n_bootstrap):
        id_sample = rng.choice(id_arr, size=len(id_arr), replace=True)
        ood_sample = rng.choice(ood_arr, size=len(ood_arr), replace=True)
        aurocs.append(compute_auroc(id_sample, ood_sample))
    return aurocs

def permutation_test(id_scores, ood_scores, n_perm=10000, seed=42):
    """Permutation test for separation significance."""
    rng = np.random.RandomState(seed)
    all_scores = np.concatenate([id_scores, ood_scores])
    observed_diff = np.mean(ood_scores) - np.mean(id_scores)
    n_id = len(id_scores)

    count_greater = 0
    for _ in range(n_perm):
        perm = rng.permutation(all_scores)
        perm_diff = np.mean(perm[n_id:]) - np.mean(perm[:n_id])
        if perm_diff >= observed_diff:
            count_greater += 1

    p_value = (count_greater + 1) / (n_perm + 1)  # +1 for continuity
    return p_value

def main():
    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    img = Image.fromarray(np.random.RandomState(42).randint(0, 255, (224, 224, 3), dtype=np.uint8))

    corruptions = ['fog', 'night', 'noise', 'blur']
    n_samples = 20

    # Collect embeddings
    print("Collecting embeddings...")
    clean_embs = []
    for i in range(n_samples):
        arr = np.array(img).astype(np.float32)
        arr += np.random.RandomState(100 + i).randn(*arr.shape) * 0.5
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        emb = extract_hidden(model, processor, Image.fromarray(arr), prompt)
        clean_embs.append(emb)
        if (i+1) % 5 == 0:
            print(f"  Clean {i+1}/{n_samples}")

    centroid = np.mean(clean_embs, axis=0)
    clean_dists = np.array([cosine_dist(e, centroid) for e in clean_embs])

    corrupt_dists = {}
    for c in corruptions:
        dists = []
        for i in range(n_samples):
            arr = np.array(img).astype(np.float32)
            arr += np.random.RandomState(200 + i).randn(*arr.shape) * 0.5
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            corrupted = apply_corruption(Image.fromarray(arr), c)
            emb = extract_hidden(model, processor, corrupted, prompt)
            dists.append(cosine_dist(emb, centroid))
        corrupt_dists[c] = np.array(dists)
        print(f"  {c}: {n_samples} samples, mean_dist={np.mean(dists):.6f}")

    results = {}

    # 1. Bootstrap CI for AUROC
    print("\n=== Bootstrap AUROC CI ===")
    bootstrap_results = {}
    for c in corruptions:
        aurocs = bootstrap_auroc(clean_dists, corrupt_dists[c], n_bootstrap=2000)
        point_auroc = compute_auroc(clean_dists, corrupt_dists[c])
        ci_low = float(np.percentile(aurocs, 2.5))
        ci_high = float(np.percentile(aurocs, 97.5))
        bootstrap_results[c] = {
            "point_estimate": float(point_auroc),
            "ci_95_low": ci_low,
            "ci_95_high": ci_high,
            "ci_width": ci_high - ci_low,
            "bootstrap_mean": float(np.mean(aurocs)),
            "bootstrap_std": float(np.std(aurocs))
        }
        print(f"  {c}: AUROC={point_auroc:.4f} [{ci_low:.4f}, {ci_high:.4f}]")
    results["bootstrap_auroc"] = bootstrap_results

    # 2. Permutation test
    print("\n=== Permutation Tests ===")
    perm_results = {}
    for c in corruptions:
        p_val = permutation_test(clean_dists, corrupt_dists[c], n_perm=5000)
        perm_results[c] = {
            "p_value": float(p_val),
            "significant_bonferroni": float(p_val) < (0.05 / len(corruptions)),
            "mean_diff": float(np.mean(corrupt_dists[c]) - np.mean(clean_dists)),
            "observed_diff": float(np.mean(corrupt_dists[c]) - np.mean(clean_dists))
        }
        print(f"  {c}: p={p_val:.6f}, significant (Bonferroni)={p_val < 0.05/4}")
    results["permutation_tests"] = perm_results

    # 3. Effect size CI (bootstrap Cohen's d)
    print("\n=== Effect Size CI ===")
    effect_results = {}
    rng = np.random.RandomState(42)
    for c in corruptions:
        cohens_ds = []
        for _ in range(2000):
            id_s = rng.choice(clean_dists, size=len(clean_dists), replace=True)
            ood_s = rng.choice(corrupt_dists[c], size=len(corrupt_dists[c]), replace=True)
            pooled = np.sqrt((np.var(id_s) + np.var(ood_s)) / 2)
            if pooled > 0:
                d = abs(np.mean(ood_s) - np.mean(id_s)) / pooled
            else:
                d = float('inf')
            cohens_ds.append(d)

        effect_results[c] = {
            "point_d": float(abs(np.mean(corrupt_dists[c]) - np.mean(clean_dists)) /
                            np.sqrt((np.var(clean_dists) + np.var(corrupt_dists[c])) / 2 + 1e-20)),
            "ci_95_low": float(np.percentile(cohens_ds, 2.5)),
            "ci_95_high": float(np.percentile(cohens_ds, 97.5)),
            "bootstrap_median": float(np.median(cohens_ds))
        }
        print(f"  {c}: d={effect_results[c]['point_d']:.1f} "
              f"[{effect_results[c]['ci_95_low']:.1f}, {effect_results[c]['ci_95_high']:.1f}]")
    results["effect_size_ci"] = effect_results

    # 4. Leave-one-out cross-validation of threshold
    print("\n=== LOO Cross-Validation ===")
    loo_results = {}
    for c in corruptions:
        fpr_list = []
        tpr_list = []
        for leave_out in range(len(clean_dists)):
            # Leave out one clean sample
            train_clean = np.delete(clean_dists, leave_out)
            threshold = max(train_clean) * 1.1

            # Test: the left-out clean sample
            is_fp = clean_dists[leave_out] > threshold
            fpr_list.append(float(is_fp))

            # Test: all corrupt samples
            n_detected = np.sum(corrupt_dists[c] > threshold)
            tpr_list.append(float(n_detected / len(corrupt_dists[c])))

        loo_results[c] = {
            "mean_fpr": float(np.mean(fpr_list)),
            "mean_tpr": float(np.mean(tpr_list)),
            "all_fpr": fpr_list,
            "all_tpr": tpr_list,
            "perfect_detection": float(np.mean(tpr_list)) == 1.0 and float(np.mean(fpr_list)) == 0.0
        }
        print(f"  {c}: FPR={np.mean(fpr_list):.4f}, TPR={np.mean(tpr_list):.4f}")
    results["loo_cv"] = loo_results

    # 5. Power analysis: minimum n for significant detection
    print("\n=== Power Analysis ===")
    power_results = {}
    for c in corruptions:
        for n in [2, 3, 5, 7, 10, 15, 20]:
            n_sig = 0
            n_trials = 100
            for trial in range(n_trials):
                rng_trial = np.random.RandomState(1000 + trial)
                id_s = rng_trial.choice(clean_dists, size=n, replace=True)
                ood_s = rng_trial.choice(corrupt_dists[c], size=n, replace=True)
                # Mann-Whitney U test approximation
                auroc = compute_auroc(id_s, ood_s)
                if auroc >= 0.99:
                    n_sig += 1
            power = n_sig / n_trials
            power_results.setdefault(c, {})[n] = float(power)
        print(f"  {c}: power at n=2: {power_results[c][2]:.2f}, n=5: {power_results[c][5]:.2f}, n=10: {power_results[c][10]:.2f}")
    results["power_analysis"] = power_results

    # 6. Multiple testing summary
    print("\n=== Multiple Testing Summary ===")
    n_tests = len(corruptions)
    alpha = 0.05
    bonferroni_alpha = alpha / n_tests
    all_significant = all(perm_results[c]["p_value"] < bonferroni_alpha for c in corruptions)
    results["multiple_testing"] = {
        "n_tests": n_tests,
        "nominal_alpha": alpha,
        "bonferroni_alpha": bonferroni_alpha,
        "all_significant_bonferroni": all_significant,
        "min_p_value": float(min(perm_results[c]["p_value"] for c in corruptions)),
        "max_p_value": float(max(perm_results[c]["p_value"] for c in corruptions))
    }
    print(f"  All significant at Bonferroni α={bonferroni_alpha}: {all_significant}")

    # Save
    out_path = "/workspace/Vizuara-VLA-Research/experiments/statistical_significance_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
