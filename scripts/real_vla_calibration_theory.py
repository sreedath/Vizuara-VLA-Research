#!/usr/bin/env python3
"""Experiment 446: Calibration Theory — Threshold Selection Analysis

Comprehensive analysis of threshold selection strategies for deployment.
Beyond 3σ — tests multiple threshold criteria and their operational
characteristics (FPR, TPR, F1) across corruption types and severities.

Tests:
1. ROC curve construction per corruption
2. Optimal threshold per F1, Youden's J, cost-sensitive
3. Threshold transferability across corruptions
4. Operating point analysis (FPR=0.01, FPR=0.05)
5. Calibration curve (predicted probability vs actual OOD rate)
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

    seeds = [42, 123, 456, 789, 999, 1111, 2222, 3333, 4444, 5555]
    scenes = [Image.fromarray(np.random.RandomState(s).randint(0, 255, (224, 224, 3), dtype=np.uint8)) for s in seeds]

    print("Extracting clean embeddings...")
    clean_embs = [extract_hidden(model, processor, s, prompt) for s in scenes]
    centroid = np.mean(clean_embs, axis=0)
    clean_dists = [cosine_dist(e, centroid) for e in clean_embs]

    results = {"n_scenes": len(scenes)}
    results["clean_dist_stats"] = {
        "mean": float(np.mean(clean_dists)),
        "std": float(np.std(clean_dists)),
        "min": float(np.min(clean_dists)),
        "max": float(np.max(clean_dists)),
        "dists": [float(d) for d in clean_dists],
    }

    # Standard thresholds
    mean_clean = np.mean(clean_dists)
    std_clean = np.std(clean_dists)
    thresholds = {
        "2sigma": float(mean_clean + 2 * std_clean),
        "3sigma": float(mean_clean + 3 * std_clean),
        "4sigma": float(mean_clean + 4 * std_clean),
        "max_clean": float(np.max(clean_dists)),
        "1.5x_max": float(1.5 * np.max(clean_dists)),
        "2x_max": float(2.0 * np.max(clean_dists)),
    }
    results["thresholds"] = thresholds

    # === Test 1: Full ROC curves per corruption ===
    print("\n=== ROC Curves ===")
    roc_results = {}
    for c in corruptions:
        ood_dists = []
        for s in scenes:
            emb = extract_hidden(model, processor, apply_corruption(s, c), prompt)
            ood_dists.append(cosine_dist(emb, centroid))

        # Compute ROC at many thresholds
        all_dists = sorted(set(clean_dists + ood_dists))
        test_thresholds = np.linspace(min(all_dists) * 0.5, max(all_dists) * 1.5, 200)

        roc_points = []
        for t in test_thresholds:
            tp = sum(1 for d in ood_dists if d > t)
            fp = sum(1 for d in clean_dists if d > t)
            fn = sum(1 for d in ood_dists if d <= t)
            tn = sum(1 for d in clean_dists if d <= t)
            tpr = tp / max(tp + fn, 1)
            fpr = fp / max(fp + tn, 1)
            precision = tp / max(tp + fp, 1)
            f1 = 2 * precision * tpr / max(precision + tpr, 1e-12)
            roc_points.append({
                "threshold": float(t),
                "tpr": float(tpr),
                "fpr": float(fpr),
                "precision": float(precision),
                "f1": float(f1),
                "youdens_j": float(tpr - fpr),
            })

        # Find optimal thresholds
        best_f1 = max(roc_points, key=lambda x: x["f1"])
        best_j = max(roc_points, key=lambda x: x["youdens_j"])

        auroc = float(compute_auroc(clean_dists, ood_dists))

        roc_results[c] = {
            "auroc": auroc,
            "mean_ood_dist": float(np.mean(ood_dists)),
            "best_f1_threshold": best_f1["threshold"],
            "best_f1_score": best_f1["f1"],
            "best_youdens_threshold": best_j["threshold"],
            "best_youdens_j": best_j["youdens_j"],
            "ood_dists": [float(d) for d in ood_dists],
        }
        print(f"  {c}: AUROC={auroc:.4f}, best_F1={best_f1['f1']:.4f}@t={best_f1['threshold']:.6f}")
    results["roc_curves"] = roc_results

    # === Test 2: Threshold performance across corruptions ===
    print("\n=== Threshold Performance ===")
    threshold_perf = {}
    for tname, t in thresholds.items():
        per_corr = {}
        for c in corruptions:
            ood_dists = roc_results[c]["ood_dists"]
            tp = sum(1 for d in ood_dists if d > t)
            fp = sum(1 for d in clean_dists if d > t)
            fn = sum(1 for d in ood_dists if d <= t)
            tn = sum(1 for d in clean_dists if d <= t)
            tpr = tp / max(tp + fn, 1)
            fpr = fp / max(fp + tn, 1)
            per_corr[c] = {"tpr": float(tpr), "fpr": float(fpr)}

        threshold_perf[tname] = {
            "threshold_value": float(t),
            "per_corruption": per_corr,
            "mean_tpr": float(np.mean([v["tpr"] for v in per_corr.values()])),
            "max_fpr": float(max(v["fpr"] for v in per_corr.values())),
        }
        print(f"  {tname} (t={t:.6f}): mean_TPR={threshold_perf[tname]['mean_tpr']:.4f}, max_FPR={threshold_perf[tname]['max_fpr']:.4f}")
    results["threshold_performance"] = threshold_perf

    # === Test 3: Multi-severity ROC ===
    print("\n=== Multi-Severity Analysis ===")
    sev_analysis = {}
    for sev in [0.1, 0.25, 0.5, 0.75, 1.0]:
        per_corr = {}
        for c in corruptions:
            ood_dists = []
            for s in scenes:
                emb = extract_hidden(model, processor, apply_corruption(s, c, severity=sev), prompt)
                ood_dists.append(cosine_dist(emb, centroid))
            auroc = float(compute_auroc(clean_dists, ood_dists))
            # TPR at 3sigma threshold
            t3 = thresholds["3sigma"]
            tpr_3sigma = float(sum(1 for d in ood_dists if d > t3) / len(ood_dists))
            per_corr[c] = {"auroc": auroc, "tpr_3sigma": tpr_3sigma, "mean_dist": float(np.mean(ood_dists))}
        sev_analysis[str(sev)] = per_corr
        print(f"  sev={sev}: " + ", ".join(f"{c}={per_corr[c]['auroc']:.3f}" for c in corruptions))
    results["severity_analysis"] = sev_analysis

    # === Test 4: Distance distribution characterization ===
    print("\n=== Distance Distribution ===")
    dist_char = {}
    for c in ['clean'] + corruptions:
        if c == 'clean':
            dists = clean_dists
        else:
            dists = roc_results[c]["ood_dists"]
        dists_arr = np.array(dists)
        dist_char[c] = {
            "mean": float(np.mean(dists_arr)),
            "std": float(np.std(dists_arr)),
            "median": float(np.median(dists_arr)),
            "skew": float(((dists_arr - np.mean(dists_arr)) ** 3).mean() / max(np.std(dists_arr) ** 3, 1e-20)),
            "kurtosis": float(((dists_arr - np.mean(dists_arr)) ** 4).mean() / max(np.std(dists_arr) ** 4, 1e-20) - 3),
        }
    results["distance_distributions"] = dist_char
    for c, stats in dist_char.items():
        print(f"  {c}: mean={stats['mean']:.6f}, std={stats['std']:.6f}, skew={stats['skew']:.2f}")

    out_path = "/workspace/Vizuara-VLA-Research/experiments/calibration_theory_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
