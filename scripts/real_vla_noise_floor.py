#!/usr/bin/env python3
"""Experiment 453: Detection Noise Floor Analysis

Precisely characterizes the noise floor of the detection system — how much
natural variation exists between clean embeddings, and what are the theoretical
limits of detection. Critical for understanding the fundamental limitations of
the centroid-based OOD detector.

Analyses:
1. Clean-Clean Distance Distribution: pairwise distances between 30 different
   clean scenes, establishing the empirical noise floor.
2. Within-Scene Reproducibility: variance of repeated forward passes on the
   same image (should be zero in deterministic eval mode).
3. Corruption Distance vs Noise Floor: at each severity, compare corruption
   signal against the clean noise floor; find detectability threshold.
4. Signal-to-Noise Ratio: SNR = (corruption_dist - mean_clean) / std_clean
   for each corruption type and severity.
5. Minimum Detectable Severity: finest severity passing the 3-sigma threshold.
6. Noise Floor Stability: running mean/std as calibration set size grows.
"""

import torch
import json
import os
import sys
import numpy as np
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor
from datetime import datetime


# ---------------------------------------------------------------------------
# Standard helpers
# ---------------------------------------------------------------------------

def make_image(seed=42):
    rng = np.random.RandomState(seed)
    return Image.fromarray(rng.randint(0, 255, (224, 224, 3), dtype=np.uint8))


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
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Hand-written AUROC with trapezoid fallback
# ---------------------------------------------------------------------------

def compute_auroc(id_scores, ood_scores):
    """Compute AUROC. Uses exact Wilcoxon-Mann-Whitney count when possible,
    falls back to trapezoidal rule on the ROC curve for large arrays."""
    id_s = np.asarray(id_scores, dtype=np.float64)
    ood_s = np.asarray(ood_scores, dtype=np.float64)
    n_id = len(id_s)
    n_ood = len(ood_s)

    if n_id == 0 or n_ood == 0:
        return 0.5

    # For manageable sizes use exact WMW count (O(n*m))
    if n_id * n_ood <= 50_000:
        count = sum(
            float(np.sum(o > id_s) + 0.5 * np.sum(o == id_s))
            for o in ood_s
        )
        return count / (n_id * n_ood)

    # Trapezoid fallback: build ROC curve
    all_scores = np.concatenate([id_s, ood_s])
    labels = np.concatenate([np.zeros(n_id), np.ones(n_ood)])
    order = np.argsort(-all_scores)
    labels_sorted = labels[order]

    tpr_list = [0.0]
    fpr_list = [0.0]
    tp = 0
    fp = 0
    for lbl in labels_sorted:
        if lbl == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / n_ood)
        fpr_list.append(fp / n_id)

    # Trapezoidal integration
    area = 0.0
    for i in range(1, len(fpr_list)):
        dx = fpr_list[i] - fpr_list[i - 1]
        area += dx * (tpr_list[i] + tpr_list[i - 1]) / 2.0
    return float(area)


# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------

# 30 seeds: 42 (first), then 50, 100, 150, ..., 1450 in steps of 50 (29 more).
# The spec lists "42, 50, 100, ..., 1500" and notes 30 evenly-spaced seeds;
# including 42 as the anchor seed gives 30 total with the step-50 tail ending at 1450.
SCENE_SEEDS = [42] + list(range(50, 1451, 50))  # length = 30
assert len(SCENE_SEEDS) == 30, f"Expected 30 seeds, got {len(SCENE_SEEDS)}"

CORRUPTION_SCENE_SEEDS = SCENE_SEEDS[:8]  # first 8 for corruption analysis

CORRUPTION_TYPES = ['fog', 'night', 'noise', 'blur']

# Severities for the main corruption vs noise-floor sweep
SEVERITIES = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0]

# Fine granularity for minimum detectable severity search
FINE_SEVERITIES = [round(s, 2) for s in np.arange(0.01, 0.21, 0.01)]

# Calibration set sizes for stability analysis
CALIBRATION_NS = [2, 5, 10, 15, 20, 25, 30]

PROMPT = "In: What action should the robot take to pick up the object?\nOut:"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "experiment": "noise_floor_analysis",
        "experiment_number": 453,
        "timestamp": ts,
        "scene_seeds": SCENE_SEEDS,
        "corruption_scene_seeds": CORRUPTION_SCENE_SEEDS,
        "corruption_types": CORRUPTION_TYPES,
        "severities": SEVERITIES,
        "fine_severities": FINE_SEVERITIES,
        "calibration_ns": CALIBRATION_NS,
    }

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    print("Loading OpenVLA-7B...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        "openvla/openvla-7b", trust_remote_code=True
    )
    model.eval()
    print("Model loaded.\n")

    # -----------------------------------------------------------------------
    # Pre-compute clean embeddings for all 30 scenes
    # -----------------------------------------------------------------------
    print("=== Pre-computing clean embeddings for 30 scenes ===")
    clean_embs = []
    for i, seed in enumerate(SCENE_SEEDS):
        img = make_image(seed)
        emb = extract_hidden(model, processor, img, PROMPT)
        clean_embs.append(emb)
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  Scene {i+1}/30 (seed={seed}) done.")
    clean_embs = np.array(clean_embs)  # (30, D)
    print()

    # -----------------------------------------------------------------------
    # Analysis 1: Clean-Clean Distance Distribution
    # -----------------------------------------------------------------------
    print("=== Analysis 1: Clean-Clean Distance Distribution ===")
    n_scenes = len(clean_embs)
    pairwise_dists = []
    for i in range(n_scenes):
        for j in range(i + 1, n_scenes):
            pairwise_dists.append(cosine_dist(clean_embs[i], clean_embs[j]))

    pairwise_dists = np.array(pairwise_dists)
    percentiles = np.percentile(pairwise_dists, [5, 25, 50, 75, 95]).tolist()

    noise_floor_mean = float(np.mean(pairwise_dists))
    noise_floor_std = float(np.std(pairwise_dists))
    noise_floor_min = float(np.min(pairwise_dists))
    noise_floor_max = float(np.max(pairwise_dists))

    results["clean_clean_distribution"] = {
        "n_pairs": int(len(pairwise_dists)),
        "mean": noise_floor_mean,
        "std": noise_floor_std,
        "min": noise_floor_min,
        "max": noise_floor_max,
        "percentile_5": percentiles[0],
        "percentile_25": percentiles[1],
        "percentile_50": percentiles[2],
        "percentile_75": percentiles[3],
        "percentile_95": percentiles[4],
        "interpretation": (
            "Pairwise cosine distances between 30 different clean random scenes. "
            "Mean is the empirical noise floor; std captures its spread."
        ),
    }

    print(f"  Pairs: {len(pairwise_dists)}")
    print(f"  Mean: {noise_floor_mean:.6f}  Std: {noise_floor_std:.6f}")
    print(f"  Min:  {noise_floor_min:.6f}  Max: {noise_floor_max:.6f}")
    print(f"  Percentiles [5,25,50,75,95]: {[f'{p:.6f}' for p in percentiles]}")
    print()

    # -----------------------------------------------------------------------
    # Analysis 2: Within-Scene Reproducibility
    # -----------------------------------------------------------------------
    print("=== Analysis 2: Within-Scene Reproducibility (10 repeated passes) ===")
    repro_img = make_image(seed=42)
    repro_embs = []
    for run in range(10):
        emb = extract_hidden(model, processor, repro_img, PROMPT)
        repro_embs.append(emb)
    repro_embs = np.array(repro_embs)  # (10, D)

    repro_pairwise = []
    for i in range(10):
        for j in range(i + 1, 10):
            repro_pairwise.append(cosine_dist(repro_embs[i], repro_embs[j]))
    repro_pairwise = np.array(repro_pairwise)

    # Per-dimension variance across the 10 runs
    per_dim_var = float(np.mean(np.var(repro_embs, axis=0)))
    max_dim_var = float(np.max(np.var(repro_embs, axis=0)))

    results["within_scene_reproducibility"] = {
        "n_runs": 10,
        "n_pairs": int(len(repro_pairwise)),
        "pairwise_dist_mean": float(np.mean(repro_pairwise)),
        "pairwise_dist_max": float(np.max(repro_pairwise)),
        "pairwise_dist_std": float(np.std(repro_pairwise)),
        "per_dim_variance_mean": per_dim_var,
        "per_dim_variance_max": max_dim_var,
        "is_deterministic": bool(float(np.max(repro_pairwise)) < 1e-10),
        "interpretation": (
            "10 repeated forward passes on the same image. "
            "Should be exactly zero if model is deterministic in eval mode."
        ),
    }

    print(f"  Pairwise dist mean: {results['within_scene_reproducibility']['pairwise_dist_mean']:.2e}")
    print(f"  Pairwise dist max:  {results['within_scene_reproducibility']['pairwise_dist_max']:.2e}")
    print(f"  Per-dim variance mean: {per_dim_var:.2e}   max: {max_dim_var:.2e}")
    print(f"  Deterministic: {results['within_scene_reproducibility']['is_deterministic']}")
    print()

    # -----------------------------------------------------------------------
    # Pre-compute clean embeddings for the 8 corruption scenes
    # -----------------------------------------------------------------------
    corr_clean_embs = clean_embs[:8]  # reuse already-computed

    # Centroid over the 8 corruption scenes
    corr_centroid = np.mean(corr_clean_embs, axis=0)

    # Clean distances to centroid (for SNR denominator)
    corr_clean_dists_to_centroid = np.array([
        cosine_dist(e, corr_centroid) for e in corr_clean_embs
    ])
    mean_clean_d = float(np.mean(corr_clean_dists_to_centroid))
    std_clean_d = float(np.std(corr_clean_dists_to_centroid))

    results["corruption_baseline"] = {
        "n_scenes": 8,
        "seeds": CORRUPTION_SCENE_SEEDS,
        "centroid_mean_clean_dist": mean_clean_d,
        "centroid_std_clean_dist": std_clean_d,
        "threshold_3sigma": mean_clean_d + 3.0 * std_clean_d,
    }

    # -----------------------------------------------------------------------
    # Analysis 3: Corruption Distance vs Noise Floor
    # -----------------------------------------------------------------------
    print("=== Analysis 3: Corruption Distance vs Noise Floor ===")
    # For each corruption × severity: mean distance from centroid
    corruption_vs_floor = {}

    for ctype in CORRUPTION_TYPES:
        print(f"  Corruption: {ctype}")
        severity_results = []

        for sev in SEVERITIES:
            scene_dists = []
            for seed in CORRUPTION_SCENE_SEEDS:
                img = make_image(seed)
                cimg = apply_corruption(img, ctype, sev)
                emb = extract_hidden(model, processor, cimg, PROMPT)
                scene_dists.append(cosine_dist(emb, corr_centroid))

            mean_d = float(np.mean(scene_dists))
            above_floor = mean_d > noise_floor_mean

            severity_results.append({
                "severity": sev,
                "mean_corruption_dist": mean_d,
                "std_corruption_dist": float(np.std(scene_dists)),
                "above_noise_floor_mean": above_floor,
                "margin_over_floor": float(mean_d - noise_floor_mean),
            })
            print(f"    sev={sev:.2f}: mean_dist={mean_d:.6f}  "
                  f"above_floor={above_floor}  margin={mean_d - noise_floor_mean:+.6f}")

        # Find first severity where corruption distance exceeds noise floor mean
        detectability_threshold_sev = None
        for sr in severity_results:
            if sr["above_noise_floor_mean"]:
                detectability_threshold_sev = sr["severity"]
                break

        corruption_vs_floor[ctype] = {
            "severity_results": severity_results,
            "detectability_threshold_severity": detectability_threshold_sev,
        }

    results["corruption_vs_noise_floor"] = corruption_vs_floor
    print()

    # -----------------------------------------------------------------------
    # Analysis 4: Signal-to-Noise Ratio
    # -----------------------------------------------------------------------
    print("=== Analysis 4: Signal-to-Noise Ratio (SNR) ===")
    snr_analysis = {}
    sigma_thresh = std_clean_d if std_clean_d > 1e-12 else 1.0  # guard against degenerate

    for ctype in CORRUPTION_TYPES:
        print(f"  Corruption: {ctype}")
        snr_results = []

        for sev in SEVERITIES:
            scene_dists = []
            for seed in CORRUPTION_SCENE_SEEDS:
                img = make_image(seed)
                cimg = apply_corruption(img, ctype, sev)
                emb = extract_hidden(model, processor, cimg, PROMPT)
                scene_dists.append(cosine_dist(emb, corr_centroid))

            mean_d = float(np.mean(scene_dists))
            snr = (mean_d - mean_clean_d) / sigma_thresh if sigma_thresh > 1e-12 else float('inf')
            reliable = snr > 3.0

            snr_results.append({
                "severity": sev,
                "mean_dist": mean_d,
                "snr": float(snr),
                "reliable_detection": reliable,
            })
            print(f"    sev={sev:.2f}: mean_dist={mean_d:.6f}  SNR={snr:.2f}  reliable={reliable}")

        snr_analysis[ctype] = snr_results

    results["snr_analysis"] = snr_analysis
    print()

    # -----------------------------------------------------------------------
    # Analysis 5: Minimum Detectable Severity (3-sigma threshold, fine grid)
    # -----------------------------------------------------------------------
    print("=== Analysis 5: Minimum Detectable Severity (3-sigma threshold) ===")
    three_sigma_thresh = mean_clean_d + 3.0 * sigma_thresh
    print(f"  3-sigma threshold: {three_sigma_thresh:.6f}")

    min_detectable = {}

    for ctype in CORRUPTION_TYPES:
        print(f"  Corruption: {ctype}")
        fine_results = []
        min_sev = None

        for sev in FINE_SEVERITIES:
            scene_dists = []
            for seed in CORRUPTION_SCENE_SEEDS:
                img = make_image(seed)
                cimg = apply_corruption(img, ctype, sev)
                emb = extract_hidden(model, processor, cimg, PROMPT)
                scene_dists.append(cosine_dist(emb, corr_centroid))

            mean_d = float(np.mean(scene_dists))
            flagged = mean_d > three_sigma_thresh

            fine_results.append({
                "severity": sev,
                "mean_dist": mean_d,
                "flagged_ood": flagged,
                "excess_over_threshold": float(mean_d - three_sigma_thresh),
            })

            if flagged and min_sev is None:
                min_sev = sev
                print(f"    ** First detection at severity={sev:.2f} "
                      f"(dist={mean_d:.6f}) **")

        min_detectable[ctype] = {
            "three_sigma_threshold": float(three_sigma_thresh),
            "min_detectable_severity": min_sev,
            "fine_results": fine_results,
        }

        if min_sev is None:
            print(f"    No detection in range [0.01, 0.20] for {ctype}")

    results["min_detectable_severity"] = min_detectable
    print()

    # -----------------------------------------------------------------------
    # Analysis 6: Noise Floor Stability
    # -----------------------------------------------------------------------
    print("=== Analysis 6: Noise Floor Stability vs Calibration Size ===")
    stability_results = []

    for n in CALIBRATION_NS:
        subset = clean_embs[:n]  # first n scenes
        centroid_n = np.mean(subset, axis=0)

        # Pairwise distances within subset
        pair_dists = []
        for i in range(n):
            for j in range(i + 1, n):
                pair_dists.append(cosine_dist(subset[i], subset[j]))

        if len(pair_dists) == 0:
            # n == 1: no pairs
            running_mean = 0.0
            running_std = 0.0
        else:
            pair_dists = np.array(pair_dists)
            running_mean = float(np.mean(pair_dists))
            running_std = float(np.std(pair_dists))

        # Distances to centroid for stability measure
        dists_to_centroid = [cosine_dist(e, centroid_n) for e in subset]
        centroid_mean = float(np.mean(dists_to_centroid))
        centroid_std = float(np.std(dists_to_centroid))

        entry = {
            "n_scenes": n,
            "n_pairs": int(len(pair_dists)) if isinstance(pair_dists, np.ndarray) else 0,
            "pairwise_mean": running_mean,
            "pairwise_std": running_std,
            "centroid_dist_mean": centroid_mean,
            "centroid_dist_std": centroid_std,
        }
        stability_results.append(entry)
        print(f"  N={n:2d}: pairwise mean={running_mean:.6f}  std={running_std:.6f}  "
              f"centroid_std={centroid_std:.6f}")

    results["noise_floor_stability"] = stability_results
    print()

    # -----------------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------------
    summary = {
        "noise_floor_mean": noise_floor_mean,
        "noise_floor_std": noise_floor_std,
        "three_sigma_threshold": float(three_sigma_thresh),
        "within_scene_max_dist": float(
            results["within_scene_reproducibility"]["pairwise_dist_max"]
        ),
        "is_deterministic": results["within_scene_reproducibility"]["is_deterministic"],
    }

    # Detectability thresholds per corruption
    detectability = {}
    for ctype in CORRUPTION_TYPES:
        detectability[ctype] = {
            "noise_floor_cross": corruption_vs_floor[ctype]["detectability_threshold_severity"],
            "min_detectable_3sigma": min_detectable[ctype]["min_detectable_severity"],
        }
    summary["detectability_thresholds"] = detectability

    # SNR > 3 crossing for each corruption
    snr3_crossings = {}
    for ctype in CORRUPTION_TYPES:
        cross = None
        for entry in snr_analysis[ctype]:
            if entry["reliable_detection"]:
                cross = entry["severity"]
                break
        snr3_crossings[ctype] = cross
    summary["snr3_crossing_severity"] = snr3_crossings

    results["summary"] = summary

    print("=== Summary ===")
    print(f"  Noise floor mean:  {noise_floor_mean:.6f}")
    print(f"  Noise floor std:   {noise_floor_std:.6f}")
    print(f"  3-sigma threshold: {three_sigma_thresh:.6f}")
    print(f"  Deterministic model: {results['within_scene_reproducibility']['is_deterministic']}")
    for ctype in CORRUPTION_TYPES:
        nf_cross = detectability[ctype]["noise_floor_cross"]
        s3_cross = detectability[ctype]["min_detectable_3sigma"]
        snr3 = snr3_crossings[ctype]
        print(f"  {ctype}: floor_cross={nf_cross}  3sigma={s3_cross}  snr3={snr3}")
    print()

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "experiments",
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"noise_floor_{ts}.json")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
