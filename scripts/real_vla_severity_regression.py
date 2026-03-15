#!/usr/bin/env python3
"""Experiment 415: Corruption Severity Regression from Embeddings

Tests whether the embedding distance can predict corruption SEVERITY
(not just detect corruption), and whether different corruptions follow
different distance-severity relationships (linear, sublinear, superlinear).

Tests:
1. Distance vs severity curves for each corruption (fine-grained 20 levels)
2. Regression fit quality (R², linear vs polynomial)
3. Cross-corruption severity calibration (can one corruption's curve predict another's?)
4. Severity ordering preservation (is higher severity always further?)
5. Just-noticeable-difference: minimum severity step detectable by distance
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

def main():
    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    corruptions = ['fog', 'night', 'noise', 'blur']

    # Generate scenes
    seeds = [42, 123, 456, 789, 999]
    scenes = [Image.fromarray(np.random.RandomState(s).randint(0, 255, (224, 224, 3), dtype=np.uint8)) for s in seeds]

    # Calibration
    print("Extracting clean embeddings...")
    clean_embs = [extract_hidden(model, processor, s, prompt) for s in scenes]
    centroid = np.mean(clean_embs, axis=0)
    clean_dists = [cosine_dist(e, centroid) for e in clean_embs]

    results = {}

    # === Test 1: Fine-grained severity curves ===
    print("\n=== Severity Curves (20 levels) ===")
    severity_levels = np.linspace(0.05, 1.0, 20)
    severity_curves = {}

    for c in corruptions:
        print(f"\n  {c}:")
        curve = {}
        for sev in severity_levels:
            dists = []
            for s in scenes:
                corrupted = apply_corruption(s, c, float(sev))
                emb = extract_hidden(model, processor, corrupted, prompt)
                dists.append(cosine_dist(emb, centroid))
            curve[f"{sev:.3f}"] = {
                "mean_dist": float(np.mean(dists)),
                "std_dist": float(np.std(dists)),
                "min_dist": float(np.min(dists)),
                "max_dist": float(np.max(dists)),
            }
            print(f"    sev={sev:.3f}: dist={np.mean(dists):.6f} ± {np.std(dists):.6f}")
        severity_curves[c] = curve
    results["severity_curves"] = severity_curves

    # === Test 2: Regression quality ===
    print("\n=== Regression Fit ===")
    regression = {}
    for c in corruptions:
        sevs = []
        mean_dists = []
        for sev_str, data in severity_curves[c].items():
            sevs.append(float(sev_str))
            mean_dists.append(data["mean_dist"])
        sevs = np.array(sevs)
        mean_dists = np.array(mean_dists)

        # Linear fit
        if np.std(sevs) > 1e-10 and np.std(mean_dists) > 1e-10:
            lin_coeffs = np.polyfit(sevs, mean_dists, 1)
            lin_pred = np.polyval(lin_coeffs, sevs)
            ss_res_lin = np.sum((mean_dists - lin_pred) ** 2)
            ss_tot = np.sum((mean_dists - np.mean(mean_dists)) ** 2)
            r2_linear = 1 - ss_res_lin / (ss_tot + 1e-10)

            # Quadratic fit
            quad_coeffs = np.polyfit(sevs, mean_dists, 2)
            quad_pred = np.polyval(quad_coeffs, sevs)
            ss_res_quad = np.sum((mean_dists - quad_pred) ** 2)
            r2_quadratic = 1 - ss_res_quad / (ss_tot + 1e-10)

            # Monotonicity check
            diffs = np.diff(mean_dists)
            monotonic = bool(np.all(diffs >= -1e-8))
            n_inversions = int(np.sum(diffs < -1e-8))
        else:
            r2_linear = 0.0
            r2_quadratic = 0.0
            lin_coeffs = [0, 0]
            quad_coeffs = [0, 0, 0]
            monotonic = True
            n_inversions = 0

        regression[c] = {
            "r2_linear": float(r2_linear),
            "r2_quadratic": float(r2_quadratic),
            "linear_slope": float(lin_coeffs[0]),
            "linear_intercept": float(lin_coeffs[1]),
            "quad_coeffs": [float(x) for x in quad_coeffs],
            "monotonic": monotonic,
            "n_inversions": n_inversions,
        }
        print(f"  {c}: R²_lin={r2_linear:.4f}, R²_quad={r2_quadratic:.4f}, "
              f"monotonic={monotonic}, inversions={n_inversions}")
    results["regression"] = regression

    # === Test 3: Cross-corruption severity prediction ===
    print("\n=== Cross-Corruption Calibration ===")
    cross_cal = {}
    for c_train in corruptions:
        train_sevs = []
        train_dists = []
        for sev_str, data in severity_curves[c_train].items():
            train_sevs.append(float(sev_str))
            train_dists.append(data["mean_dist"])
        train_sevs = np.array(train_sevs)
        train_dists = np.array(train_dists)

        # Fit linear model on training corruption
        if np.std(train_dists) > 1e-10:
            coeffs = np.polyfit(train_dists, train_sevs, 1)
        else:
            coeffs = [0, 0]

        for c_test in corruptions:
            if c_test == c_train:
                continue
            test_sevs = []
            test_dists = []
            for sev_str, data in severity_curves[c_test].items():
                test_sevs.append(float(sev_str))
                test_dists.append(data["mean_dist"])
            test_dists = np.array(test_dists)
            test_sevs = np.array(test_sevs)

            # Predict severity using training model
            pred_sevs = np.polyval(coeffs, test_dists)
            mae = float(np.mean(np.abs(pred_sevs - test_sevs)))
            rank_corr = float(np.corrcoef(np.argsort(test_dists), np.argsort(test_sevs))[0, 1]) \
                if np.std(test_dists) > 1e-10 else 0.0

            key = f"{c_train}_to_{c_test}"
            cross_cal[key] = {
                "mae": mae,
                "rank_correlation": rank_corr,
            }
            print(f"  {key}: MAE={mae:.4f}, rank_corr={rank_corr:.4f}")
    results["cross_calibration"] = cross_cal

    # === Test 4: Severity ordering preservation ===
    print("\n=== Severity Ordering ===")
    ordering = {}
    for c in corruptions:
        sevs_list = sorted(severity_curves[c].keys(), key=lambda x: float(x))
        dists_ordered = [severity_curves[c][s]["mean_dist"] for s in sevs_list]
        pairs_total = 0
        pairs_correct = 0
        for i in range(len(dists_ordered)):
            for j in range(i + 1, len(dists_ordered)):
                pairs_total += 1
                if dists_ordered[j] >= dists_ordered[i] - 1e-10:
                    pairs_correct += 1
        ordering[c] = {
            "pairs_total": pairs_total,
            "pairs_correct": pairs_correct,
            "ordering_accuracy": pairs_correct / max(pairs_total, 1),
        }
        print(f"  {c}: {pairs_correct}/{pairs_total} = {pairs_correct/max(pairs_total,1):.3f}")
    results["severity_ordering"] = ordering

    # === Test 5: Just-noticeable difference ===
    print("\n=== Just-Noticeable Difference ===")
    jnd = {}
    for c in corruptions:
        sevs_list = sorted(severity_curves[c].keys(), key=lambda x: float(x))
        dists_ordered = [severity_curves[c][s]["mean_dist"] for s in sevs_list]

        # Clean threshold: max clean distance
        clean_thresh = max(clean_dists)

        # Find first severity that consistently exceeds clean
        first_detectable = None
        for i, s in enumerate(sevs_list):
            min_d = severity_curves[c][s]["min_dist"]
            if min_d > clean_thresh:
                first_detectable = float(s)
                break

        # Find minimum severity step where consecutive levels are distinguishable
        min_step = None
        for i in range(len(dists_ordered) - 1):
            if dists_ordered[i + 1] > dists_ordered[i] + 1e-8:
                step = float(sevs_list[i + 1]) - float(sevs_list[i])
                if min_step is None or step < min_step:
                    min_step = step

        jnd[c] = {
            "first_detectable_severity": first_detectable,
            "min_distinguishable_step": min_step,
            "clean_threshold": float(clean_thresh),
        }
        print(f"  {c}: first_detect={first_detectable}, min_step={min_step}")
    results["just_noticeable_difference"] = jnd

    # Save
    out_path = "/workspace/Vizuara-VLA-Research/experiments/severity_regression_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
