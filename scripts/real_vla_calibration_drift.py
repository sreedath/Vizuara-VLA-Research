#!/usr/bin/env python3
"""Experiment 410: Calibration Drift Analysis

Tests how the detector's calibration degrades over time as scene conditions
change gradually. Simulates deployment scenarios where the environment
evolves (lighting changes, weather transitions, camera degradation).

Tests:
1. Scene evolution: gradually modify the calibration scene and measure centroid staleness
2. Threshold robustness: at what point does a fixed threshold fail?
3. Centroid update strategies: static vs rolling-window vs exponential-moving-average
4. Drift detection: can we detect when recalibration is needed?
5. Mixed drift: scene changes + emerging corruption
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

def evolve_scene(base_arr, target_arr, alpha):
    """Interpolate between two scenes."""
    blended = base_arr * (1 - alpha) + target_arr * alpha
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

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

    # Generate base and target scenes
    rng = np.random.RandomState(42)
    base_arr = rng.randint(0, 255, (224, 224, 3)).astype(np.float32)
    target_arr = np.random.RandomState(999).randint(0, 255, (224, 224, 3)).astype(np.float32)

    base_img = Image.fromarray(base_arr.astype(np.uint8))
    target_img = Image.fromarray(target_arr.astype(np.uint8))

    # Calibrate on base scene
    print("Calibrating on base scene...")
    cal_emb = extract_hidden(model, processor, base_img, prompt)
    centroid = cal_emb.copy()

    results = {}

    # === Test 1: Scene Evolution ===
    print("\n=== Scene Evolution ===")
    alphas = np.linspace(0, 1, 21)  # 0% to 100% scene change
    evolution_data = []

    for alpha in alphas:
        evolved = evolve_scene(base_arr, target_arr, alpha)
        emb = extract_hidden(model, processor, evolved, prompt)
        dist_to_cal = cosine_dist(emb, centroid)

        # Also test corruption detection at this evolution stage
        corrupted = apply_corruption(evolved, 'fog')
        corrupt_emb = extract_hidden(model, processor, corrupted, prompt)
        corrupt_dist = cosine_dist(corrupt_emb, centroid)

        margin = corrupt_dist - dist_to_cal
        auroc = compute_auroc([dist_to_cal], [corrupt_dist])

        evolution_data.append({
            "alpha": float(alpha),
            "clean_dist": float(dist_to_cal),
            "corrupt_dist": float(corrupt_dist),
            "margin": float(margin),
            "auroc": float(auroc)
        })
        print(f"  alpha={alpha:.2f}: clean={dist_to_cal:.6f}, corrupt={corrupt_dist:.6f}, margin={margin:.6f}, auroc={auroc:.4f}")

    results["scene_evolution"] = evolution_data

    # Find when detection fails
    fail_alpha = None
    for d in evolution_data:
        if d["auroc"] < 1.0:
            fail_alpha = d["alpha"]
            break
    results["first_fail_alpha"] = fail_alpha
    print(f"\n  First detection failure at alpha={fail_alpha}")

    # === Test 2: Threshold robustness ===
    print("\n=== Threshold Robustness ===")
    initial_clean_dist = evolution_data[0]["clean_dist"]
    initial_corrupt_dist = evolution_data[0]["corrupt_dist"]
    threshold = (initial_clean_dist + initial_corrupt_dist) / 2

    threshold_results = []
    for d in evolution_data:
        fp = 1 if d["clean_dist"] > threshold else 0
        fn = 1 if d["corrupt_dist"] <= threshold else 0
        threshold_results.append({
            "alpha": d["alpha"],
            "fp": fp,
            "fn": fn,
            "clean_above_thresh": d["clean_dist"] > threshold,
            "corrupt_above_thresh": d["corrupt_dist"] > threshold
        })

    first_fp_alpha = None
    for t in threshold_results:
        if t["fp"] == 1:
            first_fp_alpha = t["alpha"]
            break

    results["threshold_robustness"] = {
        "initial_threshold": float(threshold),
        "results": threshold_results,
        "first_fp_alpha": first_fp_alpha,
        "total_fps": sum(t["fp"] for t in threshold_results),
        "total_fns": sum(t["fn"] for t in threshold_results)
    }
    print(f"  Threshold: {threshold:.6f}")
    print(f"  First FP at alpha={first_fp_alpha}")
    print(f"  Total FPs={sum(t['fp'] for t in threshold_results)}, FNs={sum(t['fn'] for t in threshold_results)}")

    # === Test 3: Centroid update strategies ===
    print("\n=== Centroid Update Strategies ===")

    strategies = {
        "static": [],
        "rolling_5": [],
        "rolling_10": [],
        "ema_0.1": [],
        "ema_0.3": [],
    }

    static_centroid = cal_emb.copy()
    ema01_centroid = cal_emb.copy()
    ema03_centroid = cal_emb.copy()
    rolling5_buffer = [cal_emb.copy()]
    rolling10_buffer = [cal_emb.copy()]

    for alpha in alphas:
        evolved = evolve_scene(base_arr, target_arr, alpha)
        emb = extract_hidden(model, processor, evolved, prompt)
        corrupted = apply_corruption(evolved, 'fog')
        corrupt_emb = extract_hidden(model, processor, corrupted, prompt)

        # Static
        d_clean = cosine_dist(emb, static_centroid)
        d_corrupt = cosine_dist(corrupt_emb, static_centroid)
        strategies["static"].append({
            "alpha": float(alpha),
            "clean_dist": float(d_clean),
            "corrupt_dist": float(d_corrupt),
            "auroc": float(compute_auroc([d_clean], [d_corrupt]))
        })

        # Rolling 5
        rolling5_buffer.append(emb.copy())
        if len(rolling5_buffer) > 5:
            rolling5_buffer.pop(0)
        r5_centroid = np.mean(rolling5_buffer, axis=0)
        d_clean = cosine_dist(emb, r5_centroid)
        d_corrupt = cosine_dist(corrupt_emb, r5_centroid)
        strategies["rolling_5"].append({
            "alpha": float(alpha),
            "clean_dist": float(d_clean),
            "corrupt_dist": float(d_corrupt),
            "auroc": float(compute_auroc([d_clean], [d_corrupt]))
        })

        # Rolling 10
        rolling10_buffer.append(emb.copy())
        if len(rolling10_buffer) > 10:
            rolling10_buffer.pop(0)
        r10_centroid = np.mean(rolling10_buffer, axis=0)
        d_clean = cosine_dist(emb, r10_centroid)
        d_corrupt = cosine_dist(corrupt_emb, r10_centroid)
        strategies["rolling_10"].append({
            "alpha": float(alpha),
            "clean_dist": float(d_clean),
            "corrupt_dist": float(d_corrupt),
            "auroc": float(compute_auroc([d_clean], [d_corrupt]))
        })

        # EMA 0.1
        ema01_centroid = 0.9 * ema01_centroid + 0.1 * emb
        d_clean = cosine_dist(emb, ema01_centroid)
        d_corrupt = cosine_dist(corrupt_emb, ema01_centroid)
        strategies["ema_0.1"].append({
            "alpha": float(alpha),
            "clean_dist": float(d_clean),
            "corrupt_dist": float(d_corrupt),
            "auroc": float(compute_auroc([d_clean], [d_corrupt]))
        })

        # EMA 0.3
        ema03_centroid = 0.7 * ema03_centroid + 0.3 * emb
        d_clean = cosine_dist(emb, ema03_centroid)
        d_corrupt = cosine_dist(corrupt_emb, ema03_centroid)
        strategies["ema_0.3"].append({
            "alpha": float(alpha),
            "clean_dist": float(d_clean),
            "corrupt_dist": float(d_corrupt),
            "auroc": float(compute_auroc([d_clean], [d_corrupt]))
        })

    strategy_summary = {}
    for name, data in strategies.items():
        aurocs = [d["auroc"] for d in data]
        strategy_summary[name] = {
            "mean_auroc": float(np.mean(aurocs)),
            "min_auroc": float(np.min(aurocs)),
            "n_perfect": sum(1 for a in aurocs if a >= 1.0),
            "n_total": len(aurocs)
        }
        print(f"  {name}: mean_auroc={np.mean(aurocs):.4f}, min={np.min(aurocs):.4f}, perfect={sum(1 for a in aurocs if a >= 1.0)}/{len(aurocs)}")

    results["centroid_strategies"] = strategies
    results["strategy_summary"] = strategy_summary

    # === Test 4: Drift detection ===
    print("\n=== Drift Detection ===")
    drift_data = []
    for alpha in alphas:
        evolved = evolve_scene(base_arr, target_arr, alpha)
        emb = extract_hidden(model, processor, evolved, prompt)
        drift_dist = cosine_dist(emb, cal_emb)

        corrupted = apply_corruption(evolved, 'fog')
        corrupt_emb = extract_hidden(model, processor, corrupted, prompt)
        detection_margin = cosine_dist(corrupt_emb, cal_emb) - drift_dist

        drift_data.append({
            "alpha": float(alpha),
            "drift_distance": float(drift_dist),
            "detection_margin": float(detection_margin),
            "margin_positive": bool(detection_margin > 0)
        })

    results["drift_detection"] = drift_data

    neg_margin_alpha = None
    for d in drift_data:
        if not d["margin_positive"]:
            neg_margin_alpha = d["alpha"]
            break
    results["margin_collapse_alpha"] = neg_margin_alpha
    print(f"  Margin collapses at alpha={neg_margin_alpha}")

    # === Test 5: Multiple corruption types under drift ===
    print("\n=== Multi-Corruption Under Drift ===")
    corruption_drift = {}
    for c in ['fog', 'night', 'noise', 'blur']:
        c_aurocs = []
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            evolved = evolve_scene(base_arr, target_arr, alpha)
            emb = extract_hidden(model, processor, evolved, prompt)
            corrupted = apply_corruption(evolved, c)
            corrupt_emb = extract_hidden(model, processor, corrupted, prompt)

            clean_dist = cosine_dist(emb, cal_emb)
            corrupt_dist = cosine_dist(corrupt_emb, cal_emb)
            auroc = compute_auroc([clean_dist], [corrupt_dist])
            c_aurocs.append({"alpha": float(alpha), "auroc": float(auroc),
                            "clean_dist": float(clean_dist), "corrupt_dist": float(corrupt_dist)})

        corruption_drift[c] = c_aurocs
        auroc_str = ", ".join(f"{d['alpha']:.2f}:{d['auroc']:.4f}" for d in c_aurocs)
        print(f"  {c}: {auroc_str}")

    results["corruption_drift"] = corruption_drift

    # Save
    out_path = "/workspace/Vizuara-VLA-Research/experiments/calibration_drift_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
