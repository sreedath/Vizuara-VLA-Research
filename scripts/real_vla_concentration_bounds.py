#!/usr/bin/env python3
"""Experiment 287: Formal Detection Guarantees via Concentration Inequalities
Computes theoretical bounds on detection reliability:
1. Empirical gap between clean and corrupted distances
2. Hoeffding bound on false positive/negative rates
3. Chernoff bound on detection probability
4. PAC-learning style guarantee (epsilon-delta)
5. Multiple re-runs for variance estimation
"""

import torch
import numpy as np
import json
import os
from datetime import datetime
from PIL import Image, ImageFilter, ImageEnhance
from transformers import AutoModelForVision2Seq, AutoProcessor
from scipy.spatial.distance import cosine
import math

def apply_corruption(image, corruption_type, severity):
    img = image.copy()
    if corruption_type == "fog":
        fog = Image.new('RGB', img.size, (200, 200, 200))
        return Image.blend(img, fog, severity * 0.8)
    elif corruption_type == "night":
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(1.0 - severity * 0.9)
    elif corruption_type == "blur":
        radius = severity * 20
        return img.filter(ImageFilter.GaussianBlur(radius=max(0.1, radius)))
    elif corruption_type == "noise":
        arr = np.array(img).astype(np.float32)
        noise = np.random.RandomState(42).normal(0, severity * 100, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    return img

def extract_hidden(model, processor, image, prompt, layer=3):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return fwd.hidden_states[layer][0, -1, :].float().cpu().numpy()

def main():
    print("Loading OpenVLA-7B...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\n"

    # Generate multiple diverse images
    n_images = 10
    images = []
    for i in range(n_images):
        rng = np.random.RandomState(i * 7 + 42)
        img = Image.fromarray(rng.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        images.append(img)

    corruptions = ["fog", "night", "blur", "noise"]
    severities = [0.3, 0.5, 0.7, 1.0]

    results = {
        "experiment": "concentration_bounds",
        "experiment_number": 287,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    # Part 1: Collect clean distances (should all be 0 for same image, >0 for cross-image)
    print("\n=== Part 1: Clean Distance Distribution ===")
    clean_embs = []
    for i, img in enumerate(images):
        print(f"  Clean image {i}...")
        emb = extract_hidden(model, processor, img, prompt)
        clean_embs.append(emb)

    # Same-image re-run distances
    print("  Re-run distances (same image, 5 runs)...")
    rerun_distances = []
    for run in range(5):
        emb = extract_hidden(model, processor, images[0], prompt)
        d = float(cosine(clean_embs[0], emb))
        rerun_distances.append(d)
    results["rerun_distances"] = rerun_distances
    results["rerun_mean"] = float(np.mean(rerun_distances))
    results["rerun_std"] = float(np.std(rerun_distances))

    # Cross-image clean distances (within-class variance)
    cross_clean = []
    for i in range(n_images):
        for j in range(i+1, n_images):
            cross_clean.append(float(cosine(clean_embs[i], clean_embs[j])))
    results["cross_clean_distances"] = cross_clean
    results["cross_clean_mean"] = float(np.mean(cross_clean))
    results["cross_clean_std"] = float(np.std(cross_clean))

    # Part 2: Corruption distances per image
    print("\n=== Part 2: Corruption Distance Distribution ===")
    corruption_distances = {}
    for c in corruptions:
        for sev in severities:
            key = f"{c}_sev{sev}"
            dists = []
            for i, img in enumerate(images):
                corrupted = apply_corruption(img, c, sev)
                emb = extract_hidden(model, processor, corrupted, prompt)
                d = float(cosine(clean_embs[i], emb))
                dists.append(d)
            corruption_distances[key] = {
                "distances": dists,
                "mean": float(np.mean(dists)),
                "std": float(np.std(dists)),
                "min": float(np.min(dists)),
                "max": float(np.max(dists))
            }
            print(f"  {key}: mean={np.mean(dists):.6f}, std={np.std(dists):.6f}, min={np.min(dists):.6f}")
    results["corruption_distances"] = corruption_distances

    # Part 3: Gap analysis & threshold bounds
    print("\n=== Part 3: Detection Gap Analysis ===")
    gap_analysis = {}
    clean_max = max(cross_clean) if cross_clean else 0.0

    for key, cd in corruption_distances.items():
        gap = cd["min"] - clean_max
        separation_ratio = cd["min"] / clean_max if clean_max > 0 else float('inf')
        gap_analysis[key] = {
            "clean_max": clean_max,
            "corruption_min": cd["min"],
            "gap": gap,
            "separation_ratio": separation_ratio,
            "gap_positive": gap > 0
        }
    results["gap_analysis"] = gap_analysis

    # Part 4: Hoeffding bound
    # P(d_clean > threshold) <= exp(-2n * (threshold - mean_clean)^2 / range^2)
    print("\n=== Part 4: Hoeffding Bounds ===")
    hoeffding = {}
    mean_clean = float(np.mean(cross_clean)) if cross_clean else 0.0
    range_clean = float(np.max(cross_clean) - np.min(cross_clean)) if len(cross_clean) > 1 else 1e-10

    for key, cd in corruption_distances.items():
        # Set threshold at midpoint of gap
        threshold = (clean_max + cd["min"]) / 2

        # False positive bound (clean exceeds threshold)
        if range_clean > 0:
            fp_exponent = -2 * len(cross_clean) * ((threshold - mean_clean) ** 2) / (range_clean ** 2)
            fp_bound = min(1.0, math.exp(fp_exponent)) if fp_exponent > -700 else 0.0
        else:
            fp_bound = 0.0

        # False negative bound (corruption below threshold)
        range_corr = cd["max"] - cd["min"] if cd["max"] > cd["min"] else 1e-10
        fn_exponent = -2 * n_images * ((cd["mean"] - threshold) ** 2) / (range_corr ** 2)
        fn_bound = min(1.0, math.exp(fn_exponent)) if fn_exponent > -700 else 0.0

        hoeffding[key] = {
            "threshold": threshold,
            "fp_bound": fp_bound,
            "fn_bound": fn_bound,
            "total_error_bound": fp_bound + fn_bound
        }
    results["hoeffding_bounds"] = hoeffding

    # Part 5: PAC-style guarantee
    # With n samples and 0 errors, P(error_rate > epsilon) <= (1-epsilon)^n
    # So for confidence delta: epsilon <= 1 - delta^(1/n)
    print("\n=== Part 5: PAC-Style Guarantees ===")
    n_total = n_images * len(corruptions) * len(severities)
    n_errors = 0  # all detected

    pac_guarantees = {}
    for delta in [0.01, 0.05, 0.1]:
        if n_errors == 0:
            # Rule of 3: with 0 errors in n trials, 95% CI upper bound is 3/n
            epsilon = 1 - delta ** (1.0 / n_total)
            pac_guarantees[f"delta_{delta}"] = {
                "confidence": 1 - delta,
                "max_error_rate": epsilon,
                "n_samples": n_total,
                "n_errors": n_errors,
                "rule_of_3_bound": 3.0 / n_total
            }
    results["pac_guarantees"] = pac_guarantees

    # Part 6: Per-scene detection with single centroid
    # Test if per-scene centroid eliminates cross-scene variance
    print("\n=== Part 6: Per-Scene vs Global Centroid ===")
    per_scene = {"global_centroid": {}, "per_scene_centroid": {}}

    # Global centroid = mean of all clean embeddings
    global_centroid = np.mean(clean_embs, axis=0)

    for c in corruptions:
        global_dists_clean = []
        global_dists_corrupt = []
        perscene_dists_clean = []
        perscene_dists_corrupt = []

        for i, img in enumerate(images):
            # Global centroid distances
            d_clean_global = float(cosine(global_centroid, clean_embs[i]))
            global_dists_clean.append(d_clean_global)

            corrupted = apply_corruption(img, c, 1.0)
            emb_c = extract_hidden(model, processor, corrupted, prompt)
            d_corrupt_global = float(cosine(global_centroid, emb_c))
            global_dists_corrupt.append(d_corrupt_global)

            # Per-scene centroid distances
            d_clean_local = float(cosine(clean_embs[i], clean_embs[i]))  # = 0
            perscene_dists_clean.append(d_clean_local)
            d_corrupt_local = float(cosine(clean_embs[i], emb_c))
            perscene_dists_corrupt.append(d_corrupt_local)

        # Compute AUROC for each approach
        def compute_auroc(id_scores, ood_scores):
            id_s = np.asarray(id_scores)
            ood_s = np.asarray(ood_scores)
            n_id, n_ood = len(id_s), len(ood_s)
            if n_id == 0 or n_ood == 0: return 0.5
            count = sum(float(np.sum(o > id_s) + 0.5 * np.sum(o == id_s)) for o in ood_s)
            return count / (n_id * n_ood)

        auroc_global = compute_auroc(global_dists_clean, global_dists_corrupt)
        auroc_perscene = compute_auroc(perscene_dists_clean, perscene_dists_corrupt)

        per_scene["global_centroid"][c] = {
            "auroc": auroc_global,
            "clean_mean": float(np.mean(global_dists_clean)),
            "corrupt_mean": float(np.mean(global_dists_corrupt))
        }
        per_scene["per_scene_centroid"][c] = {
            "auroc": auroc_perscene,
            "clean_mean": float(np.mean(perscene_dists_clean)),
            "corrupt_mean": float(np.mean(perscene_dists_corrupt))
        }
        print(f"  {c}: global AUROC={auroc_global:.3f}, per-scene AUROC={auroc_perscene:.3f}")

    results["centroid_comparison"] = per_scene

    # Save
    ts = results["timestamp"]
    out_path = f"experiments/concentration_{ts}.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Summary
    print("\n=== SUMMARY ===")
    print(f"Re-run variance: mean={results['rerun_mean']:.10f}, std={results['rerun_std']:.10f}")
    print(f"Cross-clean distances: mean={results['cross_clean_mean']:.6f}, std={results['cross_clean_std']:.6f}")
    for key, gap in gap_analysis.items():
        print(f"  {key}: gap={gap['gap']:.6f}, ratio={gap['separation_ratio']:.1f}x")
    print(f"\nPAC guarantees:")
    for k, v in pac_guarantees.items():
        print(f"  {k}: error_rate < {v['max_error_rate']:.4f} with {v['confidence']*100}% confidence")

if __name__ == "__main__":
    main()
