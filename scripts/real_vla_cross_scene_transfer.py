#!/usr/bin/env python3
"""Experiment 438: Cross-Scene Transfer Analysis

Tests whether a centroid calibrated on one set of scenes generalizes
to completely different scenes. This is critical for deployment —
we can't recalibrate for every new environment.

Tests:
1. Calibrate on scenes A, test on disjoint scenes B
2. Leave-one-out cross-validation AUROC
3. Centroid drift across scene sets
4. Scene diversity vs detection quality
5. Mixed-scene calibration robustness
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

    # Create three disjoint scene sets
    seeds_A = [42, 123, 456, 789, 999]
    seeds_B = [1111, 2222, 3333, 4444, 5555]
    seeds_C = [6666, 7777, 8888, 9999, 10000]

    scenes_A = [Image.fromarray(np.random.RandomState(s).randint(0, 255, (224, 224, 3), dtype=np.uint8)) for s in seeds_A]
    scenes_B = [Image.fromarray(np.random.RandomState(s).randint(0, 255, (224, 224, 3), dtype=np.uint8)) for s in seeds_B]
    scenes_C = [Image.fromarray(np.random.RandomState(s).randint(0, 255, (224, 224, 3), dtype=np.uint8)) for s in seeds_C]

    print("Extracting embeddings for 3 scene sets (5 each)...")
    embs_A = [extract_hidden(model, processor, s, prompt) for s in scenes_A]
    embs_B = [extract_hidden(model, processor, s, prompt) for s in scenes_B]
    embs_C = [extract_hidden(model, processor, s, prompt) for s in scenes_C]

    centroid_A = np.mean(embs_A, axis=0)
    centroid_B = np.mean(embs_B, axis=0)
    centroid_C = np.mean(embs_C, axis=0)

    results = {"n_scenes_per_set": 5}

    # === Test 1: Cross-set transfer ===
    print("\n=== Cross-Set Transfer ===")
    transfer_results = {}
    for calib_name, calib_embs, calib_centroid in [("A", embs_A, centroid_A),
                                                     ("B", embs_B, centroid_B),
                                                     ("C", embs_C, centroid_C)]:
        for test_name, test_embs, test_scenes in [("A", embs_A, scenes_A),
                                                    ("B", embs_B, scenes_B),
                                                    ("C", embs_C, scenes_C)]:
            clean_dists = [cosine_dist(e, calib_centroid) for e in test_embs]
            per_corr = {}
            for c in corruptions:
                ood_dists = []
                for s in test_scenes:
                    emb = extract_hidden(model, processor, apply_corruption(s, c), prompt)
                    ood_dists.append(float(cosine_dist(emb, calib_centroid)))
                auroc = float(compute_auroc(clean_dists, ood_dists))
                per_corr[c] = auroc

            key = f"calib_{calib_name}_test_{test_name}"
            transfer_results[key] = {
                "auroc_per_corruption": per_corr,
                "mean_auroc": float(np.mean(list(per_corr.values()))),
                "mean_clean_dist": float(np.mean(clean_dists)),
            }
            print(f"  Calib {calib_name} -> Test {test_name}: mean={np.mean(list(per_corr.values())):.4f}")
    results["cross_set_transfer"] = transfer_results

    # === Test 2: Leave-one-out cross-validation ===
    print("\n=== Leave-One-Out Cross-Validation ===")
    all_seeds = seeds_A + seeds_B + seeds_C
    all_scenes = scenes_A + scenes_B + scenes_C
    all_embs = embs_A + embs_B + embs_C
    n_total = len(all_embs)

    loo_results = []
    for i in range(n_total):
        calib_embs = [all_embs[j] for j in range(n_total) if j != i]
        loo_centroid = np.mean(calib_embs, axis=0)
        clean_dist_i = cosine_dist(all_embs[i], loo_centroid)
        ood_dists = []
        for c in corruptions:
            emb = extract_hidden(model, processor, apply_corruption(all_scenes[i], c), prompt)
            ood_dists.append(float(cosine_dist(emb, loo_centroid)))
        auroc = float(compute_auroc([clean_dist_i], ood_dists))
        loo_results.append(auroc)

    results["loo_cv"] = {
        "per_scene_auroc": loo_results,
        "mean_auroc": float(np.mean(loo_results)),
        "min_auroc": float(np.min(loo_results)),
        "pct_perfect": float(np.mean(np.array(loo_results) >= 1.0) * 100),
    }
    print(f"  LOO mean AUROC: {np.mean(loo_results):.4f}, min: {np.min(loo_results):.4f}, "
          f"perfect: {np.mean(np.array(loo_results) >= 1.0)*100:.0f}%")

    # === Test 3: Centroid drift ===
    print("\n=== Centroid Drift Between Scene Sets ===")
    drift_results = {
        "A_B_cosine_dist": float(cosine_dist(centroid_A, centroid_B)),
        "A_C_cosine_dist": float(cosine_dist(centroid_A, centroid_C)),
        "B_C_cosine_dist": float(cosine_dist(centroid_B, centroid_C)),
        "A_B_cosine_sim": float(1 - cosine_dist(centroid_A, centroid_B)),
        "A_centroid_norm": float(np.linalg.norm(centroid_A)),
        "B_centroid_norm": float(np.linalg.norm(centroid_B)),
        "C_centroid_norm": float(np.linalg.norm(centroid_C)),
    }
    print(f"  A<->B: dist={drift_results['A_B_cosine_dist']:.6f}")
    print(f"  A<->C: dist={drift_results['A_C_cosine_dist']:.6f}")
    print(f"  B<->C: dist={drift_results['B_C_cosine_dist']:.6f}")
    results["centroid_drift"] = drift_results

    # === Test 4: Calibration set size vs transfer ===
    print("\n=== Calibration Set Size vs Transfer ===")
    rng = np.random.RandomState(42)
    size_results = {}
    for n_calib in [1, 2, 3, 5, 10]:
        if n_calib >= n_total:
            continue
        trial_aurocs = []
        for trial in range(20):
            idx = rng.choice(n_total, n_calib, replace=False)
            sub_embs = [all_embs[i] for i in idx]
            sub_centroid = np.mean(sub_embs, axis=0)
            test_idx = [i for i in range(n_total) if i not in idx]
            if len(test_idx) == 0:
                continue
            test_clean_dists = [cosine_dist(all_embs[i], sub_centroid) for i in test_idx]
            test_ood_dists = []
            for i in test_idx[:3]:
                for c in corruptions:
                    emb = extract_hidden(model, processor, apply_corruption(all_scenes[i], c), prompt)
                    test_ood_dists.append(float(cosine_dist(emb, sub_centroid)))
            auroc = float(compute_auroc(test_clean_dists[:3], test_ood_dists))
            trial_aurocs.append(auroc)

        if len(trial_aurocs) == 0:
            continue
        size_results[str(n_calib)] = {
            "mean_auroc": float(np.mean(trial_aurocs)),
            "std_auroc": float(np.std(trial_aurocs)),
            "min_auroc": float(np.min(trial_aurocs)),
        }
        print(f"  n={n_calib}: mean={np.mean(trial_aurocs):.4f} +/- {np.std(trial_aurocs):.4f}")
    results["calib_size_transfer"] = size_results

    # === Test 5: Universal centroid (all 15 scenes) ===
    print("\n=== Universal Centroid ===")
    universal_centroid = np.mean(all_embs, axis=0)
    universal_clean_dists = [cosine_dist(e, universal_centroid) for e in all_embs]

    universal_results = {}
    for c in corruptions:
        ood_dists = []
        for s in all_scenes:
            emb = extract_hidden(model, processor, apply_corruption(s, c), prompt)
            ood_dists.append(float(cosine_dist(emb, universal_centroid)))
        auroc = float(compute_auroc(universal_clean_dists, ood_dists))
        universal_results[c] = auroc
        print(f"  {c}: AUROC={auroc:.4f}")
    results["universal_centroid"] = universal_results

    out_path = "/workspace/Vizuara-VLA-Research/experiments/cross_scene_transfer_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
