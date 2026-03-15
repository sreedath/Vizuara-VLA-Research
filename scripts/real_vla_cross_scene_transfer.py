#!/usr/bin/env python3
"""Experiment 408: Cross-Scene Transfer Analysis

Tests whether a corruption detector calibrated on one scene generalizes to
entirely different scenes. This is critical for real-world deployment where
the detector must work on never-before-seen environments.

Tests:
1. Train centroid on scene A, test on scenes B-E (leave-one-out)
2. N-scene calibration: how many scenes needed for robust detection?
3. Scene-difficulty ranking: which scenes are easiest/hardest to transfer to?
4. Cross-scene threshold stability: does optimal threshold vary across scenes?
5. Scene-diversity impact: how different must training scenes be?
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

    # Generate 8 diverse scenes
    seeds = [42, 123, 456, 789, 999, 1234, 5678, 9999]
    scenes = []
    for seed in seeds:
        scenes.append(Image.fromarray(
            np.random.RandomState(seed).randint(0, 255, (224, 224, 3), dtype=np.uint8)))

    # Extract all embeddings
    print("Extracting embeddings for 8 scenes...")
    clean_embeddings = []
    corrupt_embeddings = {c: [] for c in corruptions}

    for si, scene in enumerate(scenes):
        print(f"  Scene {si+1}/8")
        clean_emb = extract_hidden(model, processor, scene, prompt)
        clean_embeddings.append(clean_emb)

        for c in corruptions:
            corrupted = apply_corruption(scene, c)
            corrupt_emb = extract_hidden(model, processor, corrupted, prompt)
            corrupt_embeddings[c].append(corrupt_emb)

    results = {}

    # === Test 1: Leave-one-out cross-scene transfer ===
    print("\n=== Leave-One-Out Transfer ===")
    loo_results = {}
    for test_idx in range(len(scenes)):
        train_indices = [i for i in range(len(scenes)) if i != test_idx]
        centroid = np.mean([clean_embeddings[i] for i in train_indices], axis=0)

        # ID scores: clean test scene
        id_score = cosine_dist(clean_embeddings[test_idx], centroid)

        # OOD scores: corrupted test scene
        for c in corruptions:
            ood_score = cosine_dist(corrupt_embeddings[c][test_idx], centroid)
            key = f"scene{test_idx}_{c}"
            auroc = compute_auroc([id_score], [ood_score])
            loo_results[key] = {
                "id_score": float(id_score),
                "ood_score": float(ood_score),
                "auroc": float(auroc),
                "margin": float(ood_score - id_score)
            }

        # Also test all corruptions together
        ood_scores = [cosine_dist(corrupt_embeddings[c][test_idx], centroid) for c in corruptions]
        auroc_all = compute_auroc([id_score], ood_scores)
        loo_results[f"scene{test_idx}_all"] = {
            "id_score": float(id_score),
            "ood_scores": [float(s) for s in ood_scores],
            "auroc": float(auroc_all)
        }
        print(f"  Scene {test_idx} (test): id={id_score:.6f}, auroc_all={auroc_all:.4f}")

    results["leave_one_out"] = loo_results

    # === Test 2: N-scene calibration ===
    print("\n=== N-Scene Calibration ===")
    n_scene_results = {}
    for n_train in [1, 2, 3, 4, 5, 6, 7]:
        aurocs_per_trial = []
        margins_per_trial = []

        # Try multiple combinations
        rng = np.random.RandomState(42)
        from math import factorial
        n_trials = min(20, max(5, int(factorial(8) / (factorial(n_train) * factorial(8 - n_train)))))

        for trial in range(n_trials):
            train_indices = sorted(rng.choice(8, n_train, replace=False).tolist())
            test_indices = [i for i in range(8) if i not in train_indices]

            centroid = np.mean([clean_embeddings[i] for i in train_indices], axis=0)

            id_scores = [cosine_dist(clean_embeddings[i], centroid) for i in test_indices]
            ood_scores = []
            for c in corruptions:
                for ti in test_indices:
                    ood_scores.append(cosine_dist(corrupt_embeddings[c][ti], centroid))

            auroc = compute_auroc(id_scores, ood_scores)
            margin = np.min(ood_scores) - np.max(id_scores)
            aurocs_per_trial.append(auroc)
            margins_per_trial.append(margin)

        n_scene_results[str(n_train)] = {
            "mean_auroc": float(np.mean(aurocs_per_trial)),
            "min_auroc": float(np.min(aurocs_per_trial)),
            "max_auroc": float(np.max(aurocs_per_trial)),
            "std_auroc": float(np.std(aurocs_per_trial)),
            "mean_margin": float(np.mean(margins_per_trial)),
            "n_trials": n_trials
        }
        print(f"  {n_train} scenes: AUROC={np.mean(aurocs_per_trial):.4f} ± {np.std(aurocs_per_trial):.4f}")

    results["n_scene_calibration"] = n_scene_results

    # === Test 3: Scene difficulty ranking ===
    print("\n=== Scene Difficulty ===")
    scene_difficulty = {}
    global_centroid = np.mean(clean_embeddings, axis=0)

    for si in range(len(scenes)):
        id_dist = cosine_dist(clean_embeddings[si], global_centroid)
        ood_dists = {}
        for c in corruptions:
            ood_dists[c] = cosine_dist(corrupt_embeddings[c][si], global_centroid)

        min_margin = min(ood_dists[c] - id_dist for c in corruptions)
        scene_difficulty[f"scene{si}"] = {
            "id_dist": float(id_dist),
            "ood_dists": {c: float(v) for c, v in ood_dists.items()},
            "min_margin": float(min_margin),
            "difficulty_rank": 0  # filled below
        }

    # Rank by margin (smaller margin = harder)
    ranked = sorted(scene_difficulty.keys(), key=lambda s: scene_difficulty[s]["min_margin"])
    for rank, scene in enumerate(ranked):
        scene_difficulty[scene]["difficulty_rank"] = rank + 1
        print(f"  Rank {rank+1}: {scene}, min_margin={scene_difficulty[scene]['min_margin']:.6f}")

    results["scene_difficulty"] = scene_difficulty

    # === Test 4: Threshold stability ===
    print("\n=== Threshold Stability ===")
    threshold_results = {}
    for si in range(len(scenes)):
        train_indices = [i for i in range(len(scenes)) if i != si]
        centroid = np.mean([clean_embeddings[i] for i in train_indices], axis=0)

        # Find optimal threshold (midpoint between max ID and min OOD)
        id_scores = [cosine_dist(clean_embeddings[i], centroid) for i in train_indices]
        ood_scores = []
        for c in corruptions:
            for ti in train_indices:
                ood_scores.append(cosine_dist(corrupt_embeddings[c][ti], centroid))

        max_id = max(id_scores)
        min_ood = min(ood_scores)
        optimal_thresh = (max_id + min_ood) / 2

        # Test on held-out scene
        test_id = cosine_dist(clean_embeddings[si], centroid)
        test_ood = [cosine_dist(corrupt_embeddings[c][si], centroid) for c in corruptions]

        fp = 1 if test_id > optimal_thresh else 0
        fn = sum(1 for s in test_ood if s <= optimal_thresh)

        threshold_results[f"scene{si}"] = {
            "optimal_thresh": float(optimal_thresh),
            "test_id_score": float(test_id),
            "test_ood_scores": [float(s) for s in test_ood],
            "false_positive": fp,
            "false_negatives": fn,
            "thresh_margin_id": float(optimal_thresh - test_id),
            "thresh_margin_ood": float(min(test_ood) - optimal_thresh)
        }
        print(f"  Scene {si}: thresh={optimal_thresh:.6f}, FP={fp}, FN={fn}")

    all_thresholds = [threshold_results[f"scene{si}"]["optimal_thresh"] for si in range(len(scenes))]
    threshold_results["summary"] = {
        "mean_threshold": float(np.mean(all_thresholds)),
        "std_threshold": float(np.std(all_thresholds)),
        "cv_threshold": float(np.std(all_thresholds) / np.mean(all_thresholds)) if np.mean(all_thresholds) > 0 else 0,
        "total_fp": sum(threshold_results[f"scene{si}"]["false_positive"] for si in range(len(scenes))),
        "total_fn": sum(threshold_results[f"scene{si}"]["false_negatives"] for si in range(len(scenes)))
    }
    print(f"  Threshold: {np.mean(all_thresholds):.6f} ± {np.std(all_thresholds):.6f}")
    print(f"  Total FP={threshold_results['summary']['total_fp']}, FN={threshold_results['summary']['total_fn']}")

    results["threshold_stability"] = threshold_results

    # === Test 5: Scene diversity impact ===
    print("\n=== Scene Diversity Impact ===")
    diversity_results = {}

    # Measure inter-scene distances
    inter_scene_dists = []
    for i in range(len(scenes)):
        for j in range(i+1, len(scenes)):
            d = cosine_dist(clean_embeddings[i], clean_embeddings[j])
            inter_scene_dists.append((i, j, d))

    inter_scene_dists.sort(key=lambda x: x[2])
    diversity_results["inter_scene_distances"] = {
        f"scene{i}_scene{j}": float(d) for i, j, d in inter_scene_dists
    }

    # Most similar pair vs most different pair as training sets
    most_similar = inter_scene_dists[0]
    most_different = inter_scene_dists[-1]

    for label, pair in [("most_similar", most_similar), ("most_different", most_different)]:
        train_indices = [pair[0], pair[1]]
        centroid = np.mean([clean_embeddings[i] for i in train_indices], axis=0)
        test_indices = [i for i in range(len(scenes)) if i not in train_indices]

        id_scores = [cosine_dist(clean_embeddings[i], centroid) for i in test_indices]
        ood_scores = []
        for c in corruptions:
            for ti in test_indices:
                ood_scores.append(cosine_dist(corrupt_embeddings[c][ti], centroid))

        auroc = compute_auroc(id_scores, ood_scores)
        margin = float(np.min(ood_scores) - np.max(id_scores))

        diversity_results[label] = {
            "scenes": train_indices,
            "inter_scene_dist": float(pair[2]),
            "auroc": float(auroc),
            "margin": margin,
            "max_id_score": float(np.max(id_scores)),
            "min_ood_score": float(np.min(ood_scores))
        }
        print(f"  {label} pair ({pair[0]},{pair[1]}): dist={pair[2]:.6f}, auroc={auroc:.4f}, margin={margin:.6f}")

    results["scene_diversity"] = diversity_results

    # Summary statistics
    results["n_scenes"] = len(scenes)
    results["n_corruptions"] = len(corruptions)
    results["seeds"] = seeds

    # Save
    out_path = "/workspace/Vizuara-VLA-Research/experiments/cross_scene_transfer_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
