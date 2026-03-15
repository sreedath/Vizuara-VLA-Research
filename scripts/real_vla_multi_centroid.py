#!/usr/bin/env python3
"""Experiment 294: Multi-Centroid Ensemble Detection
Tests if multiple centroids from different clean images improve detection:
1. K centroids from K different scenes
2. Min-distance rule (distance to nearest centroid)
3. Mean-distance rule (average distance to all centroids)
4. Max-distance rule (distance to farthest centroid)
5. Comparison: single-centroid vs multi-centroid AUROC
"""

import torch
import numpy as np
import json
from datetime import datetime
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor
from scipy.spatial.distance import cosine

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

def compute_auroc(id_scores, ood_scores):
    id_s = np.asarray(id_scores)
    ood_s = np.asarray(ood_scores)
    n_id, n_ood = len(id_s), len(ood_s)
    if n_id == 0 or n_ood == 0: return 0.5
    count = sum(float(np.sum(o > id_s) + 0.5 * np.sum(o == id_s)) for o in ood_s)
    return count / (n_id * n_ood)

def main():
    print("Loading OpenVLA-7B...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\nOut:"

    # Generate diverse scenes
    n_scenes = 10
    scenes = []
    clean_embs = []
    print("Generating scenes and clean embeddings...")
    for i in range(n_scenes):
        rng = np.random.RandomState(i * 7 + 42)
        pixels = rng.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(pixels)
        scenes.append(img)
        emb = extract_hidden(model, processor, img, prompt)
        clean_embs.append(emb)
        print(f"  Scene {i}: embedding computed")

    results = {
        "experiment": "multi_centroid",
        "experiment_number": 294,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    corruptions = ['fog', 'night', 'blur', 'noise']

    # Part 1: Single centroid per scene (baseline)
    print("\n=== Part 1: Single Centroid Baseline ===")
    single_results = {}
    for test_scene in range(n_scenes):
        for c in corruptions:
            corrupted = apply_corruption(scenes[test_scene], c, 1.0)
            emb = extract_hidden(model, processor, corrupted, prompt)

            # Per-scene centroid (optimal)
            d_perscene = float(cosine(clean_embs[test_scene], emb))

            key = f"scene{test_scene}_{c}"
            single_results[key] = {
                "per_scene_distance": d_perscene
            }

    # Part 2: Multi-centroid rules
    print("\n=== Part 2: Multi-Centroid Rules ===")
    multi_rules = {"min_dist": {}, "mean_dist": {}, "max_dist": {}, "global_centroid": {}}

    # Global centroid = mean of all clean embeddings
    global_centroid = np.mean(clean_embs, axis=0)

    for k_centroids in [1, 2, 3, 5, 10]:
        print(f"\n  K={k_centroids} centroids:")
        centroids = clean_embs[:k_centroids]

        # Collect all test cases
        id_min, id_mean, id_max, id_global = [], [], [], []
        ood_min, ood_mean, ood_max, ood_global = {c: [] for c in corruptions}, {c: [] for c in corruptions}, {c: [] for c in corruptions}, {c: [] for c in corruptions}

        for test_scene in range(n_scenes):
            # Clean test
            dists_to_centroids = [float(cosine(cent, clean_embs[test_scene])) for cent in centroids]
            id_min.append(min(dists_to_centroids))
            id_mean.append(np.mean(dists_to_centroids))
            id_max.append(max(dists_to_centroids))
            id_global.append(float(cosine(global_centroid, clean_embs[test_scene])))

            # Corrupted tests
            for c in corruptions:
                corrupted = apply_corruption(scenes[test_scene], c, 1.0)
                emb = extract_hidden(model, processor, corrupted, prompt)
                dists_to_centroids = [float(cosine(cent, emb)) for cent in centroids]
                ood_min[c].append(min(dists_to_centroids))
                ood_mean[c].append(np.mean(dists_to_centroids))
                ood_max[c].append(max(dists_to_centroids))
                ood_global[c].append(float(cosine(global_centroid, emb)))

        # Compute AUROCs
        for rule_name, id_dists, ood_dists_dict in [
            ("min_dist", id_min, ood_min),
            ("mean_dist", id_mean, ood_mean),
            ("max_dist", id_max, ood_max),
            ("global_centroid", id_global, ood_global)
        ]:
            if k_centroids not in multi_rules[rule_name]:
                multi_rules[rule_name][k_centroids] = {}
            for c in corruptions:
                auroc = compute_auroc(id_dists, ood_dists_dict[c])
                multi_rules[rule_name][k_centroids][c] = auroc
            avg_auroc = np.mean([multi_rules[rule_name][k_centroids][c] for c in corruptions])
            print(f"    {rule_name}: " + ", ".join([f"{c}={multi_rules[rule_name][k_centroids][c]:.3f}" for c in corruptions]) +
                  f" (avg={avg_auroc:.3f})")

    results["multi_centroid_rules"] = multi_rules

    # Part 3: Leave-one-out cross-validation
    print("\n=== Part 3: Leave-One-Out Cross-Validation ===")
    loo_results = {}
    for c in corruptions:
        loo_aurocs = []
        for test_scene in range(n_scenes):
            # Use all OTHER scenes as centroids
            other_centroids = [clean_embs[i] for i in range(n_scenes) if i != test_scene]

            # Clean distance (to nearest centroid)
            clean_dists = [float(cosine(cent, clean_embs[test_scene])) for cent in other_centroids]
            min_clean_d = min(clean_dists)

            # Corrupted distance
            corrupted = apply_corruption(scenes[test_scene], c, 1.0)
            emb = extract_hidden(model, processor, corrupted, prompt)
            corrupt_dists = [float(cosine(cent, emb)) for cent in other_centroids]
            min_corrupt_d = min(corrupt_dists)

            # Binary AUROC
            loo_aurocs.append(1.0 if min_corrupt_d > min_clean_d else 0.0)

        loo_results[c] = {
            "loo_accuracy": np.mean(loo_aurocs),
            "per_scene": loo_aurocs
        }
        print(f"  {c}: LOO accuracy = {np.mean(loo_aurocs):.3f}")
    results["loo_crossval"] = loo_results

    # Save
    ts = results["timestamp"]
    out_path = f"experiments/multicentroid_{ts}.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
