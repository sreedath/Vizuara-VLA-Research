#!/usr/bin/env python3
"""Experiment 416: Layer Ensemble Detection

Tests whether combining hidden states from multiple transformer layers
improves detection beyond any single layer. Studies layer correlation
structure, optimal layer subsets, and whether diverse layers complement
each other.

Tests:
1. All 32 layers: per-layer AUROC and distance profiles
2. Pairwise layer correlation matrix
3. Ensemble strategies: mean distance, max distance, voting
4. Optimal layer subset selection (greedy forward selection)
5. Layer diversity vs ensemble performance
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

def extract_all_layers(model, processor, image, prompt):
    """Extract hidden states from ALL transformer layers in one pass."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    # hidden_states is tuple of (n_layers+1) tensors, index 0 is embedding layer
    return [fwd.hidden_states[i][0, -1, :].float().cpu().numpy()
            for i in range(len(fwd.hidden_states))]

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

    # Generate scenes
    seeds = [42, 123, 456, 789, 999]
    scenes = [Image.fromarray(np.random.RandomState(s).randint(0, 255, (224, 224, 3), dtype=np.uint8)) for s in seeds]

    # === Extract ALL layer embeddings ===
    print("Extracting all-layer embeddings for clean scenes...")
    clean_all = [extract_all_layers(model, processor, s, prompt) for s in scenes]
    n_layers = len(clean_all[0])
    print(f"  {n_layers} layers found")

    # Per-layer centroids
    centroids = []
    for L in range(n_layers):
        layer_embs = [clean_all[s][L] for s in range(len(scenes))]
        centroids.append(np.mean(layer_embs, axis=0))

    # Per-layer clean distances
    clean_dists_per_layer = []
    for L in range(n_layers):
        dists = [cosine_dist(clean_all[s][L], centroids[L]) for s in range(len(scenes))]
        clean_dists_per_layer.append(dists)

    # Extract corrupt embeddings
    print("Extracting corrupt embeddings...")
    corrupt_all = {}
    for c in corruptions:
        corrupt_all[c] = []
        for s in scenes:
            corrupted = apply_corruption(s, c)
            corrupt_all[c].append(extract_all_layers(model, processor, corrupted, prompt))
        print(f"  {c} done")

    results = {}

    # === Test 1: Per-layer AUROC ===
    print("\n=== Per-Layer AUROC ===")
    per_layer = {}
    for L in range(n_layers):
        id_scores = clean_dists_per_layer[L]
        per_corr_auroc = {}
        for c in corruptions:
            ood_scores = [cosine_dist(corrupt_all[c][s][L], centroids[L]) for s in range(len(scenes))]
            per_corr_auroc[c] = float(compute_auroc(id_scores, ood_scores))

        all_ood = []
        for c in corruptions:
            all_ood.extend([cosine_dist(corrupt_all[c][s][L], centroids[L]) for s in range(len(scenes))])
        overall_auroc = float(compute_auroc(id_scores, all_ood))

        per_layer[str(L)] = {
            "overall_auroc": overall_auroc,
            "per_corruption": per_corr_auroc,
            "mean_clean_dist": float(np.mean(id_scores)),
        }
        if L % 5 == 0 or L == n_layers - 1:
            print(f"  L{L}: overall={overall_auroc:.4f}, fog={per_corr_auroc['fog']:.4f}, "
                  f"noise={per_corr_auroc['noise']:.4f}")
    results["per_layer"] = per_layer

    # === Test 2: Pairwise layer correlation ===
    print("\n=== Layer Correlation (sampled) ===")
    # Use distance vectors to compute correlation
    sample_layers = [0, 1, 3, 7, 15, 23, 31, n_layers - 1]
    sample_layers = [l for l in sample_layers if l < n_layers]

    dist_vectors = {}
    for L in sample_layers:
        vec = []
        for c in corruptions:
            for s in range(len(scenes)):
                vec.append(cosine_dist(corrupt_all[c][s][L], centroids[L]))
        vec.extend(clean_dists_per_layer[L])
        dist_vectors[L] = np.array(vec)

    correlation_matrix = {}
    for L1 in sample_layers:
        for L2 in sample_layers:
            if L1 <= L2:
                r = float(np.corrcoef(dist_vectors[L1], dist_vectors[L2])[0, 1])
                correlation_matrix[f"{L1}_{L2}"] = r
    results["layer_correlations"] = correlation_matrix
    print(f"  L1-L3: {correlation_matrix.get('1_3', 'N/A'):.4f}")
    print(f"  L1-L31: {correlation_matrix.get('1_31', 'N/A'):.4f}")
    print(f"  L3-L15: {correlation_matrix.get('3_15', 'N/A'):.4f}")

    # === Test 3: Ensemble strategies ===
    print("\n=== Ensemble Strategies ===")
    # Test ensemble of layers [1, 3, 7, 15, 31]
    ensemble_layers = [1, 3, 7, 15, 31]
    ensemble_layers = [l for l in ensemble_layers if l < n_layers]

    ensemble_results = {}
    for strategy in ['mean', 'max', 'vote']:
        id_ensemble = []
        for s in range(len(scenes)):
            layer_dists = [clean_dists_per_layer[L][s] for L in ensemble_layers]
            if strategy == 'mean':
                id_ensemble.append(np.mean(layer_dists))
            elif strategy == 'max':
                id_ensemble.append(np.max(layer_dists))
            elif strategy == 'vote':
                # Majority vote: each layer votes OOD if dist > median clean dist for that layer
                thresholds = [np.median(clean_dists_per_layer[L]) for L in ensemble_layers]
                votes = sum(1 for d, t in zip(layer_dists, thresholds) if d > t)
                id_ensemble.append(votes / len(ensemble_layers))

        ood_ensemble = []
        for c in corruptions:
            for s in range(len(scenes)):
                layer_dists = [cosine_dist(corrupt_all[c][s][L], centroids[L]) for L in ensemble_layers]
                if strategy == 'mean':
                    ood_ensemble.append(np.mean(layer_dists))
                elif strategy == 'max':
                    ood_ensemble.append(np.max(layer_dists))
                elif strategy == 'vote':
                    thresholds = [np.median(clean_dists_per_layer[L]) for L in ensemble_layers]
                    votes = sum(1 for d, t in zip(layer_dists, thresholds) if d > t)
                    ood_ensemble.append(votes / len(ensemble_layers))

        auroc = compute_auroc(id_ensemble, ood_ensemble)
        ensemble_results[strategy] = {
            "auroc": float(auroc),
            "layers": ensemble_layers,
        }
        print(f"  {strategy}: AUROC={auroc:.4f}")

    # Also test single best layer (L3)
    single_best = per_layer.get("3", per_layer["1"])
    ensemble_results["single_L3"] = {"auroc": single_best["overall_auroc"]}
    results["ensemble"] = ensemble_results

    # === Test 4: Greedy layer selection ===
    print("\n=== Greedy Layer Selection ===")
    available = list(range(n_layers))
    selected = []
    greedy_path = []

    for step in range(min(8, n_layers)):
        best_auroc = -1
        best_layer = -1
        for L in available:
            test_set = selected + [L]
            id_scores_ens = []
            for s in range(len(scenes)):
                dists = [clean_dists_per_layer[l][s] for l in test_set]
                id_scores_ens.append(np.mean(dists))

            ood_scores_ens = []
            for c in corruptions:
                for s in range(len(scenes)):
                    dists = [cosine_dist(corrupt_all[c][s][l], centroids[l]) for l in test_set]
                    ood_scores_ens.append(np.mean(dists))

            auroc = compute_auroc(id_scores_ens, ood_scores_ens)
            if auroc > best_auroc:
                best_auroc = auroc
                best_layer = L

        selected.append(best_layer)
        available.remove(best_layer)
        greedy_path.append({
            "step": step + 1,
            "added_layer": best_layer,
            "auroc": float(best_auroc),
            "layers_so_far": list(selected),
        })
        print(f"  Step {step+1}: add L{best_layer}, AUROC={best_auroc:.4f}")

    results["greedy_selection"] = greedy_path

    # === Test 5: Layer diversity analysis ===
    print("\n=== Layer Diversity ===")
    diversity = {}
    for L in range(n_layers):
        # How different is this layer's distance pattern from layer 3 (reference)?
        if L == 3:
            continue
        vec_L = []
        vec_3 = []
        for c in corruptions:
            for s in range(len(scenes)):
                vec_L.append(cosine_dist(corrupt_all[c][s][L], centroids[L]))
                vec_3.append(cosine_dist(corrupt_all[c][s][3], centroids[3]))
        if np.std(vec_L) > 1e-10 and np.std(vec_3) > 1e-10:
            corr = float(np.corrcoef(vec_L, vec_3)[0, 1])
        else:
            corr = 0.0
        diversity[str(L)] = {
            "correlation_with_L3": corr,
            "auroc": per_layer[str(L)]["overall_auroc"],
        }
    results["layer_diversity"] = diversity

    # Count how many layers achieve AUROC=1.0
    perfect_layers = [L for L in range(n_layers) if per_layer[str(L)]["overall_auroc"] >= 1.0]
    results["n_perfect_layers"] = len(perfect_layers)
    results["perfect_layers"] = perfect_layers
    print(f"\n  {len(perfect_layers)}/{n_layers} layers achieve AUROC=1.0")

    # Save
    out_path = "/workspace/Vizuara-VLA-Research/experiments/layer_ensemble_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
