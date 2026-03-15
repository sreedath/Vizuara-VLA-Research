#!/usr/bin/env python3
"""Experiment 432: Layer-wise Detection Comparison

Systematically compares OOD detection performance across ALL model layers.
Previous experiments used layer 3 — is this optimal? What happens at deeper
layers closer to the action output?

Tests:
1. AUROC per layer for each corruption
2. Cosine distance magnitude per layer
3. Layer correlation (do different layers agree?)
4. Ensemble detection (combining multiple layers)
5. Layer-specific discriminative dimensions
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

def extract_all_hidden(model, processor, image, prompt):
    """Extract hidden states from ALL layers."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    # Return last token embedding from each layer
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

    seeds = [42, 123, 456, 789, 999, 1234, 5678, 9999]
    scenes = [Image.fromarray(np.random.RandomState(s).randint(0, 255, (224, 224, 3), dtype=np.uint8)) for s in seeds]

    # Extract all layers for first scene to determine n_layers
    print("Probing layer count...")
    probe = extract_all_hidden(model, processor, scenes[0], prompt)
    n_layers = len(probe)
    hidden_dim = probe[0].shape[0]
    print(f"  {n_layers} layers, {hidden_dim} dims each")

    # Extract all layers for all clean scenes
    print(f"Extracting all layers for {len(scenes)} clean scenes...")
    clean_all = []  # [scene][layer] = embedding
    for i, s in enumerate(scenes):
        embs = extract_all_hidden(model, processor, s, prompt)
        clean_all.append(embs)
        print(f"  Scene {i} done")

    # Compute per-layer centroids
    centroids = []
    for layer in range(n_layers):
        layer_embs = [clean_all[i][layer] for i in range(len(scenes))]
        centroids.append(np.mean(layer_embs, axis=0))

    # Per-layer clean distances
    clean_dists_per_layer = []
    for layer in range(n_layers):
        dists = [cosine_dist(clean_all[i][layer], centroids[layer]) for i in range(len(scenes))]
        clean_dists_per_layer.append(dists)

    results = {"n_scenes": len(scenes), "n_layers": n_layers, "hidden_dim": hidden_dim}

    # === Test 1: AUROC per layer ===
    print("\n=== AUROC Per Layer ===")
    # Sample layers to keep runtime reasonable
    sample_layers = sorted(set([0, 1, 2, 3, 4, 5, 8, 12, 16, 20, 24, 28, 31,
                                n_layers // 4, n_layers // 2, 3 * n_layers // 4,
                                n_layers - 2, n_layers - 1]))
    sample_layers = [l for l in sample_layers if l < n_layers]

    layer_auroc = {}
    for layer in sample_layers:
        print(f"  Layer {layer}...")
        per_corr = {}
        for c in corruptions:
            ood_dists = []
            for s in scenes:
                embs = extract_all_hidden(model, processor, apply_corruption(s, c), prompt)
                d = cosine_dist(embs[layer], centroids[layer])
                ood_dists.append(float(d))
            auroc = float(compute_auroc(clean_dists_per_layer[layer], ood_dists))
            per_corr[c] = auroc
        layer_auroc[str(layer)] = per_corr
        mean_auroc = np.mean(list(per_corr.values()))
        print(f"    mean={mean_auroc:.4f} | fog={per_corr['fog']:.4f}, night={per_corr['night']:.4f}, "
              f"noise={per_corr['noise']:.4f}, blur={per_corr['blur']:.4f}")
    results["layer_auroc"] = layer_auroc

    # === Test 2: Distance magnitude per layer ===
    print("\n=== Distance Magnitude Per Layer ===")
    dist_magnitude = {}
    for layer in sample_layers:
        clean_mean = float(np.mean(clean_dists_per_layer[layer]))
        clean_max = float(np.max(clean_dists_per_layer[layer]))
        dist_magnitude[str(layer)] = {
            "clean_mean": clean_mean,
            "clean_max": clean_max,
        }
    results["distance_magnitude"] = dist_magnitude

    # === Test 3: Layer correlation ===
    print("\n=== Layer Correlation ===")
    correlation_results = {}
    fog_dists_per_layer = {}
    for layer in [0, 3, n_layers // 2, n_layers - 1]:
        if layer >= n_layers:
            continue
        dists = []
        for s in scenes:
            embs = extract_all_hidden(model, processor, apply_corruption(s, 'fog'), prompt)
            dists.append(float(cosine_dist(embs[layer], centroids[layer])))
        fog_dists_per_layer[layer] = dists

    layers_used = sorted(fog_dists_per_layer.keys())
    for i, l1 in enumerate(layers_used):
        for l2 in layers_used[i+1:]:
            d1, d2 = fog_dists_per_layer[l1], fog_dists_per_layer[l2]
            corr = float(np.corrcoef(d1, d2)[0, 1])
            correlation_results[f"{l1}_vs_{l2}"] = corr
            print(f"  Layer {l1} vs {l2}: r={corr:.4f}")
    results["layer_correlation"] = correlation_results

    # === Test 4: Ensemble detection ===
    print("\n=== Ensemble Detection ===")
    ensemble_results = {}
    ensemble_layers = [3, min(n_layers // 2, n_layers - 1), min(n_layers - 2, n_layers - 1)]
    ensemble_layers = sorted(set(l for l in ensemble_layers if l < n_layers))

    for c in corruptions:
        ensemble_scores_clean = []
        ensemble_scores_ood = []

        for s_idx in range(len(scenes)):
            layer_dists_clean = [clean_dists_per_layer[l][s_idx] for l in ensemble_layers]
            ensemble_scores_clean.append(float(np.mean(layer_dists_clean)))

        for s in scenes:
            embs = extract_all_hidden(model, processor, apply_corruption(s, c), prompt)
            layer_dists_ood = [float(cosine_dist(embs[l], centroids[l])) for l in ensemble_layers]
            ensemble_scores_ood.append(float(np.mean(layer_dists_ood)))

        ensemble_auroc = float(compute_auroc(ensemble_scores_clean, ensemble_scores_ood))
        ensemble_results[c] = {
            "ensemble_auroc": ensemble_auroc,
            "ensemble_layers": ensemble_layers,
        }
        print(f"  {c}: ensemble AUROC={ensemble_auroc:.4f} (layers {ensemble_layers})")
    results["ensemble_detection"] = ensemble_results

    # === Test 5: Clean embedding norm per layer ===
    print("\n=== Embedding Norms Per Layer ===")
    norm_results = {}
    for layer in sample_layers:
        norms = [float(np.linalg.norm(clean_all[i][layer])) for i in range(len(scenes))]
        norm_results[str(layer)] = {
            "mean_norm": float(np.mean(norms)),
            "std_norm": float(np.std(norms)),
        }
        print(f"  Layer {layer}: norm={np.mean(norms):.4f} ± {np.std(norms):.4f}")
    results["embedding_norms"] = norm_results

    out_path = "/workspace/Vizuara-VLA-Research/experiments/layer_comparison_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
