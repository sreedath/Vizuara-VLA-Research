#!/usr/bin/env python3
"""Experiment 401: Systematic Layer Selection for OOD Detection

Which hidden state layer is optimal for detection?
Tests ALL 33 layers (embedding + 32 transformer layers) systematically.

Tests:
1. AUROC at each layer for each corruption
2. Layer-wise embedding dimensionality (effective rank)
3. Layer-wise cosine distance magnitude
4. Early vs middle vs late layer comparison
5. Multi-layer fusion (concatenation, averaging)
"""

import json, time, os, sys
import numpy as np
import torch
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor

def cosine_dist(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    return 1.0 - np.dot(a, b) / (na * nb)

def compute_auroc(id_scores, ood_scores):
    id_s = np.asarray(id_scores)
    ood_s = np.asarray(ood_scores)
    n_id, n_ood = len(id_s), len(ood_s)
    if n_id == 0 or n_ood == 0:
        return 0.5
    count = sum(float(np.sum(o > id_s) + 0.5 * np.sum(o == id_s)) for o in ood_s)
    return count / (n_id * n_ood)

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

def main():
    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    corruptions = ['fog', 'night', 'noise', 'blur']

    scenes = []
    for seed in [42, 123, 456, 789, 999]:
        scenes.append(Image.fromarray(
            np.random.RandomState(seed).randint(0, 255, (224, 224, 3), dtype=np.uint8)))

    # Extract ALL hidden states for clean and corrupt
    print("Extracting all layer embeddings...")

    def get_all_layers(image):
        inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            fwd = model(**inputs, output_hidden_states=True)
        # hidden_states[i] is (batch, seq, dim), get last token
        return [fwd.hidden_states[i][0, -1, :].float().cpu().numpy()
                for i in range(len(fwd.hidden_states))]

    # Get clean embeddings per layer
    clean_per_layer = {}  # layer -> list of embeddings
    for si, scene in enumerate(scenes):
        print(f"  Clean scene {si+1}/5")
        all_layers = get_all_layers(scene)
        n_layers = len(all_layers)
        for li in range(n_layers):
            if li not in clean_per_layer:
                clean_per_layer[li] = []
            clean_per_layer[li].append(all_layers[li])

    print(f"  Total layers: {n_layers}")

    # Get corrupt embeddings per layer
    corrupt_per_layer = {}  # corruption -> layer -> list of embeddings
    for c in corruptions:
        corrupt_per_layer[c] = {}
        for si, scene in enumerate(scenes):
            print(f"  {c} scene {si+1}/5")
            corrupted = apply_corruption(scene, c, 1.0)
            all_layers = get_all_layers(corrupted)
            for li in range(n_layers):
                if li not in corrupt_per_layer[c]:
                    corrupt_per_layer[c][li] = []
                corrupt_per_layer[c][li].append(all_layers[li])

    results = {"n_layers": n_layers}

    # === Test 1: AUROC at each layer ===
    print("\n=== Per-Layer AUROC ===")
    layer_aurocs = {}
    for li in range(n_layers):
        # Compute centroid from clean
        centroid = np.mean(clean_per_layer[li], axis=0)
        clean_dists = [cosine_dist(e, centroid) for e in clean_per_layer[li]]

        layer_result = {}
        for c in corruptions:
            corrupt_dists = [cosine_dist(e, centroid) for e in corrupt_per_layer[c][li]]
            auroc = compute_auroc(clean_dists, corrupt_dists)
            layer_result[c] = {
                "auroc": float(auroc),
                "mean_dist": float(np.mean(corrupt_dists)),
                "clean_mean": float(np.mean(clean_dists))
            }

        mean_auroc = np.mean([layer_result[c]["auroc"] for c in corruptions])
        layer_result["mean_auroc"] = float(mean_auroc)
        layer_aurocs[str(li)] = layer_result

        if li % 4 == 0 or li == n_layers - 1:
            auroc_str = ", ".join(f"{c}={layer_result[c]['auroc']:.3f}" for c in corruptions)
            print(f"  Layer {li}: mean={mean_auroc:.3f} ({auroc_str})")

    results["layer_aurocs"] = layer_aurocs

    # === Test 2: Embedding norms and distances ===
    print("\n=== Layer-wise Statistics ===")
    layer_stats = {}
    for li in range(n_layers):
        norms = [np.linalg.norm(e) for e in clean_per_layer[li]]
        layer_stats[str(li)] = {
            "mean_norm": float(np.mean(norms)),
            "dim": int(clean_per_layer[li][0].shape[0])
        }

    results["layer_stats"] = layer_stats

    # === Test 3: Best layer per corruption ===
    print("\n=== Best Layer Per Corruption ===")
    best_layers = {}
    for c in corruptions:
        best_li = max(range(n_layers), key=lambda li: layer_aurocs[str(li)][c]["auroc"])
        best_auroc = layer_aurocs[str(best_li)][c]["auroc"]
        # Among layers with best AUROC, pick the one with largest distance
        best_candidates = [li for li in range(n_layers)
                          if layer_aurocs[str(li)][c]["auroc"] == best_auroc]
        best_by_dist = max(best_candidates,
                          key=lambda li: layer_aurocs[str(li)][c]["mean_dist"])
        best_layers[c] = {
            "best_layer": best_by_dist,
            "auroc": float(best_auroc),
            "mean_dist": float(layer_aurocs[str(best_by_dist)][c]["mean_dist"]),
            "n_perfect_layers": len(best_candidates)
        }
        print(f"  {c}: layer {best_by_dist} (auroc={best_auroc:.3f}, "
              f"{len(best_candidates)} layers tie)")

    results["best_layers"] = best_layers

    # === Test 4: Multi-layer fusion ===
    print("\n=== Multi-Layer Fusion ===")
    fusion_results = {}

    # Try concatenating embeddings from multiple layers
    layer_combos = [
        ("early", [0, 1, 2]),
        ("mid", [15, 16, 17]),
        ("late", [30, 31, 32]) if n_layers > 32 else ("late", [n_layers-3, n_layers-2, n_layers-1]),
        ("spread", [0, 8, 16, 24, n_layers-1]),
        ("all_even", list(range(0, n_layers, 2))),
    ]

    for combo_name, layer_indices in layer_combos:
        # Concatenate embeddings from selected layers
        clean_concat = []
        for si in range(len(scenes)):
            concat = np.concatenate([clean_per_layer[li][si] for li in layer_indices])
            clean_concat.append(concat)

        centroid = np.mean(clean_concat, axis=0)
        clean_dists = [cosine_dist(e, centroid) for e in clean_concat]

        combo_result = {}
        for c in corruptions:
            corrupt_concat = []
            for si in range(len(scenes)):
                concat = np.concatenate([corrupt_per_layer[c][li][si] for li in layer_indices])
                corrupt_concat.append(concat)

            corrupt_dists = [cosine_dist(e, centroid) for e in corrupt_concat]
            auroc = compute_auroc(clean_dists, corrupt_dists)
            combo_result[c] = float(auroc)

        combo_result["mean_auroc"] = float(np.mean([combo_result[c] for c in corruptions]))
        combo_result["layers"] = layer_indices
        combo_result["dim"] = len(clean_concat[0])
        fusion_results[combo_name] = combo_result

        auroc_str = ", ".join(f"{c}={combo_result[c]:.3f}" for c in corruptions)
        print(f"  {combo_name} ({layer_indices}): mean={combo_result['mean_auroc']:.3f} ({auroc_str})")

    results["fusion"] = fusion_results

    # === Test 5: Layer-wise distance profile ===
    print("\n=== Distance Profile ===")
    for c in corruptions:
        dists = [layer_aurocs[str(li)][c]["mean_dist"] for li in range(n_layers)]
        best = max(range(n_layers), key=lambda i: dists[i])
        worst = min(range(n_layers), key=lambda i: dists[i])
        print(f"  {c}: best L{best}={dists[best]:.6f}, worst L{worst}={dists[worst]:.8f}, range={dists[best]/max(dists[worst],1e-12):.0f}x")

    # Save
    out_path = "/workspace/Vizuara-VLA-Research/experiments/layer_selection_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
