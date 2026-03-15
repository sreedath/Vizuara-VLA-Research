#!/usr/bin/env python3
"""Experiment 381: Layer-Wise Detection Dynamics

How does corruption detection evolve across transformer layers?
1. Per-layer cosine distance for each corruption
2. Layer-wise AUROC progression
3. Corruption signature per layer (which layers best separate each type)
4. Layer gradient: rate of distance change between consecutive layers
5. Cross-layer correlation of detection signals
"""

import json, time, os, sys
import numpy as np
import torch
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor

def extract_all_layers(model, processor, image, prompt):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return [h[0, -1, :].float().cpu().numpy() for h in fwd.hidden_states]

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

def cosine_dist(a, b):
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return 1.0 - dot / (na * nb)

def compute_auroc(id_scores, ood_scores):
    id_s, ood_s = np.asarray(id_scores), np.asarray(ood_scores)
    n_id, n_ood = len(id_s), len(ood_s)
    if n_id == 0 or n_ood == 0: return 0.5
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
    results = {}
    ctypes = ['fog', 'night', 'noise', 'blur']

    print("Generating images...")
    seeds = list(range(0, 1000, 100))[:10]
    images = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        px = rng.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        images[seed] = Image.fromarray(px)

    # Extract ALL layer embeddings for clean images
    print("Extracting clean embeddings (all layers)...")
    clean_all_layers = {}
    for seed in seeds:
        clean_all_layers[seed] = extract_all_layers(model, processor, images[seed], prompt)

    n_layers = len(clean_all_layers[seeds[0]])
    print(f"  {n_layers} layers detected")

    # Compute per-layer centroids
    layer_centroids = []
    for L in range(n_layers):
        embs = [clean_all_layers[s][L] for s in seeds]
        layer_centroids.append(np.mean(embs, axis=0))

    # Clean distances per layer
    clean_dists_per_layer = []
    for L in range(n_layers):
        dists = [cosine_dist(clean_all_layers[s][L], layer_centroids[L]) for s in seeds]
        clean_dists_per_layer.append(dists)

    # ========== 1. Per-Layer Detection Distance ==========
    print("\n=== Per-Layer Detection Distance ===")

    layer_dists = {ct: [] for ct in ctypes}
    for ct in ctypes:
        for L in range(n_layers):
            ood_dists = []
            for seed in seeds[:5]:
                corrupt_img = apply_corruption(images[seed], ct, 0.5)
                embs = extract_all_layers(model, processor, corrupt_img, prompt)
                d = cosine_dist(embs[L], layer_centroids[L])
                ood_dists.append(d)
            layer_dists[ct].append(float(np.mean(ood_dists)))
        print(f"  {ct}: min_layer={np.argmin(layer_dists[ct])}, max_layer={np.argmax(layer_dists[ct])}, "
              f"max_dist={max(layer_dists[ct]):.6f}")

    results['layer_distances'] = layer_dists

    # ========== 2. Per-Layer AUROC ==========
    print("\n=== Per-Layer AUROC ===")

    layer_aurocs = {ct: [] for ct in ctypes}
    for ct in ctypes:
        for L in range(n_layers):
            id_scores = clean_dists_per_layer[L]
            ood_scores = []
            for seed in seeds[:5]:
                corrupt_img = apply_corruption(images[seed], ct, 0.5)
                embs = extract_all_layers(model, processor, corrupt_img, prompt)
                ood_scores.append(cosine_dist(embs[L], layer_centroids[L]))
            auroc = compute_auroc(id_scores, ood_scores)
            layer_aurocs[ct].append(float(auroc))

        perfect_layers = sum(1 for a in layer_aurocs[ct] if a >= 1.0)
        first_perfect = next((i for i, a in enumerate(layer_aurocs[ct]) if a >= 1.0), -1)
        print(f"  {ct}: {perfect_layers}/{n_layers} perfect, first_perfect=L{first_perfect}")

    results['layer_aurocs'] = layer_aurocs

    # ========== 3. Layer Gradient ==========
    print("\n=== Layer Gradient (rate of change) ===")

    layer_gradients = {}
    for ct in ctypes:
        grads = []
        for L in range(1, n_layers):
            grads.append(layer_dists[ct][L] - layer_dists[ct][L-1])
        max_jump_layer = np.argmax(np.abs(grads)) + 1
        layer_gradients[ct] = {
            'gradients': [float(g) for g in grads],
            'max_jump_layer': int(max_jump_layer),
            'max_jump_value': float(grads[max_jump_layer - 1]),
            'mean_gradient': float(np.mean(grads)),
        }
        print(f"  {ct}: max_jump at L{max_jump_layer} ({grads[max_jump_layer-1]:.6f})")

    results['layer_gradients'] = layer_gradients

    # ========== 4. Corruption Signature Per Layer ==========
    print("\n=== Corruption Signature ===")

    # For each layer, rank which corruption is most detectable
    signatures = {}
    for L in range(n_layers):
        dists = {ct: layer_dists[ct][L] for ct in ctypes}
        ranked = sorted(dists.items(), key=lambda x: x[1], reverse=True)
        signatures[f"L{L}"] = {
            'ranking': [r[0] for r in ranked],
            'distances': {ct: dists[ct] for ct in ctypes},
        }

    # Find where ranking changes
    rankings = [signatures[f"L{L}"]['ranking'] for L in range(n_layers)]
    transitions = []
    for L in range(1, n_layers):
        if rankings[L] != rankings[L-1]:
            transitions.append(L)

    results['corruption_signatures'] = {
        'per_layer': signatures,
        'ranking_transitions': transitions,
        'n_transitions': len(transitions),
    }
    print(f"  {len(transitions)} ranking transitions across {n_layers} layers")
    print(f"  First layer ranking: {rankings[0]}")
    print(f"  Last layer ranking: {rankings[-1]}")

    # ========== 5. Cross-Layer Correlation ==========
    print("\n=== Cross-Layer Correlation ===")

    # Build a matrix: for each corruption, compute distance vectors across layers
    # Then compute correlation between layer pairs
    corr_matrix = np.zeros((n_layers, n_layers))
    for i in range(n_layers):
        for j in range(n_layers):
            vec_i = [layer_dists[ct][i] for ct in ctypes]
            vec_j = [layer_dists[ct][j] for ct in ctypes]
            if np.std(vec_i) < 1e-15 or np.std(vec_j) < 1e-15:
                corr_matrix[i, j] = 1.0 if i == j else 0.0
            else:
                corr_matrix[i, j] = float(np.corrcoef(vec_i, vec_j)[0, 1])

    # Find layer clusters
    early_mid_corr = float(np.mean(corr_matrix[:n_layers//3, n_layers//3:2*n_layers//3]))
    mid_late_corr = float(np.mean(corr_matrix[n_layers//3:2*n_layers//3, 2*n_layers//3:]))
    early_late_corr = float(np.mean(corr_matrix[:n_layers//3, 2*n_layers//3:]))

    results['cross_layer_correlation'] = {
        'matrix': corr_matrix.tolist(),
        'early_mid_corr': early_mid_corr,
        'mid_late_corr': mid_late_corr,
        'early_late_corr': early_late_corr,
    }
    print(f"  Early-Mid: {early_mid_corr:.4f}")
    print(f"  Mid-Late: {mid_late_corr:.4f}")
    print(f"  Early-Late: {early_late_corr:.4f}")

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/layer_dynamics_{ts}.json"
    def convert(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)): return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return obj
    def recursive_convert(d):
        if isinstance(d, dict): return {k: recursive_convert(v) for k, v in d.items()}
        if isinstance(d, list): return [recursive_convert(x) for x in d]
        return convert(d)
    results = recursive_convert(results)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
