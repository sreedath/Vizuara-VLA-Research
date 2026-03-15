#!/usr/bin/env python3
"""Experiment 420: Token Position Analysis

Studies how different token positions in the input sequence contribute to
OOD detection. The model processes image tokens and text tokens — which
positions carry the corruption signal? Is it the image patch tokens,
the text prompt tokens, or the final aggregation token?

Tests:
1. Per-position hidden state extraction and distance
2. Image patch positions vs text positions
3. Mean/max pooling over positions vs last-token only
4. Positional sensitivity: which token positions shift most under corruption?
5. Position-wise AUROC: can individual positions detect corruption?
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

def extract_all_positions(model, processor, image, prompt, layer=3):
    """Extract hidden states for ALL token positions."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return fwd.hidden_states[layer][0, :, :].float().cpu().numpy()

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

    seeds = [42, 123, 456]
    scenes = [Image.fromarray(np.random.RandomState(s).randint(0, 255, (224, 224, 3), dtype=np.uint8)) for s in seeds]

    print("Extracting all-position embeddings...")
    clean_pos = [extract_all_positions(model, processor, s, prompt) for s in scenes]
    seq_len = clean_pos[0].shape[0]
    hidden_dim = clean_pos[0].shape[1]
    print(f"  Sequence length: {seq_len}, hidden dim: {hidden_dim}")

    corrupt_pos = {}
    for c in corruptions:
        corrupt_pos[c] = [extract_all_positions(model, processor, apply_corruption(s, c), prompt) for s in scenes]
        print(f"  {c} extracted")

    results = {"seq_len": seq_len, "hidden_dim": hidden_dim}

    # Per-position centroids
    pos_centroids = np.mean([cp for cp in clean_pos], axis=0)

    # Per-position clean distances
    clean_pos_dists = np.zeros((len(scenes), seq_len))
    for s in range(len(scenes)):
        for p in range(seq_len):
            clean_pos_dists[s, p] = cosine_dist(clean_pos[s][p], pos_centroids[p])

    corrupt_pos_dists = {}
    for c in corruptions:
        corrupt_pos_dists[c] = np.zeros((len(scenes), seq_len))
        for s in range(len(scenes)):
            for p in range(seq_len):
                corrupt_pos_dists[c][s, p] = cosine_dist(corrupt_pos[c][s][p], pos_centroids[p])

    sample_positions = list(range(0, seq_len, max(1, seq_len // 20))) + [seq_len - 1]
    sample_positions = sorted(set(sample_positions))

    # Test 1: Per-position distance
    position_dists = {}
    for p in sample_positions:
        entry = {"clean_mean": float(np.mean(clean_pos_dists[:, p]))}
        for c in corruptions:
            entry[f"{c}_mean"] = float(np.mean(corrupt_pos_dists[c][:, p]))
        position_dists[str(p)] = entry
    results["position_dists"] = position_dists

    # Test 2: Per-position AUROC
    print("\n=== Per-Position AUROC ===")
    pos_aurocs = {}
    for p in sample_positions:
        id_scores = clean_pos_dists[:, p].tolist()
        per_corr = {}
        for c in corruptions:
            ood_scores = corrupt_pos_dists[c][:, p].tolist()
            per_corr[c] = float(compute_auroc(id_scores, ood_scores))
        all_ood = []
        for c in corruptions:
            all_ood.extend(corrupt_pos_dists[c][:, p].tolist())
        overall = float(compute_auroc(id_scores, all_ood))
        pos_aurocs[str(p)] = {"overall": overall, "per_corruption": per_corr}
        if p % (seq_len // 5) == 0 or p == seq_len - 1:
            print(f"  Pos {p}: overall={overall:.4f}")
    results["position_aurocs"] = pos_aurocs

    # Test 3: Pooling strategies
    print("\n=== Pooling Strategies ===")
    pooling = {}
    for strategy in ['last', 'mean', 'max', 'first']:
        id_scores = []
        for s in range(len(scenes)):
            if strategy == 'last':
                emb = clean_pos[s][-1]
                cent = pos_centroids[-1]
            elif strategy == 'mean':
                emb = np.mean(clean_pos[s], axis=0)
                cent = np.mean(pos_centroids, axis=0)
            elif strategy == 'max':
                emb = np.max(clean_pos[s], axis=0)
                cent = np.max(pos_centroids, axis=0)
            elif strategy == 'first':
                emb = clean_pos[s][0]
                cent = pos_centroids[0]
            id_scores.append(cosine_dist(emb, cent))

        all_ood = []
        for c in corruptions:
            for s in range(len(scenes)):
                if strategy == 'last':
                    emb = corrupt_pos[c][s][-1]
                elif strategy == 'mean':
                    emb = np.mean(corrupt_pos[c][s], axis=0)
                elif strategy == 'max':
                    emb = np.max(corrupt_pos[c][s], axis=0)
                elif strategy == 'first':
                    emb = corrupt_pos[c][s][0]
                all_ood.append(cosine_dist(emb, cent))

        auroc = compute_auroc(id_scores, all_ood)
        pooling[strategy] = {"auroc": float(auroc)}
        print(f"  {strategy}: AUROC={auroc:.4f}")
    results["pooling"] = pooling

    # Test 4: Position sensitivity
    print("\n=== Position Sensitivity ===")
    sensitivity = np.zeros(seq_len)
    for c in corruptions:
        for s in range(len(scenes)):
            for p in range(seq_len):
                sensitivity[p] += cosine_dist(corrupt_pos[c][s][p], clean_pos[s][p])
    sensitivity /= (len(corruptions) * len(scenes))

    ranked = np.argsort(sensitivity)[::-1]
    top_sensitive = [(int(ranked[i]), float(sensitivity[ranked[i]])) for i in range(min(10, len(ranked)))]
    bot_sensitive = [(int(ranked[-(i+1)]), float(sensitivity[ranked[-(i+1)]])) for i in range(min(10, len(ranked)))]

    results["sensitivity"] = {
        "top_10": top_sensitive,
        "bottom_10": bot_sensitive,
        "mean": float(np.mean(sensitivity)),
        "std": float(np.std(sensitivity)),
        "last_position": float(sensitivity[-1]),
        "last_position_rank": int(np.where(ranked == seq_len - 1)[0][0]) + 1,
    }
    print(f"  Most sensitive: pos {top_sensitive[0][0]} (dist={top_sensitive[0][1]:.6f})")
    print(f"  Last position: dist={sensitivity[-1]:.6f} (rank {results['sensitivity']['last_position_rank']}/{seq_len})")

    out_path = "/workspace/Vizuara-VLA-Research/experiments/token_position_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
