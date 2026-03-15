#!/usr/bin/env python3
"""Experiment 433: Embedding Trajectory Analysis

Tracks how hidden state embeddings evolve across the sequence of input tokens.
Instead of just the last token, examines the full trajectory of representations
across positions to understand WHERE corruption information enters.

Tests:
1. Per-position cosine distance (input tokens)
2. Visual token positions vs text token positions
3. Position-specific AUROC
4. Corruption information flow (first position where detection works)
5. Position ensemble detection
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

def extract_position_hidden(model, processor, image, prompt, layer=3):
    """Extract hidden states at ALL positions from a specific layer."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    # Return all positions from specified layer
    return fwd.hidden_states[layer][0].float().cpu().numpy()  # [seq_len, hidden_dim]

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

    seeds = [42, 123, 456, 789, 999]
    scenes = [Image.fromarray(np.random.RandomState(s).randint(0, 255, (224, 224, 3), dtype=np.uint8)) for s in seeds]

    # Extract full position embeddings for clean scenes
    print("Extracting position-wise embeddings for clean scenes...")
    clean_pos_embs = []  # [scene][position] = embedding
    for i, s in enumerate(scenes):
        embs = extract_position_hidden(model, processor, s, prompt)
        clean_pos_embs.append(embs)
        print(f"  Scene {i}: seq_len={embs.shape[0]}, dim={embs.shape[1]}")

    seq_len = clean_pos_embs[0].shape[0]
    results = {"n_scenes": len(scenes), "seq_len": seq_len}

    # Compute per-position centroids
    print(f"  Sequence length: {seq_len}")
    centroids = np.zeros((seq_len, clean_pos_embs[0].shape[1]))
    for pos in range(seq_len):
        pos_embs = [clean_pos_embs[i][pos] for i in range(len(scenes))]
        centroids[pos] = np.mean(pos_embs, axis=0)

    # Per-position clean distances
    clean_dists_per_pos = []
    for pos in range(seq_len):
        dists = [cosine_dist(clean_pos_embs[i][pos], centroids[pos]) for i in range(len(scenes))]
        clean_dists_per_pos.append(dists)

    # === Test 1: Per-position cosine distance for corruptions ===
    print("\n=== Per-Position Cosine Distance ===")
    # Sample positions: first 5, every 10th, last 5
    sample_positions = sorted(set(
        list(range(5)) +
        list(range(0, seq_len, max(1, seq_len // 20))) +
        list(range(max(0, seq_len - 5), seq_len))
    ))

    pos_distances = {}
    for c in corruptions:
        print(f"  Processing {c}...")
        c_dists = {}
        for s_idx, s in enumerate(scenes):
            corr_embs = extract_position_hidden(model, processor, apply_corruption(s, c), prompt)
            for pos in sample_positions:
                if pos not in c_dists:
                    c_dists[pos] = []
                c_dists[pos].append(float(cosine_dist(corr_embs[pos], centroids[pos])))

        pos_summary = {}
        for pos in sample_positions:
            pos_summary[str(pos)] = float(np.mean(c_dists[pos]))
        pos_distances[c] = pos_summary
        # Print key positions
        first_pos = sample_positions[0]
        last_pos = sample_positions[-1]
        mid_pos = sample_positions[len(sample_positions)//2]
        print(f"    pos={first_pos}: {pos_summary[str(first_pos)]:.6f}, "
              f"pos={mid_pos}: {pos_summary[str(mid_pos)]:.6f}, "
              f"pos={last_pos}: {pos_summary[str(last_pos)]:.6f}")
    results["pos_distances"] = pos_distances

    # === Test 2: Position-specific AUROC ===
    print("\n=== Position-Specific AUROC ===")
    pos_auroc = {}
    for c in corruptions:
        auroc_by_pos = {}
        for pos in sample_positions:
            ood_dists = []
            for s in scenes:
                corr_embs = extract_position_hidden(model, processor, apply_corruption(s, c), prompt)
                ood_dists.append(float(cosine_dist(corr_embs[pos], centroids[pos])))
            auroc = float(compute_auroc(clean_dists_per_pos[pos], ood_dists))
            auroc_by_pos[str(pos)] = auroc
        pos_auroc[c] = auroc_by_pos

        # Find first position with AUROC >= 0.9
        first_good = None
        for pos in sample_positions:
            if auroc_by_pos[str(pos)] >= 0.9:
                first_good = pos
                break
        print(f"  {c}: first AUROC≥0.9 at pos={first_good}, "
              f"last_pos AUROC={auroc_by_pos[str(sample_positions[-1])]:.4f}")
    results["pos_auroc"] = pos_auroc

    # === Test 3: Position embedding norms ===
    print("\n=== Position Embedding Norms ===")
    norm_trajectory = {}
    for pos in sample_positions:
        norms = [float(np.linalg.norm(clean_pos_embs[i][pos])) for i in range(len(scenes))]
        norm_trajectory[str(pos)] = {
            "mean": float(np.mean(norms)),
            "std": float(np.std(norms)),
        }
    results["norm_trajectory"] = norm_trajectory
    print(f"  Norm range: {norm_trajectory[str(sample_positions[0])]['mean']:.2f} "
          f"to {norm_trajectory[str(sample_positions[-1])]['mean']:.2f}")

    # === Test 4: Position ensemble detection ===
    print("\n=== Position Ensemble Detection ===")
    # Try combining early + middle + late positions
    early_pos = sample_positions[1] if len(sample_positions) > 1 else 0
    mid_pos = sample_positions[len(sample_positions) // 2]
    late_pos = sample_positions[-1]
    ensemble_positions = [early_pos, mid_pos, late_pos]

    ensemble_results = {}
    for c in corruptions:
        ens_clean = []
        ens_ood = []
        for s_idx in range(len(scenes)):
            scores = [clean_dists_per_pos[p][s_idx] for p in ensemble_positions]
            ens_clean.append(float(np.mean(scores)))

        for s in scenes:
            corr_embs = extract_position_hidden(model, processor, apply_corruption(s, c), prompt)
            scores = [float(cosine_dist(corr_embs[p], centroids[p])) for p in ensemble_positions]
            ens_ood.append(float(np.mean(scores)))

        auroc = float(compute_auroc(ens_clean, ens_ood))
        ensemble_results[c] = {
            "auroc": auroc,
            "positions_used": ensemble_positions,
        }
        print(f"  {c}: ensemble AUROC={auroc:.4f} (positions {ensemble_positions})")
    results["position_ensemble"] = ensemble_results

    # === Test 5: Clean variability per position ===
    print("\n=== Clean Variability Per Position ===")
    clean_var = {}
    for pos in sample_positions:
        dists = clean_dists_per_pos[pos]
        clean_var[str(pos)] = {
            "mean_dist": float(np.mean(dists)),
            "max_dist": float(np.max(dists)),
            "std_dist": float(np.std(dists)),
        }
    results["clean_variability"] = clean_var
    print(f"  Min variability at pos={min(sample_positions, key=lambda p: np.mean(clean_dists_per_pos[p]))}")
    print(f"  Max variability at pos={max(sample_positions, key=lambda p: np.mean(clean_dists_per_pos[p]))}")

    out_path = "/workspace/Vizuara-VLA-Research/experiments/embedding_trajectory_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
