#!/usr/bin/env python3
"""Experiment 296: Information-Theoretic Analysis
Computes information-theoretic properties of the OOD signal:
1. Differential entropy of clean vs corrupted embeddings
2. Mutual information between corruption type and embedding
3. Fisher information of embedding w.r.t. severity
4. Embedding compression via SVD (effective rank)
5. Channel capacity of the corruption detection channel
"""

import torch
import numpy as np
import json
from datetime import datetime
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor
from scipy.spatial.distance import cosine
from scipy.stats import entropy

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

def main():
    print("Loading OpenVLA-7B...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    np.random.seed(42)
    pixels = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    base_img = Image.fromarray(pixels)

    results = {
        "experiment": "information_theory",
        "experiment_number": 296,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    # Collect embeddings
    corruptions = ['fog', 'night', 'blur', 'noise']
    severities = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    print("Collecting embeddings...")
    embeddings = {}
    for c in corruptions:
        embeddings[c] = []
        for s in severities:
            img = apply_corruption(base_img, c, s) if s > 0 else base_img
            emb = extract_hidden(model, processor, img, prompt)
            embeddings[c].append(emb)

    clean_emb = embeddings[corruptions[0]][0]  # severity=0

    # Part 1: Differential entropy of embeddings
    print("\n=== Part 1: Embedding Entropy ===")
    entropy_results = {}

    for c in corruptions:
        emb_matrix = np.array(embeddings[c])  # (11, 4096)
        # Compute per-dimension variance
        dim_vars = np.var(emb_matrix, axis=0)
        # Entropy of variance distribution (how spread out the information is)
        var_normalized = dim_vars / (dim_vars.sum() + 1e-30)
        dim_entropy = float(entropy(var_normalized + 1e-30))

        # Effective number of dimensions (exponential of entropy)
        effective_dims = np.exp(dim_entropy)

        # SVD to get singular values
        U, S, Vt = np.linalg.svd(emb_matrix - emb_matrix.mean(axis=0), full_matrices=False)
        sv_normalized = S / (S.sum() + 1e-30)
        sv_entropy = float(entropy(sv_normalized + 1e-30))

        # Effective rank (Vershynin definition)
        effective_rank = np.exp(sv_entropy)

        entropy_results[c] = {
            "dim_entropy": dim_entropy,
            "effective_dims": float(effective_dims),
            "sv_entropy": sv_entropy,
            "effective_rank": float(effective_rank),
            "top_5_singular_values": S[:5].tolist(),
            "variance_ratio_top1": float(S[0]**2 / (S**2).sum()),
            "variance_ratio_top3": float((S[:3]**2).sum() / (S**2).sum())
        }
        print(f"  {c}: effective_rank={effective_rank:.2f}, dim_entropy={dim_entropy:.2f}, "
              f"top1_var={S[0]**2/(S**2).sum():.3f}")

    results["entropy"] = entropy_results

    # Part 2: Fisher information of embedding w.r.t. severity
    print("\n=== Part 2: Fisher Information ===")
    fisher_results = {}

    for c in corruptions:
        emb_matrix = np.array(embeddings[c])  # (11, 4096)
        sev_array = np.array(severities)

        # Compute derivative of embedding w.r.t. severity (finite differences)
        d_emb_d_sev = np.diff(emb_matrix, axis=0) / np.diff(sev_array)[:, None]  # (10, 4096)

        # Fisher information = E[||d_emb/d_sev||^2]
        fisher = np.mean(np.sum(d_emb_d_sev**2, axis=1))

        # Per-severity Fisher
        per_sev_fisher = np.sum(d_emb_d_sev**2, axis=1).tolist()

        fisher_results[c] = {
            "mean_fisher": float(fisher),
            "per_severity_fisher": per_sev_fisher,
            "max_fisher_severity": float(severities[1 + np.argmax(per_sev_fisher)])
        }
        print(f"  {c}: mean_fisher={fisher:.6f}, max at sev={fisher_results[c]['max_fisher_severity']:.1f}")

    results["fisher_information"] = fisher_results

    # Part 3: Mutual information between corruption type and embedding direction
    print("\n=== Part 3: Corruption-Embedding Mutual Information ===")
    # Compute direction vectors for each corruption at severity 1.0
    directions = {}
    for c in corruptions:
        diff = embeddings[c][-1] - clean_emb  # severity 1.0 - clean
        norm = np.linalg.norm(diff)
        if norm > 0:
            directions[c] = diff / norm
        else:
            directions[c] = diff

    # Direction similarity matrix
    dir_matrix = {}
    for c1 in corruptions:
        for c2 in corruptions:
            sim = float(np.dot(directions[c1], directions[c2]))
            dir_matrix[f"{c1}_vs_{c2}"] = sim

    # Compute "classification accuracy" as proxy for MI
    correct = 0
    total = 0
    for c_true in corruptions:
        for sev_idx in range(1, len(severities)):  # exclude clean
            emb = embeddings[c_true][sev_idx]
            diff = emb - clean_emb
            # Classify by nearest direction
            best_c = max(corruptions, key=lambda c: np.dot(diff, directions[c]))
            if best_c == c_true:
                correct += 1
            total += 1

    classification_accuracy = correct / total

    results["mutual_information"] = {
        "direction_similarity": dir_matrix,
        "classification_accuracy": classification_accuracy,
        "n_correct": correct,
        "n_total": total
    }
    print(f"  Direction-based classification: {correct}/{total} = {classification_accuracy:.3f}")

    # Part 4: Channel capacity analysis
    print("\n=== Part 4: Channel Capacity ===")
    # Model the detection as a binary channel
    # Clean: d = 0 (deterministic)
    # Corrupted: d > 0 (always)
    # Channel capacity = 1 bit (perfect detection)

    # For severity estimation channel:
    # How many severity levels can be distinguished?
    distinguishable_levels = {}
    for c in corruptions:
        dists = [float(cosine(clean_emb, embeddings[c][i])) for i in range(len(severities))]
        # Count distinguishable levels (each must be larger than previous)
        prev = -1
        n_levels = 0
        for d in dists:
            if d > prev:
                n_levels += 1
                prev = d
        distinguishable_levels[c] = {
            "n_levels": n_levels,
            "bits": float(np.log2(n_levels)),
            "distances": dists
        }
        print(f"  {c}: {n_levels} distinguishable levels = {np.log2(n_levels):.2f} bits")

    results["channel_capacity"] = {
        "binary_detection_bits": 1.0,
        "severity_levels": distinguishable_levels
    }

    # Part 5: Compression analysis
    print("\n=== Part 5: Compression Analysis ===")
    all_embs = []
    for c in corruptions:
        for emb in embeddings[c]:
            all_embs.append(emb)
    all_embs = np.array(all_embs)

    U, S, Vt = np.linalg.svd(all_embs - all_embs.mean(axis=0), full_matrices=False)
    cum_var = np.cumsum(S**2) / np.sum(S**2)

    compression = {
        "dims_for_90pct": int(np.searchsorted(cum_var, 0.90)) + 1,
        "dims_for_95pct": int(np.searchsorted(cum_var, 0.95)) + 1,
        "dims_for_99pct": int(np.searchsorted(cum_var, 0.99)) + 1,
        "top_10_singular": S[:10].tolist(),
        "cumulative_variance": cum_var[:20].tolist()
    }
    results["compression"] = compression
    print(f"  90%: {compression['dims_for_90pct']}D, 95%: {compression['dims_for_95pct']}D, "
          f"99%: {compression['dims_for_99pct']}D")

    # Save
    ts = results["timestamp"]
    out_path = f"experiments/information_{ts}.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
