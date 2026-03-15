#!/usr/bin/env python3
"""Experiment 413: Embedding Dimensionality Analysis

Studies the effective dimensionality of the embedding space and how
corruption detection relates to intrinsic vs. ambient dimensionality.

Tests:
1. Intrinsic dimensionality estimation (PCA, participation ratio)
2. Random projection: minimum dimensions for AUROC=1.0
3. Per-corruption dimensionality requirements
4. Feature selection: which embedding dimensions matter most?
5. Sparse projections vs dense projections
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

    # Generate scenes
    seeds = [42, 123, 456, 789, 999]
    scenes = [Image.fromarray(np.random.RandomState(s).randint(0, 255, (224, 224, 3), dtype=np.uint8)) for s in seeds]

    # Extract embeddings
    print("Extracting embeddings...")
    clean_embs = [extract_hidden(model, processor, s, prompt) for s in scenes]
    centroid = np.mean(clean_embs, axis=0)
    id_scores = [cosine_dist(e, centroid) for e in clean_embs]

    corrupt_embs = {}
    ood_scores = {}
    for c in corruptions:
        corrupt_embs[c] = [extract_hidden(model, processor, apply_corruption(s, c), prompt) for s in scenes]
        ood_scores[c] = [cosine_dist(e, centroid) for e in corrupt_embs[c]]

    results = {}

    # === Test 1: PCA dimensionality ===
    print("\n=== PCA Dimensionality ===")
    all_embs = clean_embs.copy()
    for c in corruptions:
        all_embs.extend(corrupt_embs[c])
    X = np.array(all_embs)
    X_centered = X - X.mean(axis=0)

    # SVD (more efficient than eigh for thin matrix)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    eigenvalues = S**2 / (len(X) - 1)
    total_var = np.sum(eigenvalues)
    cumulative_var = np.cumsum(eigenvalues) / total_var

    # Participation ratio (effective dimensionality)
    pr = np.sum(eigenvalues)**2 / np.sum(eigenvalues**2)

    pca_results = {
        "participation_ratio": float(pr),
        "top_eigenvalues": [float(e) for e in eigenvalues[:20]],
        "dims_for_90pct": int(np.searchsorted(cumulative_var, 0.9) + 1),
        "dims_for_95pct": int(np.searchsorted(cumulative_var, 0.95) + 1),
        "dims_for_99pct": int(np.searchsorted(cumulative_var, 0.99) + 1),
        "cumulative_variance_10d": float(cumulative_var[9]),
        "cumulative_variance_20d": float(cumulative_var[19]) if len(cumulative_var) > 19 else 1.0
    }
    results["pca"] = pca_results
    print(f"  Participation ratio: {pr:.1f}")
    print(f"  90% variance: {pca_results['dims_for_90pct']}D, 95%: {pca_results['dims_for_95pct']}D, 99%: {pca_results['dims_for_99pct']}D")

    # === Test 2: Random projection minimum dimensions ===
    print("\n=== Minimum Projection Dimensions ===")
    proj_dims = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    proj_results = {}

    for d in proj_dims:
        aurocs_per_trial = []
        for trial in range(10):
            rng = np.random.RandomState(trial)
            proj = rng.randn(4096, d).astype(np.float64) / np.sqrt(d)

            proj_centroid = centroid @ proj
            proj_id = [cosine_dist(e @ proj, proj_centroid) for e in clean_embs]

            trial_aurocs = []
            for c in corruptions:
                proj_ood = [cosine_dist(e @ proj, proj_centroid) for e in corrupt_embs[c]]
                trial_aurocs.append(compute_auroc(proj_id, proj_ood))

            aurocs_per_trial.append(np.mean(trial_aurocs))

        proj_results[str(d)] = {
            "mean_auroc": float(np.mean(aurocs_per_trial)),
            "min_auroc": float(np.min(aurocs_per_trial)),
            "std_auroc": float(np.std(aurocs_per_trial)),
            "n_perfect": sum(1 for a in aurocs_per_trial if a >= 1.0)
        }
        print(f"  {d}D: AUROC={np.mean(aurocs_per_trial):.4f} ± {np.std(aurocs_per_trial):.4f}")

    results["random_projection"] = proj_results

    # === Test 3: Per-corruption dimensionality ===
    print("\n=== Per-Corruption Dimensionality ===")
    per_corruption = {}
    for c in corruptions:
        min_dim_perfect = None
        for d in proj_dims:
            aurocs = []
            for trial in range(10):
                rng = np.random.RandomState(trial)
                proj = rng.randn(4096, d).astype(np.float64) / np.sqrt(d)
                proj_centroid = centroid @ proj
                proj_id = [cosine_dist(e @ proj, proj_centroid) for e in clean_embs]
                proj_ood = [cosine_dist(e @ proj, proj_centroid) for e in corrupt_embs[c]]
                aurocs.append(compute_auroc(proj_id, proj_ood))

            if all(a >= 1.0 for a in aurocs) and min_dim_perfect is None:
                min_dim_perfect = d

            per_corruption[c] = per_corruption.get(c, {})
            per_corruption[c][str(d)] = float(np.mean(aurocs))

        per_corruption[c]["min_dim_perfect"] = min_dim_perfect
        print(f"  {c}: min dim for perfect AUROC = {min_dim_perfect}")

    results["per_corruption_dims"] = per_corruption

    # === Test 4: Feature selection (top-k dimensions by variance) ===
    print("\n=== Feature Selection (Top-k by Variance) ===")
    # Displacement vectors
    displacements = []
    for c in corruptions:
        for si in range(len(scenes)):
            displacements.append(corrupt_embs[c][si] - clean_embs[si])
    displacements = np.array(displacements)

    # Variance per dimension
    dim_var = np.var(displacements, axis=0)
    ranked_dims = np.argsort(dim_var)[::-1]

    feature_selection = {}
    for k in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        top_dims = ranked_dims[:k]

        # Use only these dimensions
        sel_centroid = centroid[top_dims]
        sel_id = [cosine_dist(e[top_dims], sel_centroid) for e in clean_embs]
        sel_ood = []
        for c in corruptions:
            sel_ood.extend([cosine_dist(e[top_dims], sel_centroid) for e in corrupt_embs[c]])

        auroc = compute_auroc(sel_id, sel_ood)
        feature_selection[str(k)] = {
            "auroc": float(auroc),
            "top_dim_indices": [int(d) for d in top_dims[:5]]
        }
        print(f"  Top-{k} dims: AUROC={auroc:.4f}")

    results["feature_selection"] = feature_selection

    # === Test 5: Sparse vs dense projection ===
    print("\n=== Sparse vs Dense Projection ===")
    sparse_results = {}
    d = 32  # fixed dimension

    for sparsity_name, proj_fn in [
        ("dense_gaussian", lambda rng: rng.randn(4096, d) / np.sqrt(d)),
        ("sparse_1/3", lambda rng: rng.choice([-1, 0, 0, 0, 0, 1], size=(4096, d)).astype(np.float64) * np.sqrt(3/d)),
        ("sparse_1/sqrt", lambda rng: rng.choice([-1, 0, 1], size=(4096, d)).astype(np.float64) / np.sqrt(d)),
        ("binary", lambda rng: rng.choice([-1, 1], size=(4096, d)).astype(np.float64) / np.sqrt(d)),
    ]:
        aurocs = []
        for trial in range(10):
            rng = np.random.RandomState(trial)
            proj = proj_fn(rng)
            proj_centroid = centroid @ proj
            proj_id = [cosine_dist(e @ proj, proj_centroid) for e in clean_embs]
            proj_ood = []
            for c in corruptions:
                proj_ood.extend([cosine_dist(e @ proj, proj_centroid) for e in corrupt_embs[c]])
            aurocs.append(compute_auroc(proj_id, proj_ood))

        sparse_results[sparsity_name] = {
            "mean_auroc": float(np.mean(aurocs)),
            "std_auroc": float(np.std(aurocs)),
            "n_perfect": sum(1 for a in aurocs if a >= 1.0)
        }
        print(f"  {sparsity_name}: AUROC={np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")

    results["sparse_vs_dense"] = sparse_results

    # Save
    out_path = "/workspace/Vizuara-VLA-Research/experiments/embedding_dimensionality_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
