#!/usr/bin/env python3
"""Experiment 424: Embedding Space Geometry Analysis

Deep analysis of the embedding space structure: eigenspectrum, nearest-neighbor
structure, manifold dimensionality, and how corruption deforms the geometry.

Tests:
1. Eigenspectrum of clean vs corrupted embedding covariance
2. Inter-corruption embedding relationships (which corruptions cluster?)
3. Embedding norm analysis (do corruptions change vector magnitudes?)
4. Principal component projection: clean vs corrupt separation
5. Mahalanobis distance vs cosine distance for detection
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

    seeds = [42, 123, 456, 789, 999, 1234, 5678, 9999]
    scenes = [Image.fromarray(np.random.RandomState(s).randint(0, 255, (224, 224, 3), dtype=np.uint8)) for s in seeds]

    print(f"Extracting embeddings for {len(scenes)} scenes...")
    clean_embs = np.array([extract_hidden(model, processor, s, prompt) for s in scenes])
    centroid = np.mean(clean_embs, axis=0)
    print(f"  Clean embeddings shape: {clean_embs.shape}")

    corrupt_embs = {}
    for c in corruptions:
        corrupt_embs[c] = np.array([extract_hidden(model, processor, apply_corruption(s, c), prompt) for s in scenes])
        print(f"  {c} extracted")

    results = {"n_scenes": len(scenes), "hidden_dim": int(clean_embs.shape[1])}

    # === Test 1: Eigenspectrum analysis ===
    print("\n=== Eigenspectrum Analysis ===")
    # Clean covariance
    clean_centered = clean_embs - centroid
    if clean_centered.shape[0] > 1:
        clean_cov = np.cov(clean_centered.T)
        clean_eigvals = np.sort(np.linalg.eigvalsh(clean_cov))[::-1]
    else:
        clean_eigvals = np.array([0.0])

    eigenspectrum = {
        "clean_top_10": [float(v) for v in clean_eigvals[:10]],
        "clean_total_variance": float(np.sum(clean_eigvals[clean_eigvals > 0])),
        "clean_effective_dim": float(np.sum(clean_eigvals > 1e-10)),
    }

    for c in corruptions:
        all_embs = np.vstack([clean_embs, corrupt_embs[c]])
        all_centered = all_embs - np.mean(all_embs, axis=0)
        if all_centered.shape[0] > 1:
            cov = np.cov(all_centered.T)
            eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
        else:
            eigvals = np.array([0.0])
        eigenspectrum[f"{c}_top_10"] = [float(v) for v in eigvals[:10]]
        eigenspectrum[f"{c}_total_variance"] = float(np.sum(eigvals[eigvals > 0]))
        eigenspectrum[f"{c}_effective_dim"] = float(np.sum(eigvals > 1e-10))
        # Participation ratio
        if np.sum(eigvals) > 0:
            p = eigvals[eigvals > 0] / np.sum(eigvals[eigvals > 0])
            pr = 1.0 / np.sum(p ** 2)
        else:
            pr = 0.0
        eigenspectrum[f"{c}_participation_ratio"] = float(pr)
        print(f"  {c}: eff_dim={eigenspectrum[f'{c}_effective_dim']:.0f}, PR={pr:.2f}")
    results["eigenspectrum"] = eigenspectrum

    # === Test 2: Inter-corruption relationships ===
    print("\n=== Inter-Corruption Clustering ===")
    corr_centroids = {c: np.mean(corrupt_embs[c], axis=0) for c in corruptions}
    inter_corruption = {}
    for c1 in corruptions:
        for c2 in corruptions:
            if c1 >= c2:
                continue
            d = cosine_dist(corr_centroids[c1], corr_centroids[c2])
            inter_corruption[f"{c1}_vs_{c2}"] = float(d)
            print(f"  {c1} vs {c2}: dist={d:.6f}")
    # Distance from each corruption centroid to clean centroid
    for c in corruptions:
        d = cosine_dist(corr_centroids[c], centroid)
        inter_corruption[f"{c}_to_clean"] = float(d)
        print(f"  {c} to clean: dist={d:.6f}")
    results["inter_corruption"] = inter_corruption

    # === Test 3: Embedding norm analysis ===
    print("\n=== Embedding Norm Analysis ===")
    norms = {"clean": {
        "mean": float(np.mean(np.linalg.norm(clean_embs, axis=1))),
        "std": float(np.std(np.linalg.norm(clean_embs, axis=1))),
        "min": float(np.min(np.linalg.norm(clean_embs, axis=1))),
        "max": float(np.max(np.linalg.norm(clean_embs, axis=1))),
    }}
    print(f"  Clean: norm={norms['clean']['mean']:.4f} ± {norms['clean']['std']:.4f}")
    for c in corruptions:
        cn = np.linalg.norm(corrupt_embs[c], axis=1)
        norms[c] = {
            "mean": float(np.mean(cn)),
            "std": float(np.std(cn)),
            "min": float(np.min(cn)),
            "max": float(np.max(cn)),
            "ratio_to_clean": float(np.mean(cn) / norms["clean"]["mean"]),
        }
        print(f"  {c}: norm={norms[c]['mean']:.4f}, ratio={norms[c]['ratio_to_clean']:.4f}")
    results["embedding_norms"] = norms

    # === Test 4: PCA projection ===
    print("\n=== PCA Projection Analysis ===")
    pca_results = {}
    for c in corruptions:
        all_embs = np.vstack([clean_embs, corrupt_embs[c]])
        mean_emb = np.mean(all_embs, axis=0)
        centered = all_embs - mean_emb
        # SVD for top 2 components
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        proj = centered @ Vt[:2].T  # (n_samples, 2)
        clean_proj = proj[:len(scenes)]
        corrupt_proj = proj[len(scenes):]

        # Separation in 2D
        clean_mean_2d = np.mean(clean_proj, axis=0)
        corrupt_mean_2d = np.mean(corrupt_proj, axis=0)
        separation = np.linalg.norm(clean_mean_2d - corrupt_mean_2d)

        # Variance explained by top 2
        total_var = np.sum(S ** 2)
        top2_var = np.sum(S[:2] ** 2)

        pca_results[c] = {
            "separation_2d": float(separation),
            "variance_explained_top2": float(top2_var / total_var) if total_var > 0 else 0.0,
            "singular_values_top5": [float(v) for v in S[:5]],
        }
        print(f"  {c}: 2D separation={separation:.4f}, var_explained={top2_var/total_var:.4f}")
    results["pca_projection"] = pca_results

    # === Test 5: Mahalanobis vs Cosine distance ===
    print("\n=== Mahalanobis vs Cosine Distance ===")
    # Compute Mahalanobis distance using clean sample covariance
    clean_centered = clean_embs - centroid

    # Use pseudo-inverse for low-rank covariance
    U, S, Vt = np.linalg.svd(clean_centered, full_matrices=False)
    # Keep components with non-negligible variance
    thresh = 1e-8 * S[0]
    k = int(np.sum(S > thresh))
    if k > 0:
        S_inv = np.zeros_like(S)
        S_inv[:k] = 1.0 / S[:k]
        # Precision matrix in reduced space
        precision_sqrt = np.diag(S_inv[:k]) @ Vt[:k]
    else:
        precision_sqrt = np.eye(clean_embs.shape[1])

    def mahal_dist(x, center, prec_sqrt):
        diff = x - center
        projected = prec_sqrt @ diff
        return float(np.linalg.norm(projected))

    maha_results = {}
    clean_cosine_dists = [cosine_dist(e, centroid) for e in clean_embs]
    clean_mahal_dists = [mahal_dist(e, centroid, precision_sqrt) for e in clean_embs]

    for c in corruptions:
        ood_cosine = [cosine_dist(e, centroid) for e in corrupt_embs[c]]
        ood_mahal = [mahal_dist(e, centroid, precision_sqrt) for e in corrupt_embs[c]]

        auroc_cosine = float(compute_auroc(clean_cosine_dists, ood_cosine))
        auroc_mahal = float(compute_auroc(clean_mahal_dists, ood_mahal))

        maha_results[c] = {
            "cosine_auroc": auroc_cosine,
            "mahal_auroc": auroc_mahal,
            "cosine_mean_ood": float(np.mean(ood_cosine)),
            "mahal_mean_ood": float(np.mean(ood_mahal)),
        }
        print(f"  {c}: cosine AUROC={auroc_cosine:.4f}, Mahalanobis AUROC={auroc_mahal:.4f}")

    # Overall
    all_ood_cos = []
    all_ood_mah = []
    for c in corruptions:
        all_ood_cos.extend([cosine_dist(e, centroid) for e in corrupt_embs[c]])
        all_ood_mah.extend([mahal_dist(e, centroid, precision_sqrt) for e in corrupt_embs[c]])
    maha_results["overall_cosine_auroc"] = float(compute_auroc(clean_cosine_dists, all_ood_cos))
    maha_results["overall_mahal_auroc"] = float(compute_auroc(clean_mahal_dists, all_ood_mah))
    print(f"  Overall: cosine={maha_results['overall_cosine_auroc']:.4f}, Mahalanobis={maha_results['overall_mahal_auroc']:.4f}")
    results["distance_comparison"] = maha_results

    out_path = "/workspace/Vizuara-VLA-Research/experiments/embedding_geometry_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
