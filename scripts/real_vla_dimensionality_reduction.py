#!/usr/bin/env python3
"""Experiment 389: Embedding Dimensionality Reduction for Detection

Tests whether the 4096-D embeddings can be compressed via random projection,
PCA, or truncation while maintaining OOD detection performance. Since intrinsic
dimensionality is ~3-4D, massive compression should be possible.

Tests:
1. Random projection to k dims (k=2,4,8,16,32,64,128,256,512,1024,2048)
2. PCA projection to k dims
3. Top-k component truncation
4. Detection AUROC at each compression level
5. Centroid stability under compression
6. Latency comparison
"""

import json, time, os, sys
import numpy as np
import torch
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor

def extract_hidden(model, processor, image, prompt, layer=3):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return fwd.hidden_states[layer][0, -1, :].float().cpu().numpy()

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

def compute_auroc(id_scores, ood_scores):
    id_s = np.asarray(id_scores)
    ood_s = np.asarray(ood_scores)
    n_id, n_ood = len(id_s), len(ood_s)
    if n_id == 0 or n_ood == 0:
        return 0.5
    count = sum(float(np.sum(o > id_s) + 0.5 * np.sum(o == id_s)) for o in ood_s)
    return count / (n_id * n_ood)

def cosine_dist(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    return 1.0 - np.dot(a, b) / (na * nb)

def main():
    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    img = Image.fromarray(np.random.RandomState(42).randint(0, 255, (224, 224, 3), dtype=np.uint8))

    corruptions = ['fog', 'night', 'noise', 'blur']
    n_clean = 10
    n_ood = 10

    # Collect full-dimensional embeddings
    print("Collecting clean embeddings...")
    clean_embs = []
    for i in range(n_clean):
        # Small pixel perturbation for variety
        arr = np.array(img).astype(np.float32)
        arr += np.random.RandomState(100 + i).randn(*arr.shape) * 0.5
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        perturbed = Image.fromarray(arr)
        emb = extract_hidden(model, processor, perturbed, prompt)
        clean_embs.append(emb)
        print(f"  Clean {i+1}/{n_clean}")

    print("Collecting corrupted embeddings...")
    corrupt_embs = {}
    for c in corruptions:
        corrupt_embs[c] = []
        for i in range(n_ood):
            arr = np.array(img).astype(np.float32)
            arr += np.random.RandomState(200 + i).randn(*arr.shape) * 0.5
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            perturbed = Image.fromarray(arr)
            corrupted = apply_corruption(perturbed, c)
            emb = extract_hidden(model, processor, corrupted, prompt)
            corrupt_embs[c].append(emb)
            print(f"  {c} {i+1}/{n_ood}")

    clean_embs = np.array(clean_embs)  # (n_clean, 4096)
    centroid_full = clean_embs.mean(axis=0)
    D = clean_embs.shape[1]

    # Full-dimensional AUROC baseline
    print("\nFull-dimensional baseline...")
    full_aurocs = {}
    for c in corruptions:
        id_scores = [cosine_dist(e, centroid_full) for e in clean_embs]
        ood_scores = [cosine_dist(e, centroid_full) for e in corrupt_embs[c]]
        full_aurocs[c] = compute_auroc(id_scores, ood_scores)
    print(f"  Full-dim AUROCs: {full_aurocs}")

    # Compression dimensions to test
    k_dims = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

    results = {
        "full_dim": D,
        "n_clean": n_clean,
        "n_ood": n_ood,
        "full_aurocs": full_aurocs,
        "random_projection": {},
        "pca_projection": {},
        "top_k_truncation": {},
        "centroid_stability": {},
        "latency": {}
    }

    # 1. Random Projection
    print("\n=== Random Projection ===")
    for k in k_dims:
        if k > D:
            continue
        # Generate random projection matrix (Gaussian)
        rng = np.random.RandomState(999)
        R = rng.randn(D, k).astype(np.float64) / np.sqrt(k)

        # Project
        clean_proj = clean_embs @ R
        centroid_proj = clean_proj.mean(axis=0)

        aurocs = {}
        for c in corruptions:
            corrupt_proj = np.array(corrupt_embs[c]) @ R
            id_scores = [cosine_dist(e, centroid_proj) for e in clean_proj]
            ood_scores = [cosine_dist(e, centroid_proj) for e in corrupt_proj]
            aurocs[c] = compute_auroc(id_scores, ood_scores)

        # Centroid alignment with full-dim
        cent_proj_back = centroid_proj @ R.T  # back-project for comparison
        cent_align = 1.0 - cosine_dist(cent_proj_back, centroid_full)

        results["random_projection"][k] = {
            "aurocs": aurocs,
            "centroid_alignment": float(cent_align)
        }
        print(f"  k={k}: {aurocs}, alignment={cent_align:.6f}")

    # 2. PCA Projection
    print("\n=== PCA Projection ===")
    # Compute PCA on clean embeddings
    centered = clean_embs - centroid_full
    # Use SVD for PCA
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    # Vt rows are principal components
    explained_var = S**2 / np.sum(S**2)
    cumulative_var = np.cumsum(explained_var)

    results["pca_explained_variance"] = explained_var[:20].tolist()
    results["pca_cumulative_variance"] = cumulative_var[:20].tolist()

    for k in k_dims:
        if k > min(n_clean, D):
            continue
        # Project using top-k PCs
        V_k = Vt[:k, :]  # (k, D)
        clean_proj = centered @ V_k.T  # (n_clean, k)
        centroid_proj = clean_proj.mean(axis=0)

        aurocs = {}
        for c in corruptions:
            corrupt_centered = np.array(corrupt_embs[c]) - centroid_full
            corrupt_proj = corrupt_centered @ V_k.T
            id_scores = [cosine_dist(e, centroid_proj) for e in clean_proj]
            ood_scores = [cosine_dist(e, centroid_proj) for e in corrupt_proj]
            aurocs[c] = compute_auroc(id_scores, ood_scores)

        results["pca_projection"][k] = {
            "aurocs": aurocs,
            "cumvar": float(cumulative_var[min(k-1, len(cumulative_var)-1)])
        }
        print(f"  k={k}: {aurocs}")

    # 3. Top-k Truncation (just take first k dims)
    print("\n=== Top-k Truncation ===")
    for k in k_dims:
        if k > D:
            continue
        clean_trunc = clean_embs[:, :k]
        centroid_trunc = clean_trunc.mean(axis=0)

        aurocs = {}
        for c in corruptions:
            corrupt_trunc = np.array(corrupt_embs[c])[:, :k]
            id_scores = [cosine_dist(e, centroid_trunc) for e in clean_trunc]
            ood_scores = [cosine_dist(e, centroid_trunc) for e in corrupt_trunc]
            aurocs[c] = compute_auroc(id_scores, ood_scores)

        results["top_k_truncation"][k] = {"aurocs": aurocs}
        print(f"  k={k}: {aurocs}")

    # 4. Centroid stability under compression
    print("\n=== Centroid Stability ===")
    for k in [4, 16, 64, 256, 1024]:
        if k > D:
            continue
        # Random projection stability: how much does centroid vary across random seeds?
        centroids = []
        for seed in range(10):
            R = np.random.RandomState(seed).randn(D, k).astype(np.float64) / np.sqrt(k)
            proj = clean_embs @ R
            centroids.append(proj.mean(axis=0))

        # Pairwise cosine distances between centroids
        dists = []
        for i in range(len(centroids)):
            for j in range(i+1, len(centroids)):
                dists.append(cosine_dist(centroids[i], centroids[j]))

        results["centroid_stability"][k] = {
            "mean_dist": float(np.mean(dists)),
            "max_dist": float(np.max(dists)),
            "min_dist": float(np.min(dists))
        }
        print(f"  k={k}: mean_dist={np.mean(dists):.8f}, max={np.max(dists):.8f}")

    # 5. Latency comparison
    print("\n=== Latency Comparison ===")
    n_trials = 1000
    emb_test = clean_embs[0]

    # Full dim
    t0 = time.time()
    for _ in range(n_trials):
        cosine_dist(emb_test, centroid_full)
    full_lat = (time.time() - t0) / n_trials

    for k in [4, 16, 64, 256, 1024]:
        if k > D:
            continue
        R = np.random.RandomState(999).randn(D, k).astype(np.float64) / np.sqrt(k)
        proj_emb = emb_test @ R
        proj_cent = centroid_full @ R

        # Projection + cosine
        t0 = time.time()
        for _ in range(n_trials):
            p = emb_test @ R
            cosine_dist(p, proj_cent)
        proj_lat = (time.time() - t0) / n_trials

        # Just cosine in reduced space (pre-projected)
        t0 = time.time()
        for _ in range(n_trials):
            cosine_dist(proj_emb, proj_cent)
        cos_lat = (time.time() - t0) / n_trials

        results["latency"][k] = {
            "projection_plus_cosine_us": float(proj_lat * 1e6),
            "cosine_only_us": float(cos_lat * 1e6),
            "full_dim_cosine_us": float(full_lat * 1e6),
            "speedup_cosine": float(full_lat / cos_lat) if cos_lat > 0 else 0
        }
        print(f"  k={k}: proj+cos={proj_lat*1e6:.1f}μs, cos_only={cos_lat*1e6:.1f}μs, "
              f"full={full_lat*1e6:.1f}μs, speedup={full_lat/cos_lat:.1f}x")

    # 6. Minimum k for perfect detection per corruption
    print("\n=== Minimum k for AUROC=1.0 ===")
    min_k = {}
    for c in corruptions:
        found = False
        for k in [2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024]:
            if k > D:
                break
            R = np.random.RandomState(999).randn(D, k).astype(np.float64) / np.sqrt(k)
            clean_proj = clean_embs @ R
            centroid_proj = clean_proj.mean(axis=0)
            corrupt_proj = np.array(corrupt_embs[c]) @ R
            id_scores = [cosine_dist(e, centroid_proj) for e in clean_proj]
            ood_scores = [cosine_dist(e, centroid_proj) for e in corrupt_proj]
            auroc = compute_auroc(id_scores, ood_scores)
            if auroc >= 1.0:
                min_k[c] = k
                found = True
                break
        if not found:
            min_k[c] = -1
        print(f"  {c}: min_k={min_k[c]}")

    results["min_k_perfect"] = min_k

    # Save
    out_path = "/workspace/Vizuara-VLA-Research/experiments/dimensionality_reduction_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
