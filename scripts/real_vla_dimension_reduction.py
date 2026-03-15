#!/usr/bin/env python3
"""Experiment 359: Dimensionality Reduction for Real-Time Detection

Can we compress 4096-D embeddings for faster inference?
1. Random projection to various dimensions
2. PCA projection quality vs dimension
3. Sparse random projection (memory-efficient)
4. Detection accuracy vs compression ratio
5. Reconstruction error analysis
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

def cosine_dist(a, b):
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return 1.0 - dot / (na * nb)

def compute_auroc(id_scores, ood_scores):
    id_s = np.asarray(id_scores)
    ood_s = np.asarray(ood_scores)
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
    results = {}
    ctypes = ['fog', 'night', 'noise', 'blur']

    # Generate embeddings
    print("Generating embeddings...")
    seeds = list(range(0, 2000, 100))[:20]
    clean_embs = {}
    corrupt_embs = {ct: {} for ct in ctypes}

    for seed in seeds:
        rng = np.random.RandomState(seed)
        px = rng.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(px)
        clean_embs[seed] = extract_hidden(model, processor, img, prompt)

        for ct in ctypes:
            corrupted = apply_corruption(img, ct, 0.5)
            corrupt_embs[ct][seed] = extract_hidden(model, processor, corrupted, prompt)

    d_orig = len(clean_embs[seeds[0]])
    print(f"  Original dimension: {d_orig}")

    # Compute full-dim baselines
    centroid = np.mean([clean_embs[s] for s in seeds], axis=0)
    clean_dists_full = [float(cosine_dist(centroid, clean_embs[s])) for s in seeds]

    baseline_aurocs = {}
    for ct in ctypes:
        ood_dists = [float(cosine_dist(centroid, corrupt_embs[ct][s])) for s in seeds]
        baseline_aurocs[ct] = float(compute_auroc(clean_dists_full, ood_dists))
    print(f"  Full-dim AUROCs: " + str(baseline_aurocs))

    # ========== 1. PCA Projection ==========
    print("\n=== PCA Projection ===")

    all_clean = np.array([clean_embs[s] for s in seeds])
    mean_vec = all_clean.mean(axis=0)
    centered = all_clean - mean_vec

    # Compute PCA via SVD
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    explained_var = S ** 2
    total_var = explained_var.sum()
    cumvar = np.cumsum(explained_var) / total_var

    pca_results = {}
    for d in [1, 2, 3, 4, 5, 10, 20, 50, 100, 500]:
        if d > len(S):
            continue
        # Project
        proj_matrix = Vt[:d]  # d x 4096

        proj_centroid = proj_matrix @ centroid
        proj_clean = [proj_matrix @ clean_embs[s] for s in seeds]
        clean_dists_proj = [float(cosine_dist(proj_centroid, pc)) for pc in proj_clean]

        per_type = {}
        for ct in ctypes:
            proj_corrupt = [proj_matrix @ corrupt_embs[ct][s] for s in seeds]
            ood_dists_proj = [float(cosine_dist(proj_centroid, pc)) for pc in proj_corrupt]
            auroc = float(compute_auroc(clean_dists_proj, ood_dists_proj))
            per_type[ct] = auroc

        var_explained = float(cumvar[d-1]) if d <= len(cumvar) else 1.0
        compression = d_orig / d

        pca_results[str(d)] = {
            'per_type': per_type,
            'var_explained': var_explained,
            'compression_ratio': float(compression),
            'mean_auroc': float(np.mean(list(per_type.values()))),
        }
        auroc_str = ', '.join(ct + '=' + format(per_type[ct], '.3f') for ct in ctypes)
        print(f"  d={d}: {auroc_str} (var={var_explained:.4f}, compression={compression:.0f}x)")

    results['pca'] = pca_results

    # ========== 2. Random Projection ==========
    print("\n=== Random Projection ===")

    rp_results = {}
    for d in [2, 5, 10, 20, 50, 100, 500]:
        # Multiple random projection matrices
        trial_aurocs = {ct: [] for ct in ctypes}

        for trial in range(5):
            rng = np.random.RandomState(42 + trial)
            # Gaussian random projection
            proj_matrix = rng.randn(d, d_orig) / np.sqrt(d)

            proj_centroid = proj_matrix @ centroid
            proj_clean = [proj_matrix @ clean_embs[s] for s in seeds]
            clean_dists_proj = [float(cosine_dist(proj_centroid, pc)) for pc in proj_clean]

            for ct in ctypes:
                proj_corrupt = [proj_matrix @ corrupt_embs[ct][s] for s in seeds]
                ood_dists_proj = [float(cosine_dist(proj_centroid, pc)) for pc in proj_corrupt]
                auroc = float(compute_auroc(clean_dists_proj, ood_dists_proj))
                trial_aurocs[ct].append(auroc)

        per_type = {ct: float(np.mean(trial_aurocs[ct])) for ct in ctypes}
        per_type_std = {ct: float(np.std(trial_aurocs[ct])) for ct in ctypes}

        rp_results[str(d)] = {
            'per_type_mean': per_type,
            'per_type_std': per_type_std,
            'mean_auroc': float(np.mean(list(per_type.values()))),
            'compression_ratio': float(d_orig / d),
        }
        auroc_str = ', '.join(ct + '=' + format(per_type[ct], '.3f') for ct in ctypes)
        print(f"  d={d}: {auroc_str} (compression={d_orig/d:.0f}x)")

    results['random_projection'] = rp_results

    # ========== 3. Sparse Random Projection ==========
    print("\n=== Sparse Random Projection ===")

    sparse_results = {}
    for d in [10, 50, 100, 500]:
        trial_aurocs = {ct: [] for ct in ctypes}
        for trial in range(5):
            rng = np.random.RandomState(42 + trial)
            # Sparse: sqrt(3)*{-1, 0, 0, 0, 0, 1} (Achlioptas)
            s = np.sqrt(3)
            choices = rng.choice([-s, 0, 0, 0, 0, s], size=(d, d_orig))
            proj_matrix = choices / np.sqrt(d)

            proj_centroid = proj_matrix @ centroid
            proj_clean = [proj_matrix @ clean_embs[seed] for seed in seeds]
            clean_dists_proj = [float(cosine_dist(proj_centroid, pc)) for pc in proj_clean]

            for ct in ctypes:
                proj_corrupt = [proj_matrix @ corrupt_embs[ct][seed] for seed in seeds]
                ood_dists_proj = [float(cosine_dist(proj_centroid, pc)) for pc in proj_corrupt]
                auroc = float(compute_auroc(clean_dists_proj, ood_dists_proj))
                trial_aurocs[ct].append(auroc)

        per_type = {ct: float(np.mean(trial_aurocs[ct])) for ct in ctypes}
        sparse_results[str(d)] = {
            'per_type_mean': per_type,
            'mean_auroc': float(np.mean(list(per_type.values()))),
            'compression_ratio': float(d_orig / d),
            'sparsity': 4.0 / 6.0,  # fraction of zeros
        }
        auroc_str = ', '.join(ct + '=' + format(per_type[ct], '.3f') for ct in ctypes)
        print(f"  d={d}: {auroc_str} (compression={d_orig/d:.0f}x, sparsity=67%)")

    results['sparse_projection'] = sparse_results

    # ========== 4. Distance Preservation Analysis ==========
    print("\n=== Distance Preservation ===")

    preservation = {}
    for d in [5, 10, 50, 100]:
        rng = np.random.RandomState(42)
        proj_matrix = rng.randn(d, d_orig) / np.sqrt(d)

        # Compare original vs projected distances
        orig_dists = []
        proj_dists = []
        for i in range(min(10, len(seeds))):
            for ct in ctypes:
                od = cosine_dist(clean_embs[seeds[i]], corrupt_embs[ct][seeds[i]])
                pe = proj_matrix @ clean_embs[seeds[i]]
                pc = proj_matrix @ corrupt_embs[ct][seeds[i]]
                pd = cosine_dist(pe, pc)
                orig_dists.append(od)
                proj_dists.append(pd)

        orig_arr = np.array(orig_dists)
        proj_arr = np.array(proj_dists)
        ratio = proj_arr / (orig_arr + 1e-10)

        preservation[str(d)] = {
            'mean_ratio': float(np.mean(ratio)),
            'std_ratio': float(np.std(ratio)),
            'min_ratio': float(min(ratio)),
            'max_ratio': float(max(ratio)),
            'correlation': float(np.corrcoef(orig_arr, proj_arr)[0, 1]),
        }
        print(f"  d={d}: ratio={np.mean(ratio):.3f}±{np.std(ratio):.3f}, corr={np.corrcoef(orig_arr, proj_arr)[0,1]:.4f}")

    results['distance_preservation'] = preservation

    # ========== 5. Minimum Dimensions for Perfect Detection ==========
    print("\n=== Minimum Dimensions for AUROC=1.0 ===")

    min_dims = {}
    for ct in ctypes:
        found_d = None
        for d in [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50]:
            if d > len(S):
                continue
            proj_matrix = Vt[:d]
            proj_centroid = proj_matrix @ centroid
            proj_clean = [proj_matrix @ clean_embs[s] for s in seeds]
            clean_dists_proj = [float(cosine_dist(proj_centroid, pc)) for pc in proj_clean]
            proj_corrupt = [proj_matrix @ corrupt_embs[ct][s] for s in seeds]
            ood_dists_proj = [float(cosine_dist(proj_centroid, pc)) for pc in proj_corrupt]
            auroc = compute_auroc(clean_dists_proj, ood_dists_proj)
            if auroc >= 1.0 - 1e-10 and found_d is None:
                found_d = d
        min_dims[ct] = found_d
        print(f"  {ct}: min_d={found_d} for AUROC=1.0")

    results['min_dims_perfect'] = min_dims

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/dimension_reduction_{ts}.json"
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
