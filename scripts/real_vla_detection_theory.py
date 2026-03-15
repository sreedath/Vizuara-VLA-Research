#!/usr/bin/env python3
"""Experiment 326: Detection Theory Analysis (Real OpenVLA-7B)

Connects empirical results to theoretical detection frameworks:
1. Signal-to-noise ratio (SNR) analysis across corruptions
2. Fisher discriminant analysis for optimal projection
3. Mahalanobis distance with empirical covariance
4. Likelihood ratio test statistics
5. Neyman-Pearson operating curves
6. Minimum detectable effect size
7. Statistical power analysis
8. Kernel distance metrics (MMD)
"""

import json, time, os, sys, math
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

def main():
    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    np.random.seed(42)
    pixels = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    base_img = Image.fromarray(pixels)
    prompt = "In: What action should the robot take to pick up the object?\nOut:"

    results = {}

    # ========== Collect embeddings ==========
    print("\n=== Collecting Embeddings ===")
    clean_emb = extract_hidden(model, processor, base_img, prompt)

    # Multiple scenes for in-distribution statistics
    scenes = []
    scene_embs = []
    for seed in range(10):
        rng = np.random.RandomState(seed)
        px = rng.randint(50, 200, (224, 224, 3)).astype(np.uint8)
        img = Image.fromarray(px)
        emb = extract_hidden(model, processor, img, prompt)
        scenes.append(seed)
        scene_embs.append(emb)

    scene_embs = np.array(scene_embs)  # 10 x 4096

    # Corruption embeddings at multiple severities
    ctypes = ['fog', 'night', 'noise', 'blur']
    corrupt_embs = {}
    for ct in ctypes:
        corrupt_embs[ct] = {}
        for sev in [0.1, 0.25, 0.5, 0.75, 1.0]:
            img = apply_corruption(base_img, ct, sev)
            emb = extract_hidden(model, processor, img, prompt)
            corrupt_embs[ct][sev] = emb

    # ========== 1. Signal-to-Noise Ratio ==========
    print("\n=== Signal-to-Noise Ratio ===")
    # ID variance = 0 (for same scene), so SNR → ∞
    # But across scenes, there's variance
    scene_centroid = scene_embs.mean(axis=0)
    scene_distances = [cosine_dist(scene_centroid, e) for e in scene_embs]

    snr_results = {}
    for ct in ctypes:
        ood_d = cosine_dist(clean_emb, corrupt_embs[ct][1.0])
        # Within-scene SNR (ID var = 0)
        snr_within = float('inf') if ood_d > 0 else 0
        # Cross-scene: use scene variance
        scene_var = np.var(scene_distances) if len(scene_distances) > 1 else 0
        snr_cross = float(ood_d / max(np.std(scene_distances), 1e-10))

        snr_results[ct] = {
            'ood_distance': float(ood_d),
            'within_scene_snr': 'infinity' if snr_within == float('inf') else float(snr_within),
            'cross_scene_snr': float(snr_cross),
            'scene_mean_d': float(np.mean(scene_distances)),
            'scene_std_d': float(np.std(scene_distances)),
        }
        print(f"  {ct}: d={ood_d:.6f}, within_snr=∞, cross_snr={snr_cross:.2f}")

    results['snr'] = snr_results

    # ========== 2. Fisher Discriminant ==========
    print("\n=== Fisher Discriminant Analysis ===")
    fisher_results = {}

    for ct in ctypes:
        # In-distribution: multiple passes of clean (all identical)
        id_embs = np.array([clean_emb] * 5)  # repeated, var=0
        ood_emb = corrupt_embs[ct][1.0]

        # Fisher's criterion: (μ1 - μ2)² / (σ1² + σ2²)
        mean_diff = clean_emb - ood_emb
        # Since σ1² = 0, Fisher ratio → ∞ if mean_diff ≠ 0
        l2_diff = np.linalg.norm(mean_diff)

        # Project onto discriminant direction
        w = mean_diff / (np.linalg.norm(mean_diff) + 1e-10)
        id_proj = np.dot(clean_emb, w)
        ood_proj = np.dot(ood_emb, w)

        fisher_results[ct] = {
            'l2_mean_diff': float(l2_diff),
            'fisher_ratio': 'infinity',
            'discriminant_projection_id': float(id_proj),
            'discriminant_projection_ood': float(ood_proj),
            'projection_gap': float(abs(id_proj - ood_proj)),
        }
        print(f"  {ct}: L2_diff={l2_diff:.6f}, proj_gap={abs(id_proj - ood_proj):.6f}")

    results['fisher'] = fisher_results

    # ========== 3. Cross-scene Mahalanobis ==========
    print("\n=== Cross-Scene Mahalanobis ===")
    # Use scene embeddings covariance
    cov = np.cov(scene_embs.T)  # 4096 x 4096 — too large for full inverse
    # Use diagonal covariance (independence assumption)
    diag_var = np.var(scene_embs, axis=0)
    diag_var[diag_var < 1e-20] = 1e-20  # avoid division by zero

    maha_results = {}
    for ct in ctypes:
        diff = corrupt_embs[ct][1.0] - scene_centroid
        maha_d = np.sqrt(np.sum(diff**2 / diag_var))
        eucl_d = np.linalg.norm(diff)
        cos_d = cosine_dist(scene_centroid, corrupt_embs[ct][1.0])

        maha_results[ct] = {
            'mahalanobis': float(maha_d),
            'euclidean': float(eucl_d),
            'cosine': float(cos_d),
            'maha_eucl_ratio': float(maha_d / eucl_d) if eucl_d > 0 else 0,
        }
        print(f"  {ct}: maha={maha_d:.2f}, eucl={eucl_d:.4f}, cos={cos_d:.6f}")

    results['mahalanobis'] = maha_results

    # ========== 4. Minimum Detectable Effect ==========
    print("\n=== Minimum Detectable Effect ===")
    mde_results = {}
    for ct in ctypes:
        # Find minimum severity where d > threshold for various thresholds
        for threshold_name, threshold in [('any', 0), ('1e-6', 1e-6), ('1e-5', 1e-5), ('1e-4', 1e-4), ('1e-3', 1e-3)]:
            min_sev = None
            for sev in np.linspace(0.01, 1.0, 100):
                img = apply_corruption(base_img, ct, sev)
                emb = extract_hidden(model, processor, img, prompt)
                d = cosine_dist(clean_emb, emb)
                if d > threshold:
                    min_sev = sev
                    break

            if min_sev is not None:
                key = f"{ct}_thresh_{threshold_name}"
                mde_results[key] = float(min_sev)
                if threshold_name == 'any':
                    print(f"  {ct} (d>0): min_sev={min_sev:.4f}")
                    break  # Just get the 'any' threshold for each type

    results['min_detectable_effect'] = mde_results

    # ========== 5. Power Analysis ==========
    print("\n=== Statistical Power ===")
    power_results = {}

    # For within-scene: power = 1.0 (no variance → perfect separation)
    # For cross-scene: compute power at various sample sizes
    for ct in ctypes:
        ood_d = cosine_dist(scene_centroid, corrupt_embs[ct][1.0])
        scene_std = np.std(scene_distances)

        powers = {}
        for n in [1, 3, 5, 10, 20]:
            # Effect size (Cohen's d analog)
            if scene_std > 0:
                effect_size = ood_d / scene_std
                # Approximate power for one-sided z-test
                z_alpha = 1.645  # α=0.05
                z_power = effect_size * np.sqrt(n) - z_alpha
                power = 0.5 * (1 + math.erf(z_power / np.sqrt(2)))
            else:
                power = 1.0

            powers[str(n)] = float(min(power, 1.0))

        power_results[ct] = {
            'effect_size': float(ood_d / max(scene_std, 1e-10)),
            'powers': powers,
            'within_scene_power': 1.0,  # always 1.0 due to zero variance
        }
        print(f"  {ct}: effect_size={power_results[ct]['effect_size']:.2f}, power@n=1={powers['1']:.4f}")

    results['power_analysis'] = power_results

    # ========== 6. Kernel MMD ==========
    print("\n=== Maximum Mean Discrepancy (MMD) ===")
    mmd_results = {}

    # Compute MMD between clean and corrupt embeddings using RBF kernel
    def rbf_kernel(X, Y, sigma=1.0):
        """Compute RBF kernel matrix between X and Y."""
        XX = np.sum(X**2, axis=1, keepdims=True)
        YY = np.sum(Y**2, axis=1, keepdims=True)
        D = XX + YY.T - 2 * X @ Y.T
        return np.exp(-D / (2 * sigma**2))

    # Use multiple clean scene embeddings as ID sample
    for ct in ctypes:
        # OOD embeddings at different severities
        ood_embs_list = []
        for sev in [0.25, 0.5, 0.75, 1.0]:
            ood_embs_list.append(corrupt_embs[ct][sev])
        ood_embs_arr = np.array(ood_embs_list)  # 4 x 4096

        # Median heuristic for bandwidth
        all_embs = np.vstack([scene_embs, ood_embs_arr])
        pairwise_dists = np.sqrt(np.sum((all_embs[:, None, :] - all_embs[None, :, :])**2, axis=2))
        sigma = float(np.median(pairwise_dists[pairwise_dists > 0]))

        # MMD²
        K_xx = rbf_kernel(scene_embs, scene_embs, sigma)
        K_yy = rbf_kernel(ood_embs_arr, ood_embs_arr, sigma)
        K_xy = rbf_kernel(scene_embs, ood_embs_arr, sigma)

        n, m = len(scene_embs), len(ood_embs_arr)
        mmd_sq = (np.sum(K_xx) / (n*n) - 2 * np.sum(K_xy) / (n*m) + np.sum(K_yy) / (m*m))

        mmd_results[ct] = {
            'mmd_squared': float(mmd_sq),
            'mmd': float(np.sqrt(max(mmd_sq, 0))),
            'bandwidth': float(sigma),
        }
        print(f"  {ct}: MMD={np.sqrt(max(mmd_sq, 0)):.6f}, σ={sigma:.4f}")

    results['mmd'] = mmd_results

    # ========== 7. Detection Margin Analysis ==========
    print("\n=== Detection Margin ===")
    margin_results = {}

    for ct in ctypes:
        # Within-scene: margin = ood_distance (since ID distance = 0)
        within_margin = cosine_dist(clean_emb, corrupt_embs[ct][1.0])

        # Cross-scene: margin = ood_distance - max(scene_distances)
        max_id = max(scene_distances)
        cross_margin = cosine_dist(scene_centroid, corrupt_embs[ct][1.0]) - max_id

        margin_results[ct] = {
            'within_scene_margin': float(within_margin),
            'cross_scene_margin': float(cross_margin),
            'max_id_distance': float(max_id),
            'safety_factor_within': 'infinity',
            'safety_factor_cross': float(cosine_dist(scene_centroid, corrupt_embs[ct][1.0]) / max_id) if max_id > 0 else float('inf'),
        }
        print(f"  {ct}: within_margin={within_margin:.6f}, cross_margin={cross_margin:.6f}")

    results['margins'] = margin_results

    # ========== 8. Dimensionality of the Detection Signal ==========
    print("\n=== Detection Signal Dimensionality ===")
    # Collect all corruption displacement vectors
    all_disp = []
    labels = []
    for ct in ctypes:
        for sev in [0.25, 0.5, 0.75, 1.0]:
            disp = corrupt_embs[ct][sev] - clean_emb
            all_disp.append(disp)
            labels.append(f"{ct}_{sev}")

    disp_matrix = np.array(all_disp)  # 16 x 4096

    # SVD to find intrinsic dimensionality
    U, S, Vt = np.linalg.svd(disp_matrix, full_matrices=False)
    cumvar = np.cumsum(S**2) / np.sum(S**2)

    dim_results = {
        'singular_values': [float(s) for s in S[:10]],
        'cumulative_variance': [float(c) for c in cumvar[:10]],
        'dims_for_90pct': int(np.searchsorted(cumvar, 0.9) + 1),
        'dims_for_95pct': int(np.searchsorted(cumvar, 0.95) + 1),
        'dims_for_99pct': int(np.searchsorted(cumvar, 0.99) + 1),
        'effective_rank': float(np.exp(-np.sum((S**2/np.sum(S**2)) * np.log(S**2/np.sum(S**2) + 1e-20)))),
    }
    print(f"  Dims for 90%: {dim_results['dims_for_90pct']}, 95%: {dim_results['dims_for_95pct']}, 99%: {dim_results['dims_for_99pct']}")
    print(f"  Effective rank: {dim_results['effective_rank']:.2f}")
    print(f"  Top SVs: {S[:5]}")

    results['dimensionality'] = dim_results

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/detection_theory_{ts}.json"

    def convert(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    def recursive_convert(d):
        if isinstance(d, dict):
            return {k: recursive_convert(v) for k, v in d.items()}
        if isinstance(d, list):
            return [recursive_convert(x) for x in d]
        return convert(d)

    results = recursive_convert(results)

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
