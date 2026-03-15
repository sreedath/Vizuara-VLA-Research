#!/usr/bin/env python3
"""Experiment 440: Mahalanobis Distance vs Cosine Distance Detection

Compares Mahalanobis distance (accounts for covariance structure) against
cosine distance for OOD detection. Mahalanobis distance is the gold standard
in OOD detection literature (Lee et al. 2018) — does it outperform our
simpler cosine distance approach on VLA embeddings?

Tests:
1. Mahalanobis vs cosine AUROC comparison across corruptions
2. Effect of covariance regularization strength
3. Per-dimension variance analysis (which dims are informative?)
4. PCA-reduced detection (can we detect in lower dimensions?)
5. Euclidean vs cosine vs Mahalanobis head-to-head
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

def euclidean_dist(a, b):
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))

def mahalanobis_dist(x, mean, cov_inv):
    diff = np.asarray(x, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
    return float(np.sqrt(np.clip(diff @ cov_inv @ diff, 0, None)))

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

    seeds = [42, 123, 456, 789, 999, 1111, 2222, 3333]
    scenes = [Image.fromarray(np.random.RandomState(s).randint(0, 255, (224, 224, 3), dtype=np.uint8)) for s in seeds]

    print("Extracting clean embeddings...")
    clean_embs = [extract_hidden(model, processor, s, prompt) for s in scenes]
    centroid = np.mean(clean_embs, axis=0)
    emb_dim = len(centroid)
    print(f"  Embedding dimension: {emb_dim}")

    results = {"n_scenes": len(scenes), "emb_dim": emb_dim}

    # Compute covariance matrix from clean embeddings
    emb_matrix = np.array(clean_embs, dtype=np.float64)
    centered = emb_matrix - centroid
    cov_matrix = (centered.T @ centered) / max(len(clean_embs) - 1, 1)

    # === Test 1: Mahalanobis vs Cosine vs Euclidean ===
    print("\n=== Distance Metric Comparison ===")
    metric_results = {}

    # Regularized covariance inverse for different strengths
    reg_strengths = [0.001, 0.01, 0.1, 1.0]

    for reg in reg_strengths:
        cov_reg = cov_matrix + reg * np.eye(emb_dim)
        try:
            cov_inv = np.linalg.inv(cov_reg)
        except np.linalg.LinAlgError:
            print(f"  reg={reg}: covariance inversion failed, skipping")
            continue

        # Compute distances for clean
        clean_cosine = [cosine_dist(e, centroid) for e in clean_embs]
        clean_euclidean = [euclidean_dist(e, centroid) for e in clean_embs]
        clean_mahal = [mahalanobis_dist(e, centroid, cov_inv) for e in clean_embs]

        per_corruption = {}
        for c in corruptions:
            ood_cosine = []
            ood_euclidean = []
            ood_mahal = []
            for s in scenes:
                emb = extract_hidden(model, processor, apply_corruption(s, c), prompt)
                ood_cosine.append(cosine_dist(emb, centroid))
                ood_euclidean.append(euclidean_dist(emb, centroid))
                ood_mahal.append(mahalanobis_dist(emb, centroid, cov_inv))

            auroc_cosine = float(compute_auroc(clean_cosine, ood_cosine))
            auroc_euclidean = float(compute_auroc(clean_euclidean, ood_euclidean))
            auroc_mahal = float(compute_auroc(clean_mahal, ood_mahal))

            per_corruption[c] = {
                "cosine_auroc": auroc_cosine,
                "euclidean_auroc": auroc_euclidean,
                "mahalanobis_auroc": auroc_mahal,
                "mean_cosine_ood": float(np.mean(ood_cosine)),
                "mean_euclidean_ood": float(np.mean(ood_euclidean)),
                "mean_mahal_ood": float(np.mean(ood_mahal)),
            }
            print(f"  reg={reg}, {c}: cos={auroc_cosine:.4f}, euc={auroc_euclidean:.4f}, mah={auroc_mahal:.4f}")

        metric_results[str(reg)] = per_corruption
    results["metric_comparison"] = metric_results

    # === Test 2: Per-dimension variance analysis ===
    print("\n=== Per-Dimension Variance Analysis ===")
    dim_variances = np.var(emb_matrix, axis=0)
    sorted_dims = np.argsort(dim_variances)[::-1]

    # Top and bottom variance dimensions
    top_var_dims = sorted_dims[:20].tolist()
    bot_var_dims = sorted_dims[-20:].tolist()

    # Check if high-variance dims are more discriminative
    fog_embs = [extract_hidden(model, processor, apply_corruption(s, 'fog'), prompt) for s in scenes[:4]]
    night_embs = [extract_hidden(model, processor, apply_corruption(s, 'night'), prompt) for s in scenes[:4]]

    fog_mean_shift = np.mean(np.array(fog_embs, dtype=np.float64) - centroid, axis=0)
    night_mean_shift = np.mean(np.array(night_embs, dtype=np.float64) - centroid, axis=0)

    # Correlation between variance and shift magnitude
    fog_shift_mag = np.abs(fog_mean_shift)
    night_shift_mag = np.abs(night_mean_shift)

    var_fog_corr = float(np.corrcoef(dim_variances, fog_shift_mag)[0, 1])
    var_night_corr = float(np.corrcoef(dim_variances, night_shift_mag)[0, 1])

    results["dimension_analysis"] = {
        "total_variance": float(np.sum(dim_variances)),
        "mean_variance": float(np.mean(dim_variances)),
        "max_variance_dim": int(sorted_dims[0]),
        "max_variance": float(dim_variances[sorted_dims[0]]),
        "min_variance_dim": int(sorted_dims[-1]),
        "min_variance": float(dim_variances[sorted_dims[-1]]),
        "top20_var_dims": top_var_dims,
        "variance_fog_shift_correlation": var_fog_corr,
        "variance_night_shift_correlation": var_night_corr,
        "pct_variance_top10": float(np.sum(dim_variances[sorted_dims[:10]]) / np.sum(dim_variances) * 100),
        "pct_variance_top50": float(np.sum(dim_variances[sorted_dims[:50]]) / np.sum(dim_variances) * 100),
    }
    print(f"  Total variance: {np.sum(dim_variances):.4f}")
    print(f"  Top 10 dims hold: {np.sum(dim_variances[sorted_dims[:10]]) / np.sum(dim_variances) * 100:.1f}% variance")
    print(f"  Variance↔fog_shift correlation: {var_fog_corr:.4f}")
    print(f"  Variance↔night_shift correlation: {var_night_corr:.4f}")

    # === Test 3: PCA-reduced detection ===
    print("\n=== PCA-Reduced Detection ===")
    # SVD on centered embeddings
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    explained_var = (S ** 2) / np.sum(S ** 2)
    cumulative_var = np.cumsum(explained_var)

    pca_results = {}
    for n_comp in [2, 5, 10, 25, 50, 100, 256, 512]:
        if n_comp > min(emb_dim, len(clean_embs)):
            continue
        # Project to PCA space
        proj_matrix = Vt[:n_comp]  # (n_comp, emb_dim)
        clean_proj = [proj_matrix @ (e - centroid) for e in clean_embs]
        clean_proj_centroid = np.mean(clean_proj, axis=0)

        clean_dists_pca = [euclidean_dist(p, clean_proj_centroid) for p in clean_proj]

        per_corr_pca = {}
        for c in corruptions:
            ood_dists_pca = []
            for s in scenes:
                emb = extract_hidden(model, processor, apply_corruption(s, c), prompt)
                proj = proj_matrix @ (emb - centroid)
                ood_dists_pca.append(euclidean_dist(proj, clean_proj_centroid))
            auroc_pca = float(compute_auroc(clean_dists_pca, ood_dists_pca))
            per_corr_pca[c] = auroc_pca

        var_explained = float(cumulative_var[n_comp - 1]) if n_comp <= len(cumulative_var) else 1.0
        pca_results[str(n_comp)] = {
            "auroc_per_corruption": per_corr_pca,
            "mean_auroc": float(np.mean(list(per_corr_pca.values()))),
            "variance_explained": var_explained,
        }
        print(f"  n_comp={n_comp}: mean_auroc={np.mean(list(per_corr_pca.values())):.4f}, var_explained={var_explained:.4f}")
    results["pca_detection"] = pca_results

    # === Test 4: Singular value spectrum ===
    print("\n=== Singular Value Spectrum ===")
    results["svd_spectrum"] = {
        "singular_values": S.tolist(),
        "explained_variance_ratio": explained_var.tolist(),
        "cumulative_variance": cumulative_var.tolist(),
        "effective_rank": int(np.sum(S > S[0] * 0.01)),
        "top1_pct": float(explained_var[0] * 100),
    }
    print(f"  Effective rank (1% threshold): {np.sum(S > S[0] * 0.01)}")
    print(f"  Top singular value explains: {explained_var[0]*100:.1f}%")
    print(f"  Top 3 explain: {cumulative_var[2]*100:.1f}%")

    # === Test 5: Distance metric sensitivity at low severity ===
    print("\n=== Low-Severity Detection Comparison ===")
    # Use best regularization from Test 1
    best_reg = 0.01
    cov_inv_best = np.linalg.inv(cov_matrix + best_reg * np.eye(emb_dim))
    clean_cosine = [cosine_dist(e, centroid) for e in clean_embs]
    clean_euclidean = [euclidean_dist(e, centroid) for e in clean_embs]
    clean_mahal = [mahalanobis_dist(e, centroid, cov_inv_best) for e in clean_embs]

    severity_results = {}
    for sev in [0.1, 0.25, 0.5, 0.75, 1.0]:
        per_corr_sev = {}
        for c in corruptions:
            ood_cos = []
            ood_euc = []
            ood_mah = []
            for s in scenes:
                emb = extract_hidden(model, processor, apply_corruption(s, c, severity=sev), prompt)
                ood_cos.append(cosine_dist(emb, centroid))
                ood_euc.append(euclidean_dist(emb, centroid))
                ood_mah.append(mahalanobis_dist(emb, centroid, cov_inv_best))

            per_corr_sev[c] = {
                "cosine_auroc": float(compute_auroc(clean_cosine, ood_cos)),
                "euclidean_auroc": float(compute_auroc(clean_euclidean, ood_euc)),
                "mahalanobis_auroc": float(compute_auroc(clean_mahal, ood_mah)),
            }

        mean_cos = float(np.mean([v["cosine_auroc"] for v in per_corr_sev.values()]))
        mean_euc = float(np.mean([v["euclidean_auroc"] for v in per_corr_sev.values()]))
        mean_mah = float(np.mean([v["mahalanobis_auroc"] for v in per_corr_sev.values()]))
        severity_results[str(sev)] = {
            "per_corruption": per_corr_sev,
            "mean_cosine": mean_cos,
            "mean_euclidean": mean_euc,
            "mean_mahalanobis": mean_mah,
        }
        print(f"  sev={sev}: cos={mean_cos:.4f}, euc={mean_euc:.4f}, mah={mean_mah:.4f}")
    results["severity_comparison"] = severity_results

    out_path = "/workspace/Vizuara-VLA-Research/experiments/mahalanobis_detection_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
