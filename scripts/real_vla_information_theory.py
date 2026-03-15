#!/usr/bin/env python3
"""Experiment 348: Information-Theoretic Analysis of OOD Detection

Comprehensive information-theoretic characterization:
1. Entropy of embedding distributions (clean vs corrupted)
2. Mutual information: I(corruption_type; embedding)
3. Channel capacity of the detection signal
4. KL divergence between clean and corrupted distributions
5. Rate-distortion: minimum bits for corruption classification
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

def entropy_gaussian(cov_matrix):
    """Differential entropy of multivariate Gaussian: H = 0.5 * ln((2*pi*e)^d * |Sigma|)"""
    d = cov_matrix.shape[0]
    sign, logdet = np.linalg.slogdet(cov_matrix)
    if sign <= 0:
        return float('nan')
    return 0.5 * (d * np.log(2 * np.pi * np.e) + logdet)

def kl_divergence_gaussian(mu1, cov1, mu2, cov2):
    """KL(N(mu1,cov1) || N(mu2,cov2))"""
    d = len(mu1)
    try:
        cov2_inv = np.linalg.inv(cov2)
    except np.linalg.LinAlgError:
        cov2_inv = np.linalg.pinv(cov2)
    diff = mu2 - mu1
    sign1, logdet1 = np.linalg.slogdet(cov1)
    sign2, logdet2 = np.linalg.slogdet(cov2)
    if sign1 <= 0 or sign2 <= 0:
        return float('nan')
    kl = 0.5 * (np.trace(cov2_inv @ cov1) + diff @ cov2_inv @ diff - d + logdet2 - logdet1)
    return float(kl)

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

    # Generate multi-scene embeddings
    print("Generating embeddings across scenes...")
    seeds = list(range(0, 2000, 100))[:20]

    clean_embs = []
    corrupt_embs = {ct: [] for ct in ctypes}
    distances = {ct: [] for ct in ctypes}

    for seed in seeds:
        rng = np.random.RandomState(seed)
        px = rng.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(px)

        cal = extract_hidden(model, processor, img, prompt)
        clean_embs.append(cal)

        for ct in ctypes:
            corrupted = apply_corruption(img, ct, 0.5)
            emb = extract_hidden(model, processor, corrupted, prompt)
            corrupt_embs[ct].append(emb)
            distances[ct].append(float(cosine_dist(cal, emb)))

        if (seed // 100 + 1) % 5 == 0:
            print(f"  {seed // 100 + 1}/{len(seeds)} scenes done")

    clean_arr = np.array(clean_embs)  # (20, 4096)

    # ========== 1. PCA-reduced entropy analysis ==========
    print("\n=== Entropy Analysis (PCA-reduced) ===")

    # Use PCA to reduce to manageable dimensionality for covariance
    from numpy.linalg import svd

    # Combine all embeddings for PCA
    all_embs = list(clean_embs)
    for ct in ctypes:
        all_embs.extend(corrupt_embs[ct])
    all_arr = np.array(all_embs)
    mean_all = all_arr.mean(axis=0)
    centered = all_arr - mean_all

    U, S, Vt = svd(centered, full_matrices=False)

    # Project to top-k dimensions
    entropy_results = {}
    for k in [3, 5, 10, 20]:
        proj = Vt[:k]  # (k, 4096)

        clean_proj = (clean_arr - mean_all) @ proj.T  # (20, k)
        clean_mean = clean_proj.mean(axis=0)
        clean_cov = np.cov(clean_proj.T) + np.eye(k) * 1e-10

        h_clean = entropy_gaussian(clean_cov)

        per_type = {}
        for ct in ctypes:
            corr_arr = np.array(corrupt_embs[ct])
            corr_proj = (corr_arr - mean_all) @ proj.T
            corr_mean = corr_proj.mean(axis=0)
            corr_cov = np.cov(corr_proj.T) + np.eye(k) * 1e-10

            h_corrupt = entropy_gaussian(corr_cov)
            kl = kl_divergence_gaussian(clean_mean, clean_cov, corr_mean, corr_cov)

            per_type[ct] = {
                'entropy': float(h_corrupt),
                'kl_from_clean': float(kl),
            }

        entropy_results[f'k={k}'] = {
            'clean_entropy': float(h_clean),
            'variance_explained': float(np.sum(S[:k]**2) / np.sum(S**2)),
            'per_type': per_type,
        }
        print(f"  k={k}: H(clean)={h_clean:.2f}, var_explained={np.sum(S[:k]**2)/np.sum(S**2):.3f}")
        for ct in ctypes:
            print(f"    {ct}: H={per_type[ct]['entropy']:.2f}, KL={per_type[ct]['kl_from_clean']:.2f}")

    results['entropy'] = entropy_results

    # ========== 2. 1D Distance Distribution Analysis ==========
    print("\n=== 1D Distance Distribution ===")

    # Analyze the cosine distance as a 1D detection channel
    dist_analysis = {}

    # Clean distances (re-embedding same images)
    clean_dists = []
    for seed in seeds:
        rng = np.random.RandomState(seed)
        px = rng.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(px)
        emb = extract_hidden(model, processor, img, prompt)
        clean_dists.append(float(cosine_dist(clean_embs[seeds.index(seed)], emb)))

    for ct in ctypes:
        ood = np.array(distances[ct])
        clean = np.array(clean_dists)

        # Entropy of discretized distance (8 bins)
        all_d = np.concatenate([clean, ood])
        bins = np.linspace(min(all_d) - 1e-10, max(all_d) + 1e-10, 9)

        clean_hist, _ = np.histogram(clean, bins=bins, density=False)
        clean_hist = clean_hist / max(clean_hist.sum(), 1)
        ood_hist, _ = np.histogram(ood, bins=bins, density=False)
        ood_hist = ood_hist / max(ood_hist.sum(), 1)

        # Shannon entropy
        h_clean_1d = -sum(p * np.log2(p) for p in clean_hist if p > 0)
        h_ood_1d = -sum(p * np.log2(p) for p in ood_hist if p > 0)

        # Joint entropy H(distance, label)
        joint = np.array([clean_hist * 0.5, ood_hist * 0.5]).flatten()
        h_joint = -sum(p * np.log2(p) for p in joint if p > 0)

        # Mutual information I(D;L) = H(D) + H(L) - H(D,L)
        marginal = (clean_hist + ood_hist) / 2
        h_d = -sum(p * np.log2(p) for p in marginal if p > 0)
        h_l = 1.0  # H(Bernoulli(0.5)) = 1 bit
        mi = h_d + h_l - h_joint

        # Channel capacity approximation
        # For binary detection with perfect separation: C = 1 bit
        overlap = sum(min(c, o) for c, o in zip(clean_hist, ood_hist))
        separation_score = 1.0 - overlap

        dist_analysis[ct] = {
            'clean_mean': float(np.mean(clean)),
            'clean_std': float(np.std(clean)),
            'ood_mean': float(np.mean(ood)),
            'ood_std': float(np.std(ood)),
            'h_clean_1d': float(h_clean_1d),
            'h_ood_1d': float(h_ood_1d),
            'mutual_information': float(mi),
            'channel_capacity_approx': float(min(mi, 1.0)),
            'histogram_overlap': float(overlap),
            'separation_score': float(separation_score),
        }
        print(f"  {ct}: MI={mi:.4f} bits, overlap={overlap:.4f}, separation={separation_score:.4f}")

    results['distance_channel'] = dist_analysis

    # ========== 3. Multi-class mutual information ==========
    print("\n=== Multi-Class Corruption Classification ===")

    # Can we distinguish WHICH corruption type from the embedding?
    # Project all corrupt embeddings to PCA space and compute classification accuracy
    k = 10
    proj = Vt[:k]

    # Compute pairwise KL between corruption types
    type_means = {}
    type_covs = {}
    for ct in ctypes:
        corr_arr = np.array(corrupt_embs[ct])
        corr_proj = (corr_arr - mean_all) @ proj.T
        type_means[ct] = corr_proj.mean(axis=0)
        type_covs[ct] = np.cov(corr_proj.T) + np.eye(k) * 1e-10

    pairwise_kl = {}
    for i, ct1 in enumerate(ctypes):
        for j, ct2 in enumerate(ctypes):
            if i >= j:
                continue
            kl_12 = kl_divergence_gaussian(type_means[ct1], type_covs[ct1],
                                            type_means[ct2], type_covs[ct2])
            kl_21 = kl_divergence_gaussian(type_means[ct2], type_covs[ct2],
                                            type_means[ct1], type_covs[ct1])
            sym_kl = (kl_12 + kl_21) / 2

            key = f"{ct1}_vs_{ct2}"
            pairwise_kl[key] = {
                'kl_forward': float(kl_12),
                'kl_backward': float(kl_21),
                'symmetric_kl': float(sym_kl),
            }
            print(f"  {key}: sym_KL={sym_kl:.2f}")

    # Nearest centroid classification (leave-one-out)
    correct = 0
    total = 0
    confusion = {ct1: {ct2: 0 for ct2 in ctypes} for ct1 in ctypes}

    for ct_true in ctypes:
        for i in range(len(seeds)):
            emb = corrupt_embs[ct_true][i]
            emb_proj = (emb - mean_all) @ proj.T

            best_ct = None
            best_dist = float('inf')
            for ct in ctypes:
                # Leave-one-out: exclude this sample if same type
                if ct == ct_true:
                    others = [corrupt_embs[ct][j] for j in range(len(seeds)) if j != i]
                else:
                    others = corrupt_embs[ct]

                centroid_proj = np.mean([(e - mean_all) @ proj.T for e in others], axis=0)
                d = np.linalg.norm(emb_proj - centroid_proj)
                if d < best_dist:
                    best_dist = d
                    best_ct = ct

            confusion[ct_true][best_ct] += 1
            if best_ct == ct_true:
                correct += 1
            total += 1

    classification_accuracy = correct / total

    multiclass = {
        'pairwise_kl': pairwise_kl,
        'classification_accuracy': float(classification_accuracy),
        'confusion_matrix': confusion,
        'n_classes': len(ctypes),
        'max_possible_mi': float(np.log2(len(ctypes))),  # log2(4) = 2 bits
    }
    print(f"  Classification accuracy: {classification_accuracy:.3f}")
    print(f"  Max MI for 4 classes: {np.log2(4):.2f} bits")

    results['multiclass'] = multiclass

    # ========== 4. Severity as information channel ==========
    print("\n=== Severity Information Channel ===")

    severity_channel = {}
    for ct in ctypes:
        # How many bits of severity information does the embedding carry?
        sevs = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
        sev_dists = []

        for sev in sevs:
            sev_d = []
            for seed in seeds[:10]:
                rng = np.random.RandomState(seed)
                px = rng.randint(50, 200, (224, 224, 3), dtype=np.uint8)
                img = Image.fromarray(px)
                corrupted = apply_corruption(img, ct, sev)
                emb = extract_hidden(model, processor, corrupted, prompt)
                d = cosine_dist(clean_embs[seeds.index(seed)], emb)
                sev_d.append(float(d))
            sev_dists.append(sev_d)

        # Spearman rank correlation between severity and distance
        all_sevs_flat = []
        all_dists_flat = []
        for i, sev in enumerate(sevs):
            for d in sev_dists[i]:
                all_sevs_flat.append(sev)
                all_dists_flat.append(d)

        # Compute rank correlation
        sev_ranks = np.argsort(np.argsort(all_sevs_flat)).astype(float)
        dist_ranks = np.argsort(np.argsort(all_dists_flat)).astype(float)
        n = len(sev_ranks)
        spearman = 1 - 6 * np.sum((sev_ranks - dist_ranks)**2) / (n * (n**2 - 1))

        # Can we distinguish severity levels? (pairwise t-test equivalent)
        distinguishable_pairs = 0
        total_pairs = 0
        for i in range(len(sevs)):
            for j in range(i+1, len(sevs)):
                d_i = np.array(sev_dists[i])
                d_j = np.array(sev_dists[j])
                # Check if distributions are separated
                if min(d_j) > max(d_i):
                    distinguishable_pairs += 1
                total_pairs += 1

        severity_channel[ct] = {
            'severities': sevs,
            'mean_distances': [float(np.mean(sd)) for sd in sev_dists],
            'spearman_rho': float(spearman),
            'distinguishable_pairs': distinguishable_pairs,
            'total_pairs': total_pairs,
            'severity_bits': float(np.log2(distinguishable_pairs + 1)) if distinguishable_pairs > 0 else 0,
        }
        print(f"  {ct}: rho={spearman:.4f}, {distinguishable_pairs}/{total_pairs} pairs separated, "
              f"~{np.log2(distinguishable_pairs + 1):.1f} severity bits")

    results['severity_channel'] = severity_channel

    # ========== 5. Compression information bounds ==========
    print("\n=== Compression Bounds ===")

    # How many PCA dimensions needed for perfect detection?
    compression = {}
    for k in [1, 2, 3, 5, 10, 20, 50, 100]:
        if k > min(len(all_embs), 4096):
            continue
        proj_k = Vt[:k]

        per_type = {}
        for ct in ctypes:
            clean_proj = (clean_arr - mean_all) @ proj_k.T
            corr_proj = (np.array(corrupt_embs[ct]) - mean_all) @ proj_k.T

            # Compute cosine dist in projected space
            proj_dists = []
            for i in range(len(seeds)):
                c = clean_proj[i]
                o = corr_proj[i]
                d = 1.0 - np.dot(c, o) / (np.linalg.norm(c) * np.linalg.norm(o) + 1e-10)
                proj_dists.append(float(d))

            # AUROC: clean re-embedding is always 0 in projected space
            n_detected = sum(1 for d in proj_dists if d > 0)
            per_type[ct] = {
                'mean_dist': float(np.mean(proj_dists)),
                'min_dist': float(min(proj_dists)),
                'detection_rate': float(n_detected / len(proj_dists)),
            }

        var_explained = float(np.sum(S[:k]**2) / np.sum(S**2))
        compression[f'k={k}'] = {
            'variance_explained': var_explained,
            'per_type': per_type,
            'all_detected': all(
                per_type[ct]['detection_rate'] == 1.0 for ct in ctypes
            ),
        }
        detected_str = "ALL" if compression[f'k={k}']['all_detected'] else "PARTIAL"
        print(f"  k={k}: var={var_explained:.3f}, detection={detected_str}")

    results['compression'] = compression

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/information_theory_{ts}.json"
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
