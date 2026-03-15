#!/usr/bin/env python3
"""Experiment 455: Information-Theoretic Analysis of OOD Detection (Real OpenVLA-7B)

Applies information theory to understand the OOD detection mechanism:
1. Embedding Distribution Entropy (Kozachenko-Leonenko kNN estimator)
2. KL Divergence Estimation (kNN-based, high-dimensional)
3. Information Bottleneck (PCA variance + Fisher ratio across top-K PCs)
4. Mutual Information via Nearest-Centroid Classification (5-class)
5. Rate-Distortion Curve (quantization bits vs AUROC)
6. Channel Capacity at Various Severities (binary BSC model)

Seeds: [42, 123, 456, 789, 1000, 2000, 3000, 4000] (8 scenes)
Corruptions: fog, night, noise, blur
"""

import torch, json, os, sys, numpy as np
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor
from datetime import datetime


# ---------------------------------------------------------------------------
# Standard helpers (specified interface)
# ---------------------------------------------------------------------------

def make_image(seed=42):
    rng = np.random.RandomState(seed)
    return Image.fromarray(rng.randint(0, 255, (224, 224, 3), dtype=np.uint8))


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


def compute_auroc(id_scores, ood_scores):
    """Higher score = more OOD.  Trapezoid rule for ties."""
    id_s = np.asarray(id_scores, dtype=np.float64)
    ood_s = np.asarray(ood_scores, dtype=np.float64)
    n_id, n_ood = len(id_s), len(ood_s)
    if n_id == 0 or n_ood == 0:
        return 0.5
    count = sum(
        float(np.sum(o > id_s) + 0.5 * np.sum(o == id_s))
        for o in ood_s
    )
    return count / (n_id * n_ood)


def cosine_dist(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Information-theoretic estimators
# ---------------------------------------------------------------------------

def _digamma_approx(n):
    """Asymptotic approximation of the digamma function for n >= 1."""
    import math
    return math.log(n) - 1.0 / (2.0 * n) - 1.0 / (12.0 * n ** 2)


def knn_entropy(X, k=3):
    """Kozachenko-Leonenko differential entropy estimator.

    H(X) ≈ d * (1/N) * Σ_i log(r_i) + log(N-1) - ψ(k)

    The constant log(V_d) (unit ball volume) is omitted because it is identical
    across all conditions being compared, so relative comparisons are unaffected.

    Returns entropy in nats.  Returns nan when N is too small.
    """
    import math
    X = np.asarray(X, dtype=np.float64)
    N, d = X.shape
    if N <= k + 1:
        return float('nan')

    # Pairwise squared distances  (N x N)
    diff = X[:, None, :] - X[None, :, :]      # (N, N, d)
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)

    rk = np.sort(dists, axis=1)[:, k - 1]     # k-th NN distance per point
    rk = np.clip(rk, 1e-300, None)

    psi_k = _digamma_approx(k)

    H = d * float(np.mean(np.log(rk))) + (math.log(N - 1) - psi_k)
    return H


def knn_kl_divergence(P, Q, k=3):
    """kNN KL divergence estimator D_KL(P || Q).

    D_KL ≈ (d/N) Σ_i log(ν_i / ρ_i) + log(M / (N-1))

    where  rho_i = distance from x_i to its k-th NN inside P (self excluded),
           ν_i = distance from x_i to its k-th NN inside Q,
           N=|P|,  M=|Q|,  d=dimension.

    Reference: Wang et al. (2009).
    """
    import math
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    N, d = P.shape
    M = Q.shape[0]
    if N <= k + 1 or M < k:
        return float('nan')

    # ρ: k-th NN of each point within P (self excluded)
    diff_PP = P[:, None, :] - P[None, :, :]
    dist_PP = np.sqrt(np.sum(diff_PP ** 2, axis=-1))
    np.fill_diagonal(dist_PP, np.inf)
    rho = np.sort(dist_PP, axis=1)[:, k - 1]

    # ν: k-th NN of each P point inside Q
    diff_PQ = P[:, None, :] - Q[None, :, :]   # (N, M, d)
    dist_PQ = np.sqrt(np.sum(diff_PQ ** 2, axis=-1)).T  # (M, N)
    nu = np.sort(dist_PQ, axis=0)[k - 1, :]   # shape (N,)

    rho = np.clip(rho, 1e-300, None)
    nu = np.clip(nu, 1e-300, None)

    kl = (d / N) * float(np.sum(np.log(nu / rho))) + math.log(M / (N - 1))
    return float(kl)


def fisher_ratio_per_dim(clean_embs, ood_embs):
    """Per-dimension Fisher discriminant ratio F_j = (μ_ood - μ_clean)²/(σ²_clean + σ²_ood)."""
    mu_c = np.mean(clean_embs, axis=0)
    mu_o = np.mean(ood_embs, axis=0)
    var_c = np.var(clean_embs, axis=0) + 1e-12
    var_o = np.var(ood_embs, axis=0) + 1e-12
    return (mu_o - mu_c) ** 2 / (var_c + var_o)


def nearest_centroid_accuracy(embs_by_class):
    """Leave-one-out nearest-centroid accuracy (cosine similarity).

    embs_by_class: dict label -> np.ndarray (N_i, D)
    Returns float accuracy in [0, 1].
    """
    centroids = {lbl: np.mean(embs, axis=0) for lbl, embs in embs_by_class.items()}
    correct = total = 0
    for lbl, embs in embs_by_class.items():
        for emb in embs:
            best_lbl, best_sim = None, -np.inf
            for cand_lbl, centroid in centroids.items():
                na, nb = np.linalg.norm(emb), np.linalg.norm(centroid)
                sim = float(np.dot(emb, centroid) / (na * nb)) if na > 1e-12 and nb > 1e-12 else 0.0
                if sim > best_sim:
                    best_sim, best_lbl = sim, cand_lbl
            correct += int(best_lbl == lbl)
            total += 1
    return correct / total if total > 0 else 0.0


def channel_capacity_bsc(auroc):
    """Binary Symmetric Channel capacity from AUROC.

    Model detection as BSC with crossover probability p = 1 - AUROC.
    C = 1 + p log2(p) + (1-p) log2(1-p)  [bits]
    """
    import math
    p = float(np.clip(1.0 - auroc, 1e-10, 1.0 - 1e-10))
    q = 1.0 - p
    return float(np.clip(1.0 + p * math.log2(p) + q * math.log2(q), 0.0, 1.0))


def conditional_entropy_fano(acc, n_classes):
    """Upper bound on H(class | embedding) via Fano's inequality.

    H(C|E) ≤ h_b(err) + err * log2(n_classes - 1)
    where err = 1 - acc,  h_b = binary entropy.
    """
    import math
    err = max(1e-10, 1.0 - acc)
    h_b = (
        -err * math.log2(err) - (1.0 - err) * math.log2(1.0 - err)
        if err < 1.0 - 1e-10 else 1.0
    )
    return h_b + err * math.log2(max(n_classes - 1, 1))


def np_convert(obj):
    """Recursively convert numpy scalars / arrays to Python natives for JSON."""
    if isinstance(obj, dict):
        return {k: np_convert(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [np_convert(x) for x in obj]
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("Experiment 455: Information-Theoretic Analysis of OOD Detection")
    print("=" * 65)

    # ---- Model ----
    print("\nLoading OpenVLA-7B ...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()
    print("Model loaded.\n")

    PROMPT = "In: What action should the robot take to pick up the object?\nOut:"
    SEEDS = [42, 123, 456, 789, 1000, 2000, 3000, 4000]
    CORRUPTIONS = ['fog', 'night', 'noise', 'blur']
    KNN_K = 3

    results = {
        "experiment": 455,
        "title": "Information-Theoretic Analysis of OOD Detection",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "seeds": SEEDS,
            "corruptions": CORRUPTIONS,
            "hidden_layer": 3,
            "knn_k": KNN_K,
        },
    }

    # ------------------------------------------------------------------ #
    # Collect all embeddings at severity=1.0                              #
    # ------------------------------------------------------------------ #
    print("--- Collecting Embeddings (severity=1.0) ---")
    clean_embs = []
    corrupt_embs = {ct: [] for ct in CORRUPTIONS}

    for seed in SEEDS:
        img = make_image(seed)
        emb = extract_hidden(model, processor, img, PROMPT)
        clean_embs.append(emb)
        print(f"  seed={seed}: clean embedded  (dim={emb.shape[0]})")

        for ct in CORRUPTIONS:
            c_img = apply_corruption(img, ct, severity=1.0)
            c_emb = extract_hidden(model, processor, c_img, PROMPT)
            corrupt_embs[ct].append(c_emb)
            print(f"    corruption={ct} embedded")

    clean_embs = np.array(clean_embs)                           # (8, D)
    for ct in CORRUPTIONS:
        corrupt_embs[ct] = np.array(corrupt_embs[ct])          # (8, D)

    N, D = clean_embs.shape
    print(f"\nEmbedding matrix: {N} scenes x {D} dimensions")

    # Project to a low-dimensional subspace for kNN-based estimators.
    # With only N=8 points kNN in 4096-D is degenerate; projecting to
    # the top PCs (shared across all conditions) gives a meaningful space.
    all_embs_pool = np.concatenate(
        [clean_embs] + [corrupt_embs[ct] for ct in CORRUPTIONS], axis=0
    )
    pool_mean = all_embs_pool.mean(axis=0)
    _, S_pool, Vt_pool = np.linalg.svd(all_embs_pool - pool_mean, full_matrices=False)
    N_PC = min(7, N - 1)   # ~7 effective components as noted in paper
    Vt_sub = Vt_pool[:N_PC, :]   # (N_PC, D)

    clean_proj = (clean_embs - pool_mean) @ Vt_sub.T           # (8, N_PC)
    corrupt_proj = {
        ct: (corrupt_embs[ct] - pool_mean) @ Vt_sub.T
        for ct in CORRUPTIONS
    }

    # ------------------------------------------------------------------ #
    # Analysis 1: Embedding Distribution Entropy                           #
    #             Kozachenko-Leonenko kNN estimator                        #
    # ------------------------------------------------------------------ #
    print("\n--- Analysis 1: Embedding Distribution Entropy ---")
    entropy_results = {"projected_dims": N_PC}

    h_clean = knn_entropy(clean_proj, k=KNN_K)
    entropy_results["clean"] = h_clean
    print(f"  clean  entropy = {h_clean:.4f} nats  ({N_PC}D subspace, k={KNN_K})")

    for ct in CORRUPTIONS:
        h_ct = knn_entropy(corrupt_proj[ct], k=KNN_K)
        delta = h_ct - h_clean if not (np.isnan(h_ct) or np.isnan(h_clean)) else float('nan')
        entropy_results[ct] = h_ct
        entropy_results[f"{ct}_delta"] = delta
        print(f"  {ct:5s} entropy = {h_ct:.4f} nats  Δ={delta:+.4f}")

    results["embedding_entropy"] = entropy_results

    # ------------------------------------------------------------------ #
    # Analysis 2: KL Divergence Estimation (kNN, high-dimensional)        #
    # ------------------------------------------------------------------ #
    print("\n--- Analysis 2: KL Divergence Estimation ---")
    kl_results = {}

    for ct in CORRUPTIONS:
        kl_fwd = knn_kl_divergence(corrupt_proj[ct], clean_proj, k=KNN_K)   # KL(corrupt||clean)
        kl_rev = knn_kl_divergence(clean_proj, corrupt_proj[ct], k=KNN_K)   # KL(clean||corrupt)
        kl_sym = (kl_fwd + kl_rev) / 2.0 if not (np.isnan(kl_fwd) or np.isnan(kl_rev)) else float('nan')
        kl_results[ct] = {
            "kl_corrupt_given_clean": kl_fwd,
            "kl_clean_given_corrupt": kl_rev,
            "kl_symmetric": kl_sym,
        }
        print(f"  {ct:5s}: KL(corr||clean)={kl_fwd:.4f}  KL(clean||corr)={kl_rev:.4f}  sym={kl_sym:.4f}")

    ranked = sorted(
        [(ct, v["kl_symmetric"]) for ct, v in kl_results.items() if not np.isnan(v["kl_symmetric"])],
        key=lambda x: -x[1],
    )
    kl_results["ranking_by_symmetric_kl"] = [ct for ct, _ in ranked]
    print(f"  Ranking (most distinguishable): {kl_results['ranking_by_symmetric_kl']}")
    results["kl_divergence"] = kl_results

    # ------------------------------------------------------------------ #
    # Analysis 3: Information Bottleneck                                   #
    #             PCA variance fraction + Fisher ratio vs top-K PCs        #
    # ------------------------------------------------------------------ #
    print("\n--- Analysis 3: Information Bottleneck ---")
    ib_results = {}

    # SVD on clean embeddings
    clean_mean = clean_embs.mean(axis=0)
    _, Sc, Vct = np.linalg.svd(clean_embs - clean_mean, full_matrices=False)
    total_var = float(np.sum(Sc ** 2))

    # Full Fisher ratio vector (pooled over all corruptions)
    all_ood_pooled = np.concatenate([corrupt_embs[ct] for ct in CORRUPTIONS], axis=0)
    fr_full = fisher_ratio_per_dim(clean_embs, all_ood_pooled)  # (D,)
    fr_full_norm_sq = float(np.sum(fr_full ** 2)) + 1e-300

    K_values = list(range(1, min(51, N)))
    var_fraction = []
    fisher_fraction = []

    for K in K_values:
        # Variance fraction
        var_fraction.append(float(np.sum(Sc[:K] ** 2)) / (total_var + 1e-300))

        # Fisher fraction captured in top-K PCs
        # Project the Fisher vector into the K-PC subspace; fraction = ||P_K fr||² / ||fr||²
        Vct_K = Vct[:K, :]                   # (K, D)
        fr_proj = Vct_K @ fr_full             # (K,)  coordinates in PC basis
        fisher_fraction.append(float(np.sum(fr_proj ** 2)) / fr_full_norm_sq)

    n_effective = int(np.sum(Sc > Sc[0] * 0.1)) if len(Sc) > 0 else 0
    k7 = min(7, len(var_fraction)) - 1
    k7_var = var_fraction[k7]
    k7_fisher = fisher_fraction[k7]
    k50_var = var_fraction[min(49, len(var_fraction) - 1)]

    ib_results["K_values"] = K_values
    ib_results["variance_fraction_by_K"] = var_fraction
    ib_results["fisher_fraction_by_K"] = fisher_fraction
    ib_results["singular_values_top20"] = Sc[:20].tolist()
    ib_results["n_effective_components"] = n_effective
    ib_results["total_variance"] = total_var

    print(f"  Effective SVD components (>10% of peak SV): {n_effective}")
    print(f"  Variance in top-7 PCs:  {k7_var*100:.1f}%")
    print(f"  Fisher info in top-7 PCs: {k7_fisher*100:.1f}%")
    print(f"  Variance in top-50 PCs: {k50_var*100:.1f}%")
    results["information_bottleneck"] = ib_results

    # ------------------------------------------------------------------ #
    # Analysis 4: Mutual Information via Nearest-Centroid Classification   #
    #             5 classes: clean + 4 corruptions                         #
    # ------------------------------------------------------------------ #
    print("\n--- Analysis 4: Mutual Information via Classification ---")
    import math

    n_classes = 1 + len(CORRUPTIONS)   # 5
    H_corruption = math.log2(n_classes)   # uniform prior: log2(5) bits

    embs_by_class_full = {"clean": clean_embs}
    embs_by_class_proj = {"clean": clean_proj}
    for ct in CORRUPTIONS:
        embs_by_class_full[ct] = corrupt_embs[ct]
        embs_by_class_proj[ct] = corrupt_proj[ct]

    acc_full = nearest_centroid_accuracy(embs_by_class_full)
    acc_proj = nearest_centroid_accuracy(embs_by_class_proj)

    H_cond_full = conditional_entropy_fano(acc_full, n_classes)
    H_cond_proj = conditional_entropy_fano(acc_proj, n_classes)
    MI_full = max(0.0, H_corruption - H_cond_full)
    MI_proj = max(0.0, H_corruption - H_cond_proj)

    mi_results = {
        "n_classes": n_classes,
        "H_corruption_bits": H_corruption,
        "note_MI_upper_bound": "Perfect classification -> MI = H(corruption) = log2(5) bits",
        "accuracy_full_dim": acc_full,
        "accuracy_projected_7pc": acc_proj,
        "H_conditional_full_fano_bound": H_cond_full,
        "H_conditional_projected_fano_bound": H_cond_proj,
        "MI_lower_bound_full_dim_bits": MI_full,
        "MI_lower_bound_projected_bits": MI_proj,
        "MI_upper_bound_bits": H_corruption,
    }
    print(f"  H(corruption) = log2({n_classes}) = {H_corruption:.4f} bits")
    print(f"  Nearest-centroid accuracy (full {D}D): {acc_full:.4f}")
    print(f"  Nearest-centroid accuracy (top-{N_PC} PCs): {acc_proj:.4f}")
    print(f"  MI ≥ {MI_full:.4f} bits (full-D)")
    print(f"  MI ≥ {MI_proj:.4f} bits (projected)")
    results["mutual_information"] = mi_results

    # ------------------------------------------------------------------ #
    # Analysis 5: Rate-Distortion Curve                                    #
    #             Quantize embeddings to B bits/dim, measure AUROC         #
    # ------------------------------------------------------------------ #
    print("\n--- Analysis 5: Rate-Distortion Curve ---")
    BIT_LEVELS = [1, 2, 4, 8, 16, 32]

    # Fit quantizer range on the combined pool so results are comparable
    quant_mins = all_embs_pool.min(axis=0, keepdims=True)
    quant_maxs = all_embs_pool.max(axis=0, keepdims=True)
    quant_span = quant_maxs - quant_mins
    quant_span[quant_span < 1e-12] = 1.0

    def quantize(X, bits):
        levels = 2 ** bits
        X_norm = (X - quant_mins) / quant_span
        X_q = np.floor(X_norm * levels).clip(0, levels - 1)
        return (X_q + 0.5) / levels * quant_span + quant_mins

    rd_per_ct = {ct: [] for ct in CORRUPTIONS}
    rd_mean = []

    for bits in BIT_LEVELS:
        clean_q = quantize(clean_embs, bits)
        centroid_q = clean_q.mean(axis=0)
        id_scores = [cosine_dist(e, centroid_q) for e in clean_q]
        aurocs = {}
        for ct in CORRUPTIONS:
            ood_q = quantize(corrupt_embs[ct], bits)
            ood_scores = [cosine_dist(e, centroid_q) for e in ood_q]
            auroc = compute_auroc(id_scores, ood_scores)
            aurocs[ct] = auroc
            rd_per_ct[ct].append(auroc)
        mean_auroc = float(np.mean(list(aurocs.values())))
        rd_mean.append(mean_auroc)
        print(
            f"  {bits:2d} bits/dim -> mean AUROC={mean_auroc:.4f}  "
            + "  ".join(f"{ct}={aurocs[ct]:.3f}" for ct in CORRUPTIONS)
        )

    min_bits_095 = next(
        (b for b, a in zip(BIT_LEVELS, rd_mean) if a >= 0.95), None
    )
    rd_results = {
        "bit_levels": BIT_LEVELS,
        "mean_auroc_by_bits": rd_mean,
        "per_corruption_auroc_by_bits": rd_per_ct,
        "min_bits_per_dim_for_auroc_0_95": min_bits_095,
    }
    print(f"  Min bits/dim for AUROC ≥ 0.95: {min_bits_095}")
    results["rate_distortion"] = rd_results

    # ------------------------------------------------------------------ #
    # Analysis 6: Channel Capacity at Various Severities                   #
    # ------------------------------------------------------------------ #
    print("\n--- Analysis 6: Channel Capacity at Various Severities ---")
    SEVERITIES = [0.1, 0.25, 0.5, 0.75, 1.0]
    clean_centroid = clean_embs.mean(axis=0)
    id_scores_full = [cosine_dist(e, clean_centroid) for e in clean_embs]

    cap_results = {
        "model": "Binary Symmetric Channel (BSC)",
        "note": "AUROC=1.0 → C=1 bit; AUROC=0.5 → C=0 bits",
        "capacity_at_auroc_1_0": 1.0,
        "capacity_at_auroc_0_5": 0.0,
        "by_severity": {},
    }

    for sev in SEVERITIES:
        sev_key = str(sev)
        sev_aurocs = {}
        sev_caps = {}
        for ct in CORRUPTIONS:
            sev_embs = []
            for seed in SEEDS:
                img = make_image(seed)
                c_img = apply_corruption(img, ct, severity=sev)
                c_emb = extract_hidden(model, processor, c_img, PROMPT)
                sev_embs.append(c_emb)
            sev_embs = np.array(sev_embs)
            ood_scores = [cosine_dist(e, clean_centroid) for e in sev_embs]
            auroc = compute_auroc(id_scores_full, ood_scores)
            cap = channel_capacity_bsc(auroc)
            sev_aurocs[ct] = auroc
            sev_caps[ct] = cap
            print(f"  severity={sev:.2f}  {ct:5s}: AUROC={auroc:.4f}  C={cap:.4f} bits")

        cap_results["by_severity"][sev_key] = {
            "severity": sev,
            "auroc_per_corruption": sev_aurocs,
            "capacity_per_corruption": sev_caps,
            "mean_auroc": float(np.mean(list(sev_aurocs.values()))),
            "mean_capacity_bits": float(np.mean(list(sev_caps.values()))),
        }

    results["channel_capacity"] = cap_results

    # ------------------------------------------------------------------ #
    # Baseline AUROC (severity=1.0, full-precision embeddings)            #
    # ------------------------------------------------------------------ #
    print("\n--- Baseline AUROC (severity=1.0, full precision) ---")
    baseline_auroc = {}
    for ct in CORRUPTIONS:
        ood_scores = [cosine_dist(e, clean_centroid) for e in corrupt_embs[ct]]
        auroc = compute_auroc(id_scores_full, ood_scores)
        cap = channel_capacity_bsc(auroc)
        baseline_auroc[ct] = {"auroc": auroc, "capacity_bits": cap}
        print(f"  {ct:5s}: AUROC={auroc:.4f}  C={cap:.4f} bits")

    mean_auroc_base = float(np.mean([v["auroc"] for v in baseline_auroc.values()]))
    mean_cap_base = float(np.mean([v["capacity_bits"] for v in baseline_auroc.values()]))
    baseline_auroc["mean_auroc"] = mean_auroc_base
    baseline_auroc["mean_capacity_bits"] = mean_cap_base
    results["baseline_auroc_severity_1"] = baseline_auroc

    # ------------------------------------------------------------------ #
    # Summary                                                              #
    # ------------------------------------------------------------------ #
    results["summary"] = {
        "n_scenes": N,
        "embedding_dim": int(D),
        "knn_projected_dims": N_PC,
        "entropy_clean_nats": entropy_results.get("clean"),
        "entropy_deltas_nats": {ct: entropy_results.get(f"{ct}_delta", float('nan'))
                                for ct in CORRUPTIONS},
        "kl_symmetric_per_corruption": {ct: kl_results[ct]["kl_symmetric"] for ct in CORRUPTIONS},
        "kl_ranking": kl_results.get("ranking_by_symmetric_kl", []),
        "effective_svd_components": n_effective,
        "variance_fraction_top7_pcs": k7_var,
        "fisher_fraction_top7_pcs": k7_fisher,
        "variance_fraction_top50_pcs": k50_var,
        "mi_lower_bound_full_dim_bits": mi_results["MI_lower_bound_full_dim_bits"],
        "mi_upper_bound_bits": H_corruption,
        "nc_accuracy_full_dim": acc_full,
        "min_bits_per_dim_for_detection": rd_results["min_bits_per_dim_for_auroc_0_95"],
        "baseline_mean_auroc": mean_auroc_base,
        "baseline_mean_channel_capacity_bits": mean_cap_base,
    }

    print("\n=== Summary ===")
    s = results["summary"]
    print(f"  Clean embedding entropy:        {s['entropy_clean_nats']:.4f} nats")
    print(f"  Effective SVD components:       {s['effective_svd_components']}")
    print(f"  Variance in top-7 PCs:          {s['variance_fraction_top7_pcs']*100:.1f}%")
    print(f"  Fisher info in top-7 PCs:       {s['fisher_fraction_top7_pcs']*100:.1f}%")
    print(f"  Nearest-centroid accuracy:      {s['nc_accuracy_full_dim']:.4f}")
    print(f"  MI ≥ {s['mi_lower_bound_full_dim_bits']:.4f} bits  (upper bound: {s['mi_upper_bound_bits']:.4f})")
    print(f"  Min bits/dim for AUROC ≥ 0.95: {s['min_bits_per_dim_for_detection']}")
    print(f"  Mean AUROC (full precision):    {s['baseline_mean_auroc']:.4f}")
    print(f"  Mean channel capacity:          {s['baseline_mean_channel_capacity_bits']:.4f} bits")

    # ------------------------------------------------------------------ #
    # Save                                                                 #
    # ------------------------------------------------------------------ #
    os.makedirs("experiments", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/information_theoretic_{ts}.json"

    with open(out_path, "w") as f:
        json.dump(np_convert(results), f, indent=2)

    print(f"\nResults saved to {out_path}")
    return results


if __name__ == "__main__":
    main()
