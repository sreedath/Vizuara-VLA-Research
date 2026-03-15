"""
Feature Attribution Analysis for OOD Detection.

Experiment 448: Analyzes WHICH dimensions of the hidden state embedding
are most important for OOD detection, and whether different corruptions
activate different subsets of dimensions.

Analyses:
  1. Dimension Importance Ranking — top-20 dims per corruption type
  2. Corruption-Specific vs Shared Dimensions — intersection of top-50 across corruptions
  3. Sparse Detection Test — AUROC at K=10..2048 vs full 4096-d cosine distance
  4. Dimension Correlation with Severity — Pearson r per top-50 dim across severities
  5. PCA of Corruption Directions — mean shift vectors, explained variance
  6. Random Projection Baseline — random K dims vs informed top-K (10 seeds)
"""

import torch, json, os, sys, numpy as np
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor
from datetime import datetime

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
EXPERIMENTS_DIR = os.path.join(REPO_DIR, "experiments")
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Image helpers (from standard CalibDrive patterns)
# ---------------------------------------------------------------------------

def make_image(seed=42):
    rng = np.random.RandomState(seed)
    return Image.fromarray(rng.randint(0, 255, (224, 224, 3), dtype=np.uint8))


def apply_corruption(image, ctype, severity=1.0):
    arr = np.array(image).astype(np.float32) / 255.0
    if ctype == "fog":
        arr = arr * (1 - 0.6 * severity) + 0.6 * severity
    elif ctype == "night":
        arr = arr * max(0.01, 1.0 - 0.95 * severity)
    elif ctype == "noise":
        arr = arr + np.random.RandomState(42).randn(*arr.shape) * 0.3 * severity
        arr = np.clip(arr, 0, 1)
    elif ctype == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=10 * severity))
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract_hidden(model, processor, image, prompt, layer=3):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return fwd.hidden_states[layer][0, -1, :].float().cpu().numpy()


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------

def cosine_dist_full(a, b):
    """Cosine distance using all dimensions."""
    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def cosine_dist_subset(a, b, indices):
    """Cosine distance restricted to given dimension indices."""
    am = a[indices]
    bm = b[indices]
    return float(1.0 - np.dot(am, bm) / (np.linalg.norm(am) * np.linalg.norm(bm) + 1e-10))


# ---------------------------------------------------------------------------
# AUROC helper (no sklearn dependency for a single score vector)
# ---------------------------------------------------------------------------

def auroc(scores, labels):
    """Binary AUROC via trapezoidal rule. labels: 1=positive (OOD), 0=negative (clean)."""
    labels = np.array(labels)
    scores = np.array(scores)
    n_pos = int(labels.sum())
    n_neg = int((1 - labels).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    tp = np.cumsum(labels_sorted) / n_pos
    fp = np.cumsum(1 - labels_sorted) / n_neg
    fp = np.concatenate([[0.0], fp])
    tp = np.concatenate([[0.0], tp])
    return float(np.trapezoid(tp, fp)) if hasattr(np, 'trapezoid') else float(np.trapz(tp, fp))


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    print("=" * 72, flush=True)
    print("EXPERIMENT 448: Feature Attribution Analysis", flush=True)
    print("=" * 72, flush=True)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print("\nLoading OpenVLA-7B...", flush=True)
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()
    print("Model loaded.", flush=True)

    prompt = "In: What action should the robot take to pick up the object?\nOut:"

    SEEDS = [42, 123, 456, 789, 1000, 2000, 3000, 4000]
    CORRUPTION_TYPES = ["fog", "night", "noise", "blur"]
    LAYER = 3
    MAIN_SEVERITY = 1.0
    SEVERITIES = [0.1, 0.25, 0.5, 0.75, 1.0]
    K_VALUES = [10, 20, 50, 100, 200, 500, 1000, 2048]
    TOP_N_RANK = 20          # top dims to report per corruption
    TOP_N_SHARED = 50        # top dims used for shared/specific analysis
    TOP_N_SEVERITY = 50      # top dims used for severity correlation
    N_RANDOM_SEEDS = 10

    # ------------------------------------------------------------------
    # Step 1: Collect clean and corrupted embeddings at severity=1.0
    # ------------------------------------------------------------------
    print(f"\n[Step 1] Collecting embeddings (layer={LAYER}, "
          f"scenes={len(SEEDS)}, corruptions={CORRUPTION_TYPES})...", flush=True)

    clean_embeddings = []  # shape: (n_scenes, D)
    corrupted_embeddings = {c: [] for c in CORRUPTION_TYPES}

    for i, seed in enumerate(SEEDS):
        clean_img = make_image(seed)
        h_clean = extract_hidden(model, processor, clean_img, prompt, LAYER)
        clean_embeddings.append(h_clean)

        for ctype in CORRUPTION_TYPES:
            corr_img = apply_corruption(clean_img, ctype, severity=MAIN_SEVERITY)
            h_corr = extract_hidden(model, processor, corr_img, prompt, LAYER)
            corrupted_embeddings[ctype].append(h_corr)

        print(f"  Scene {i+1}/{len(SEEDS)} (seed={seed}) done.", flush=True)

    clean_embeddings = np.array(clean_embeddings)      # (8, D)
    for c in CORRUPTION_TYPES:
        corrupted_embeddings[c] = np.array(corrupted_embeddings[c])  # (8, D)

    D = clean_embeddings.shape[1]
    print(f"\nEmbedding dimension D={D}", flush=True)

    # Clean centroid (used throughout)
    clean_centroid = clean_embeddings.mean(axis=0)

    # ------------------------------------------------------------------
    # Step 2: Collect severity-variant embeddings for correlation analysis
    # ------------------------------------------------------------------
    print(f"\n[Step 2] Collecting severity-variant embeddings "
          f"(severities={SEVERITIES})...", flush=True)

    # severity_embeddings[ctype][sev_idx] -> (n_scenes, D)
    severity_embeddings = {c: [] for c in CORRUPTION_TYPES}

    for sev in SEVERITIES:
        for ctype in CORRUPTION_TYPES:
            sev_embeds = []
            for seed in SEEDS:
                img = make_image(seed)
                corr_img = apply_corruption(img, ctype, severity=sev)
                h = extract_hidden(model, processor, corr_img, prompt, LAYER)
                sev_embeds.append(h)
            severity_embeddings[ctype].append(np.array(sev_embeds))  # (8, D)
        print(f"  Severity {sev} done.", flush=True)

    # severity_embeddings[ctype] is now list of length len(SEVERITIES),
    # each element (n_scenes, D)

    # ------------------------------------------------------------------
    # Analysis 1: Dimension Importance Ranking
    # ------------------------------------------------------------------
    print("\n[Analysis 1] Dimension Importance Ranking...", flush=True)

    # mean absolute difference per dimension, per corruption
    dim_importance = {}          # ctype -> array (D,)
    top20_per_corruption = {}    # ctype -> list of {rank, dim, mean_abs_diff}

    for ctype in CORRUPTION_TYPES:
        diff = np.abs(corrupted_embeddings[ctype] - clean_embeddings)  # (8, D)
        mean_diff = diff.mean(axis=0)                                   # (D,)
        dim_importance[ctype] = mean_diff

        ranked = np.argsort(mean_diff)[::-1]
        top20 = []
        for rank in range(TOP_N_RANK):
            d = int(ranked[rank])
            top20.append({"rank": rank, "dim": d, "mean_abs_diff": float(mean_diff[d])})
        top20_per_corruption[ctype] = top20
        print(f"  {ctype}: top dim={ranked[0]}, mean_abs_diff={mean_diff[ranked[0]]:.6f}",
              flush=True)

    # ------------------------------------------------------------------
    # Analysis 2: Corruption-Specific vs Shared Dimensions
    # ------------------------------------------------------------------
    print("\n[Analysis 2] Corruption-Specific vs Shared Dimensions...", flush=True)

    top50_sets = {}
    for ctype in CORRUPTION_TYPES:
        ranked = np.argsort(dim_importance[ctype])[::-1]
        top50_sets[ctype] = set(int(d) for d in ranked[:TOP_N_SHARED])

    shared_dims = set.intersection(*top50_sets.values())

    corruption_specific = {}
    for ctype in CORRUPTION_TYPES:
        others = set.union(*(top50_sets[c] for c in CORRUPTION_TYPES if c != ctype))
        corruption_specific[ctype] = sorted(top50_sets[ctype] - others)

    pairwise_overlap = {}
    ctypes = CORRUPTION_TYPES
    for i in range(len(ctypes)):
        for j in range(i + 1, len(ctypes)):
            a, b = ctypes[i], ctypes[j]
            overlap = len(top50_sets[a] & top50_sets[b])
            pairwise_overlap[f"{a}_vs_{b}"] = overlap

    print(f"  Shared dims (all 4 corruptions): {len(shared_dims)}", flush=True)
    for ctype in CORRUPTION_TYPES:
        print(f"  {ctype}-specific (not in any other top-50): "
              f"{len(corruption_specific[ctype])}", flush=True)
    for pair, overlap in pairwise_overlap.items():
        print(f"  Pairwise overlap {pair}: {overlap}", flush=True)

    # ------------------------------------------------------------------
    # Analysis 3: Sparse Detection Test
    # ------------------------------------------------------------------
    print("\n[Analysis 3] Sparse Detection Test...", flush=True)

    # Build test set: all clean vs all corrupted across all corruptions
    # Label 0=clean, 1=corrupted
    test_clean = clean_embeddings           # (8, D)
    test_corr = np.concatenate(
        [corrupted_embeddings[c] for c in CORRUPTION_TYPES], axis=0)  # (32, D)
    test_labels = [0] * len(test_clean) + [1] * len(test_corr)
    test_all = np.concatenate([test_clean, test_corr], axis=0)        # (40, D)

    # Full AUROC baseline
    full_scores = np.array([cosine_dist_full(e, clean_centroid) for e in test_all])
    full_auroc = auroc(full_scores, test_labels)
    print(f"  Full (D={D}) AUROC: {full_auroc:.4f}", flush=True)

    # Informed top-K: use union of top-K dims across all corruptions
    # (dims ranked by mean importance across all corruption types)
    global_importance = np.mean(
        np.stack([dim_importance[c] for c in CORRUPTION_TYPES], axis=0), axis=0)  # (D,)
    global_ranked = np.argsort(global_importance)[::-1]

    sparse_results = {"full": {"k": D, "auroc": full_auroc}}
    for k in K_VALUES:
        if k >= D:
            k_actual = D
        else:
            k_actual = k
        top_k_idx = global_ranked[:k_actual]
        scores_k = np.array([cosine_dist_subset(e, clean_centroid, top_k_idx) for e in test_all])
        auc_k = auroc(scores_k, test_labels)
        sparse_results[str(k)] = {
            "k": k_actual,
            "auroc": auc_k,
            "auroc_vs_full": auc_k - full_auroc,
        }
        print(f"  Top-{k_actual:5d} dims AUROC: {auc_k:.4f} "
              f"(delta={auc_k - full_auroc:+.4f})", flush=True)

    # ------------------------------------------------------------------
    # Analysis 4: Dimension Correlation with Severity
    # ------------------------------------------------------------------
    print("\n[Analysis 4] Dimension Correlation with Severity...", flush=True)

    # Use global top-50 dims for brevity
    top50_global = [int(d) for d in global_ranked[:TOP_N_SEVERITY]]
    severity_array = np.array(SEVERITIES)  # (S,)

    # For each corruption: per-dim activation vs severity
    # Activation = mean absolute value across scenes at each severity
    severity_correlation = {}

    for ctype in CORRUPTION_TYPES:
        # Stack mean absolute activations: shape (S, D)
        sev_mean_abs = np.array([
            np.abs(severity_embeddings[ctype][si]).mean(axis=0)   # (D,)
            for si in range(len(SEVERITIES))
        ])  # (S, D)

        per_dim_corr = []
        for dim_idx in top50_global:
            dim_vals = sev_mean_abs[:, dim_idx]  # (S,)
            # Pearson correlation
            if dim_vals.std() < 1e-10:
                r = 0.0
            else:
                r = float(np.corrcoef(severity_array, dim_vals)[0, 1])
            per_dim_corr.append({"dim": dim_idx, "pearson_r": r})

        # Sort by |r|
        per_dim_corr.sort(key=lambda x: abs(x["pearson_r"]), reverse=True)

        high_corr = [x for x in per_dim_corr if abs(x["pearson_r"]) >= 0.9]
        med_corr = [x for x in per_dim_corr if 0.5 <= abs(x["pearson_r"]) < 0.9]
        low_corr = [x for x in per_dim_corr if abs(x["pearson_r"]) < 0.5]

        severity_correlation[ctype] = {
            "top10_dims_by_correlation": per_dim_corr[:10],
            "n_high_corr_r_gte_0.9": len(high_corr),
            "n_med_corr_0.5_to_0.9": len(med_corr),
            "n_low_corr_lt_0.5": len(low_corr),
            "mean_abs_r": float(np.mean([abs(x["pearson_r"]) for x in per_dim_corr])),
        }
        print(f"  {ctype}: high_corr(|r|>=0.9)={len(high_corr)}, "
              f"mean|r|={severity_correlation[ctype]['mean_abs_r']:.3f}", flush=True)

    # ------------------------------------------------------------------
    # Analysis 5: PCA of Corruption Directions
    # ------------------------------------------------------------------
    print("\n[Analysis 5] PCA of Corruption Directions...", flush=True)

    # Mean shift vector per corruption: corrupted_centroid - clean_centroid
    shift_vectors = np.array([
        corrupted_embeddings[c].mean(axis=0) - clean_centroid
        for c in CORRUPTION_TYPES
    ])  # (4, D)

    # SVD of the 4 x D matrix of shift vectors
    U, S_sv, Vt = np.linalg.svd(shift_vectors, full_matrices=False)
    total_variance = float((S_sv ** 2).sum())
    explained_variance_ratio = [(float(s ** 2) / total_variance) for s in S_sv]
    cumulative_evr = list(np.cumsum(explained_variance_ratio))

    # Cosine similarities between shift vectors (pair-wise)
    shift_cosine_sim = {}
    for i in range(len(CORRUPTION_TYPES)):
        for j in range(i + 1, len(CORRUPTION_TYPES)):
            a, b = CORRUPTION_TYPES[i], CORRUPTION_TYPES[j]
            va = shift_vectors[i]
            vb = shift_vectors[j]
            sim = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-10))
            shift_cosine_sim[f"{a}_vs_{b}"] = sim

    pca_results = {
        "n_corruption_types": len(CORRUPTION_TYPES),
        "corruption_types": CORRUPTION_TYPES,
        "singular_values": [float(s) for s in S_sv],
        "explained_variance_ratio": explained_variance_ratio,
        "cumulative_explained_variance": cumulative_evr,
        "shift_vector_norms": [float(np.linalg.norm(shift_vectors[i]))
                               for i in range(len(CORRUPTION_TYPES))],
        "pairwise_cosine_similarity": shift_cosine_sim,
        "interpretation": {
            "n_components_for_90pct": int(
                np.searchsorted(np.array(cumulative_evr), 0.90) + 1),
            "first_component_evr": float(explained_variance_ratio[0]),
        },
    }

    print(f"  Explained variance ratio: "
          f"{[f'{v:.3f}' for v in explained_variance_ratio]}", flush=True)
    print(f"  Pairwise cosine similarities:", flush=True)
    for pair, sim in shift_cosine_sim.items():
        print(f"    {pair}: {sim:.4f}", flush=True)

    # ------------------------------------------------------------------
    # Analysis 6: Random Projection Baseline
    # ------------------------------------------------------------------
    print("\n[Analysis 6] Random Projection Baseline...", flush=True)

    K_PROJ_VALUES = [10, 50, 100, 500]
    random_projection_results = {}

    for k in K_PROJ_VALUES:
        if k >= D:
            k_actual = D
        else:
            k_actual = k

        # Informed (top-K) AUROC already computed above
        top_k_idx = global_ranked[:k_actual]
        scores_informed = np.array(
            [cosine_dist_subset(e, clean_centroid, top_k_idx) for e in test_all])
        auc_informed = auroc(scores_informed, test_labels)

        # Random projections: 10 seeds
        rand_aurocs = []
        rng_proj = np.random.RandomState(0)
        for rs in range(N_RANDOM_SEEDS):
            rand_idx = rng_proj.choice(D, k_actual, replace=False)
            scores_rand = np.array(
                [cosine_dist_subset(e, clean_centroid, rand_idx) for e in test_all])
            rand_aurocs.append(auroc(scores_rand, test_labels))

        mean_rand = float(np.mean(rand_aurocs))
        std_rand = float(np.std(rand_aurocs))
        gain = float(auc_informed - mean_rand)

        random_projection_results[str(k)] = {
            "k": k_actual,
            "informed_auroc": auc_informed,
            "random_auroc_mean": mean_rand,
            "random_auroc_std": std_rand,
            "random_auroc_all_seeds": [float(v) for v in rand_aurocs],
            "informed_vs_random_gain": gain,
        }
        print(f"  K={k_actual:5d}: informed={auc_informed:.4f}, "
              f"random={mean_rand:.4f}+/-{std_rand:.4f}, gain={gain:+.4f}", flush=True)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output = {
        "experiment": "feature_attribution",
        "experiment_number": 448,
        "timestamp": timestamp,
        "config": {
            "seeds": SEEDS,
            "corruption_types": CORRUPTION_TYPES,
            "layer": LAYER,
            "main_severity": MAIN_SEVERITY,
            "severities_for_correlation": SEVERITIES,
            "k_values_sparse": K_VALUES,
            "k_values_random_proj": K_PROJ_VALUES,
            "top_n_rank": TOP_N_RANK,
            "top_n_shared": TOP_N_SHARED,
            "top_n_severity_corr": TOP_N_SEVERITY,
            "n_random_seeds": N_RANDOM_SEEDS,
            "embedding_dim": D,
        },
        "analysis_1_dimension_importance": {
            ctype: {
                "top20_dims": top20_per_corruption[ctype],
                "global_importance_stats": {
                    "max": float(dim_importance[ctype].max()),
                    "mean": float(dim_importance[ctype].mean()),
                    "std": float(dim_importance[ctype].std()),
                },
            }
            for ctype in CORRUPTION_TYPES
        },
        "analysis_2_shared_vs_specific": {
            "top_n": TOP_N_SHARED,
            "n_shared_all_corruptions": len(shared_dims),
            "shared_dims": sorted(shared_dims),
            "top50_per_corruption": {c: sorted(top50_sets[c]) for c in CORRUPTION_TYPES},
            "corruption_specific_dims": corruption_specific,
            "pairwise_overlap_counts": pairwise_overlap,
        },
        "analysis_3_sparse_detection": sparse_results,
        "analysis_4_severity_correlation": severity_correlation,
        "analysis_5_pca_corruption_directions": pca_results,
        "analysis_6_random_projection_baseline": random_projection_results,
    }

    output_path = os.path.join(EXPERIMENTS_DIR, f"feature_attribution_{timestamp}.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}", flush=True)
    print("=" * 72, flush=True)
    print("EXPERIMENT 448 COMPLETE", flush=True)
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
