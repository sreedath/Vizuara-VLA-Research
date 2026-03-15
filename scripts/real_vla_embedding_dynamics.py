"""
Embedding Dynamics Across Model Depth on Real OpenVLA-7B.

Traces how the embedding of a single image evolves across ALL transformer
layers, and how corruption affects this trajectory. Gives insight into WHERE
in the model the corruption signal first emerges.

Analyses:
  1. Layer-to-Layer Cosine Similarity
  2. Corruption Divergence Emergence
  3. Embedding Velocity (L2 norm of consecutive-layer diffs)
  4. Representational Similarity Analysis (RSA)
  5. Corruption Signal Amplification
  6. Layer Clustering (silhouette score, NN accuracy)

Experiment 456 in the CalibDrive series.
"""
import torch
import json
import os
import sys
import numpy as np
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor
from datetime import datetime

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "experiments",
)
os.makedirs(RESULTS_DIR, exist_ok=True)

SEEDS = [42, 123, 456, 789, 1000, 2000, 3000, 4000]
CORRUPTION_TYPES = ["fog", "night", "noise", "blur"]
SEVERITY = 1.0
PROMPT = "In: What action should the robot take to drive forward safely?\nOut:"


# ---------------------------------------------------------------------------
# Image helpers
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
# Forward pass — extract all hidden layers in one pass
# ---------------------------------------------------------------------------

def extract_all_layers(model, processor, image, prompt):
    """Return list of per-layer last-token embeddings as float32 numpy arrays."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return [h[0, -1, :].float().cpu().numpy() for h in fwd.hidden_states]


# ---------------------------------------------------------------------------
# Math helpers (no sklearn)
# ---------------------------------------------------------------------------

def cosine_similarity(a, b):
    """Scalar cosine similarity between two 1-D arrays."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def cosine_distance(a, b):
    return 1.0 - cosine_similarity(a, b)


def pairwise_cosine_distances(matrix):
    """
    matrix: (N, D) float array.
    Returns (N, N) cosine distance matrix.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1e-12, norms)
    normed = matrix / norms
    sim = normed @ normed.T
    sim = np.clip(sim, -1.0, 1.0)
    return 1.0 - sim


def pearson_correlation(x, y):
    """Scalar Pearson r between two 1-D arrays."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


# ---------------------------------------------------------------------------
# Hand-written silhouette score (no sklearn)
# ---------------------------------------------------------------------------

def silhouette_score_handwritten(embeddings, labels):
    """
    Compute mean silhouette coefficient without sklearn.

    s(i) = (b(i) - a(i)) / max(a(i), b(i))

    a(i): mean intra-cluster cosine distance (excluding self).
    b(i): mean cosine distance to nearest other cluster.

    Returns mean silhouette over all samples, or 0.0 if only one cluster.
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0

    n = len(embeddings)
    dist_mat = pairwise_cosine_distances(embeddings)

    silhouettes = []
    for i in range(n):
        lbl_i = labels[i]

        # a(i): mean intra-cluster distance
        same_mask = (labels == lbl_i)
        same_mask[i] = False  # exclude self
        if same_mask.sum() == 0:
            a_i = 0.0
        else:
            a_i = float(dist_mat[i, same_mask].mean())

        # b(i): mean distance to nearest other cluster
        b_i = np.inf
        for other_lbl in unique_labels:
            if other_lbl == lbl_i:
                continue
            other_mask = (labels == other_lbl)
            mean_dist = float(dist_mat[i, other_mask].mean())
            if mean_dist < b_i:
                b_i = mean_dist

        if np.isinf(b_i):
            b_i = 0.0

        denom = max(a_i, b_i)
        s_i = (b_i - a_i) / denom if denom > 1e-12 else 0.0
        silhouettes.append(s_i)

    return float(np.mean(silhouettes))


def nearest_neighbor_accuracy(embeddings, labels):
    """
    1-NN leave-one-out accuracy using cosine distance.
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    labels = np.asarray(labels)
    n = len(embeddings)
    dist_mat = pairwise_cosine_distances(embeddings)
    np.fill_diagonal(dist_mat, np.inf)

    correct = 0
    for i in range(n):
        nn_idx = int(np.argmin(dist_mat[i]))
        if labels[nn_idx] == labels[i]:
            correct += 1
    return correct / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70, flush=True)
    print("EXPERIMENT 456: EMBEDDING DYNAMICS ACROSS MODEL DEPTH", flush=True)
    print("=" * 70, flush=True)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print("\nLoading OpenVLA-7B...", flush=True)
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        "openvla/openvla-7b",
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.", flush=True)

    # ------------------------------------------------------------------
    # Collect all embeddings
    # conditions: "clean" + 4 corruption types  => 5 per scene
    # scenes: 8 seeds
    # total forward passes: 40
    # ------------------------------------------------------------------
    conditions = ["clean"] + CORRUPTION_TYPES
    n_scenes = len(SEEDS)
    n_conditions = len(conditions)
    total_passes = n_scenes * n_conditions

    print(f"\nScenes: {n_scenes}  |  Conditions: {n_conditions}  "
          f"|  Total forward passes: {total_passes}", flush=True)

    # all_layers[scene_idx][condition_idx] = list of per-layer numpy arrays
    all_layers = []
    n_layers = None
    pass_idx = 0

    for s_idx, seed in enumerate(SEEDS):
        scene_rows = []
        base_image = make_image(seed)
        for c_idx, cond in enumerate(conditions):
            pass_idx += 1
            if cond == "clean":
                image = base_image
            else:
                image = apply_corruption(base_image, cond, SEVERITY)

            print(f"  [{pass_idx}/{total_passes}] scene seed={seed}  "
                  f"condition={cond}", flush=True)

            layers = extract_all_layers(model, processor, image, PROMPT)

            if n_layers is None:
                n_layers = len(layers)
                print(f"  -> {n_layers} hidden layers detected.", flush=True)

            scene_rows.append(layers)
        all_layers.append(scene_rows)

    print(f"\nAll embeddings collected. n_layers={n_layers}", flush=True)

    # ------------------------------------------------------------------
    # Index helpers
    # ------------------------------------------------------------------
    CLEAN_IDX = 0  # conditions[0] == "clean"

    # ------------------------------------------------------------------
    # Analysis 1: Layer-to-Layer Cosine Similarity
    # For each pair of adjacent layers (L_i, L_{i+1}), compute cosine
    # similarity. Compare clean vs corrupted trajectories.
    # ------------------------------------------------------------------
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS 1: Layer-to-Layer Cosine Similarity", flush=True)
    print("=" * 70, flush=True)
    print("  (mean adjacent-layer cosine similarity across all scenes)", flush=True)
    header = f"  {'Layer':>6}  {'Clean':>8}  " + \
             "  ".join(f"{c:>8}" for c in CORRUPTION_TYPES)
    print(header, flush=True)
    print("  " + "-" * (len(header) - 2), flush=True)

    adj_cos_per_layer = {cond: [] for cond in conditions}

    for layer_i in range(n_layers - 1):
        row_parts = [f"  {layer_i:>6}"]
        for c_idx, cond in enumerate(conditions):
            sims = []
            for s_idx in range(n_scenes):
                h_curr = all_layers[s_idx][c_idx][layer_i]
                h_next = all_layers[s_idx][c_idx][layer_i + 1]
                sims.append(cosine_similarity(h_curr, h_next))
            mean_sim = float(np.mean(sims))
            adj_cos_per_layer[cond].append(mean_sim)
            row_parts.append(f"{mean_sim:>8.4f}")
        print("  ".join(row_parts), flush=True)

    # ------------------------------------------------------------------
    # Analysis 2: Corruption Divergence Emergence
    # At each layer, compute cosine distance between clean and corrupted
    # embedding (same scene). Where does the corruption signal first
    # appear? Where is it strongest?
    # ------------------------------------------------------------------
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS 2: Corruption Divergence Emergence", flush=True)
    print("=" * 70, flush=True)
    print("  (mean cosine distance between clean & corrupted embedding per layer)", flush=True)
    header = f"  {'Layer':>6}  " + \
             "  ".join(f"{c:>8}" for c in CORRUPTION_TYPES)
    print(header, flush=True)
    print("  " + "-" * (len(header) - 2), flush=True)

    divergence_per_layer = {ctype: [] for ctype in CORRUPTION_TYPES}

    for layer_i in range(n_layers):
        row_parts = [f"  {layer_i:>6}"]
        for c_idx, ctype in enumerate(CORRUPTION_TYPES, start=1):
            dists = []
            for s_idx in range(n_scenes):
                h_clean = all_layers[s_idx][CLEAN_IDX][layer_i]
                h_corr = all_layers[s_idx][c_idx][layer_i]
                dists.append(cosine_distance(h_clean, h_corr))
            mean_dist = float(np.mean(dists))
            divergence_per_layer[ctype].append(mean_dist)
            row_parts.append(f"{mean_dist:>8.4f}")
        print("  ".join(row_parts), flush=True)

    print("\n  Peak divergence layer per corruption type:", flush=True)
    peak_divergence_layers = {}
    emergence_layers = {}
    for ctype in CORRUPTION_TYPES:
        traj = divergence_per_layer[ctype]
        peak_layer = int(np.argmax(traj))
        peak_divergence_layers[ctype] = peak_layer
        # Emergence: first layer where divergence exceeds 10% of peak value
        threshold = 0.10 * traj[peak_layer]
        emergence_layer = next(
            (i for i, v in enumerate(traj) if v > threshold), peak_layer
        )
        emergence_layers[ctype] = emergence_layer
        print(f"    {ctype:<8}: peak_layer={peak_layer}  "
              f"emergence_layer={emergence_layer}  "
              f"peak_dist={traj[peak_layer]:.4f}", flush=True)

    # ------------------------------------------------------------------
    # Analysis 3: Embedding Velocity
    # velocity(L) = ||h_{L+1} - h_L||_2
    # Measures how fast the representation changes across depth.
    # Compare clean vs corrupted velocity profiles.
    # ------------------------------------------------------------------
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS 3: Embedding Velocity  ||h_{L+1} - h_L||_2", flush=True)
    print("=" * 70, flush=True)
    header = f"  {'Layer':>6}  {'Clean':>10}  " + \
             "  ".join(f"{c:>10}" for c in CORRUPTION_TYPES)
    print(header, flush=True)
    print("  " + "-" * (len(header) - 2), flush=True)

    velocity_per_layer = {cond: [] for cond in conditions}

    for layer_i in range(n_layers - 1):
        row_parts = [f"  {layer_i:>6}"]
        for c_idx, cond in enumerate(conditions):
            vels = []
            for s_idx in range(n_scenes):
                h_curr = all_layers[s_idx][c_idx][layer_i]
                h_next = all_layers[s_idx][c_idx][layer_i + 1]
                vels.append(float(np.linalg.norm(h_next - h_curr)))
            mean_vel = float(np.mean(vels))
            velocity_per_layer[cond].append(mean_vel)
            row_parts.append(f"{mean_vel:>10.3f}")
        print("  ".join(row_parts), flush=True)

    # ------------------------------------------------------------------
    # Analysis 4: Representational Similarity Analysis (RSA)
    # At each layer, compute the pairwise cosine distance matrix between
    # all 40 conditions (8 scenes x 5 conditions). Correlate these RDMs
    # across adjacent layers and against layer 0.
    # ------------------------------------------------------------------
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS 4: RSA — Cross-Layer Geometry Correlation", flush=True)
    print("=" * 70, flush=True)
    print("  Pearson r of flattened pairwise distance matrices between layers.", flush=True)
    print("  Adj_r: correlation of RDM(L) vs RDM(L-1).", flush=True)
    print("  ToL0_r: correlation of RDM(L) vs RDM(0).", flush=True)

    # Project onto first RSA_MAX_DIM dims to keep computation tractable
    RSA_MAX_DIM = 512

    rsa_adjacent_corr = []
    rsa_to_layer0_corr = []
    rdm_layer0_flat = None
    prev_rdm_flat = None

    for layer_i in range(n_layers):
        mat = []
        for s_idx in range(n_scenes):
            for c_idx in range(n_conditions):
                h = all_layers[s_idx][c_idx][layer_i][:RSA_MAX_DIM]
                mat.append(h)
        mat = np.asarray(mat, dtype=np.float64)  # (n_scenes*n_conditions, RSA_MAX_DIM)
        rdm = pairwise_cosine_distances(mat)
        idx_upper = np.triu_indices(len(mat), k=1)
        rdm_flat = rdm[idx_upper]

        if layer_i == 0:
            rdm_layer0_flat = rdm_flat.copy()
            rsa_to_layer0_corr.append(1.0)
            rsa_adjacent_corr.append(float("nan"))
        else:
            r_to_0 = pearson_correlation(rdm_flat, rdm_layer0_flat)
            rsa_to_layer0_corr.append(r_to_0)
            r_adj = pearson_correlation(rdm_flat, prev_rdm_flat)
            rsa_adjacent_corr.append(r_adj)

        prev_rdm_flat = rdm_flat.copy()

    print(f"\n  {'Layer':>6}  {'Adj_r':>8}  {'ToL0_r':>8}", flush=True)
    print("  " + "-" * 28, flush=True)
    for layer_i in range(n_layers):
        adj_val = rsa_adjacent_corr[layer_i]
        if isinstance(adj_val, float) and adj_val != adj_val:  # nan
            adj_str = "     N/A"
        else:
            adj_str = f"{adj_val:>8.4f}"
        print(f"  {layer_i:>6}  {adj_str}  "
              f"{rsa_to_layer0_corr[layer_i]:>8.4f}", flush=True)

    # Largest representational change: layer with lowest adjacent r
    valid_adj = [(i, v) for i, v in enumerate(rsa_adjacent_corr)
                 if isinstance(v, float) and v == v]  # exclude nan
    if valid_adj:
        min_adj_layer, min_adj_val = min(valid_adj, key=lambda x: x[1])
        print(f"\n  Largest representational change: layer {min_adj_layer - 1} -> "
              f"{min_adj_layer}  (adj_r={min_adj_val:.4f})", flush=True)

    # ------------------------------------------------------------------
    # Analysis 5: Corruption Signal Amplification
    # ratio(L) = corruption_divergence(L) / clean_variance(L)
    # clean_variance: mean pairwise cosine distance among 8 clean embeds.
    # ------------------------------------------------------------------
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS 5: Corruption Signal Amplification", flush=True)
    print("=" * 70, flush=True)
    print("  ratio = corruption_divergence[L] / clean_variance[L]", flush=True)

    clean_variance_per_layer = []
    for layer_i in range(n_layers):
        clean_mat = np.asarray(
            [all_layers[s_idx][CLEAN_IDX][layer_i] for s_idx in range(n_scenes)],
            dtype=np.float64,
        )
        rdm_clean = pairwise_cosine_distances(clean_mat)
        idx_upper = np.triu_indices(n_scenes, k=1)
        clean_variance_per_layer.append(float(rdm_clean[idx_upper].mean()))

    header = f"  {'Layer':>6}  {'CleanVar':>10}  " + \
             "  ".join(f"{c:>10}" for c in CORRUPTION_TYPES)
    print(header, flush=True)
    print("  " + "-" * (len(header) - 2), flush=True)

    amplification_per_layer = {ctype: [] for ctype in CORRUPTION_TYPES}

    for layer_i in range(n_layers):
        clean_var = clean_variance_per_layer[layer_i]
        row_parts = [f"  {layer_i:>6}", f"{clean_var:>10.4f}"]
        for ctype in CORRUPTION_TYPES:
            div = divergence_per_layer[ctype][layer_i]
            ratio = div / (clean_var + 1e-12)
            amplification_per_layer[ctype].append(float(ratio))
            row_parts.append(f"{ratio:>10.3f}")
        print("  ".join(row_parts), flush=True)

    print("\n  Layer of maximum amplification per corruption type:", flush=True)
    peak_amplification_layers = {}
    for ctype in CORRUPTION_TYPES:
        traj = amplification_per_layer[ctype]
        peak_layer = int(np.argmax(traj))
        peak_amplification_layers[ctype] = peak_layer
        print(f"    {ctype:<8}: layer={peak_layer}  "
              f"ratio={traj[peak_layer]:.3f}", flush=True)

    # ------------------------------------------------------------------
    # Analysis 6: Layer Clustering
    # At each layer: silhouette score and 1-NN accuracy for 5-class
    # clustering (clean + 4 corruption types), 40 embeddings total.
    # Silhouette is hand-written (no sklearn).
    # ------------------------------------------------------------------
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS 6: Layer Clustering (Silhouette + NN Accuracy)", flush=True)
    print("=" * 70, flush=True)
    print("  5-class clustering: clean + 4 corruption types.", flush=True)
    print("  40 embeddings per layer (8 scenes x 5 conditions).", flush=True)
    print("  Hand-written silhouette: s(i) = (b(i)-a(i)) / max(a(i),b(i))", flush=True)
    print(f"\n  {'Layer':>6}  {'Silhouette':>12}  {'NN_Acc':>8}", flush=True)
    print("  " + "-" * 32, flush=True)

    # Labels: condition index (0=clean, 1=fog, 2=night, 3=noise, 4=blur)
    labels = np.array(
        [c_idx for s_idx in range(n_scenes) for c_idx in range(n_conditions)]
    )

    silhouette_per_layer = []
    nn_acc_per_layer = []

    for layer_i in range(n_layers):
        embeddings = [
            all_layers[s_idx][c_idx][layer_i]
            for s_idx in range(n_scenes)
            for c_idx in range(n_conditions)
        ]
        sil = silhouette_score_handwritten(embeddings, labels)
        nn_acc = nearest_neighbor_accuracy(embeddings, labels)
        silhouette_per_layer.append(sil)
        nn_acc_per_layer.append(nn_acc)
        print(f"  {layer_i:>6}  {sil:>12.4f}  {nn_acc:>8.4f}", flush=True)

    best_sil_layer = int(np.argmax(silhouette_per_layer))
    best_nn_layer = int(np.argmax(nn_acc_per_layer))
    print(f"\n  Best silhouette layer: {best_sil_layer}  "
          f"(sil={silhouette_per_layer[best_sil_layer]:.4f})", flush=True)
    print(f"  Best NN-accuracy layer: {best_nn_layer}  "
          f"(acc={nn_acc_per_layer[best_nn_layer]:.4f})", flush=True)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"  n_layers: {n_layers}", flush=True)
    print(f"  n_scenes: {n_scenes}", flush=True)
    print(f"  conditions: {conditions}", flush=True)
    print(f"  Peak divergence layers: {peak_divergence_layers}", flush=True)
    print(f"  Corruption emergence layers (10% threshold): {emergence_layers}", flush=True)
    print(f"  Peak amplification layers: {peak_amplification_layers}", flush=True)
    print(f"  Best silhouette layer: {best_sil_layer}", flush=True)
    print(f"  Best NN-accuracy layer: {best_nn_layer}", flush=True)

    # ------------------------------------------------------------------
    # Save results as valid JSON
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def to_json_safe(v):
        """Recursively convert numpy types and nan/inf to JSON-safe values."""
        if isinstance(v, dict):
            return {k: to_json_safe(val) for k, val in v.items()}
        if isinstance(v, list):
            return [to_json_safe(x) for x in v]
        if isinstance(v, (np.floating, np.integer)):
            v = v.item()
        if isinstance(v, float):
            if v != v:    # nan
                return None
            if v == float("inf") or v == float("-inf"):
                return None
        return v

    output = {
        "experiment": "embedding_dynamics_across_model_depth",
        "experiment_number": 456,
        "timestamp": timestamp,
        "config": {
            "seeds": SEEDS,
            "corruption_types": CORRUPTION_TYPES,
            "severity": SEVERITY,
            "prompt": PROMPT,
            "n_layers": n_layers,
            "n_scenes": n_scenes,
            "n_conditions": n_conditions,
            "conditions": conditions,
            "rsa_max_dim": RSA_MAX_DIM,
        },
        "analysis_1_layer_cosine_similarity": {
            cond: adj_cos_per_layer[cond] for cond in conditions
        },
        "analysis_2_corruption_divergence": divergence_per_layer,
        "analysis_2_peak_divergence_layers": peak_divergence_layers,
        "analysis_2_emergence_layers": emergence_layers,
        "analysis_3_embedding_velocity": {
            cond: velocity_per_layer[cond] for cond in conditions
        },
        "analysis_4_rsa_adjacent_corr": rsa_adjacent_corr,
        "analysis_4_rsa_to_layer0_corr": rsa_to_layer0_corr,
        "analysis_5_clean_variance_per_layer": clean_variance_per_layer,
        "analysis_5_amplification": amplification_per_layer,
        "analysis_5_peak_amplification_layers": peak_amplification_layers,
        "analysis_6_silhouette_per_layer": silhouette_per_layer,
        "analysis_6_nn_accuracy_per_layer": nn_acc_per_layer,
        "analysis_6_best_silhouette_layer": best_sil_layer,
        "analysis_6_best_nn_accuracy_layer": best_nn_layer,
    }

    output = to_json_safe(output)

    output_path = os.path.join(
        RESULTS_DIR, f"embedding_dynamics_{timestamp}.json"
    )
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
