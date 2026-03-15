#!/usr/bin/env python3
"""Experiment 290: Embedding Space Topology
Characterizes the topology of corruption embeddings using:
1. Nearest-neighbor graph structure
2. Cluster separability metrics (silhouette, Calinski-Harabasz)
3. Dimensionality estimation (intrinsic dimensionality)
4. Manifold curvature via local PCA
5. Convex hull analysis in PCA space
"""

import torch
import numpy as np
import json
from datetime import datetime
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor
from scipy.spatial.distance import cdist, cosine
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

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
    elif ctype == 'snow':
        rng = np.random.RandomState(42)
        mask = rng.random(arr.shape[:2]) < severity * 0.3
        arr[mask] = 1.0
    elif ctype == 'rain':
        rng = np.random.RandomState(42)
        for _ in range(int(severity * 200)):
            x = rng.randint(0, arr.shape[1])
            y_start = rng.randint(0, arr.shape[0])
            length = rng.randint(5, 20)
            y_end = min(y_start + length, arr.shape[0])
            arr[y_start:y_end, x, :] = arr[y_start:y_end, x, :] * 0.5 + 0.5
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))

def extract_hidden(model, processor, image, prompt, layer=3):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return fwd.hidden_states[layer][0, -1, :].float().cpu().numpy()

def main():
    print("Loading OpenVLA-7B...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    np.random.seed(42)
    pixels = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    base_img = Image.fromarray(pixels)

    results = {
        "experiment": "embedding_topology",
        "experiment_number": 290,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    # Collect embeddings across corruptions and severities
    print("Collecting embeddings...")
    corruptions = ['fog', 'night', 'blur', 'noise', 'snow', 'rain']
    severities = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    embeddings = []
    labels = []
    severity_vals = []

    for c in corruptions:
        for s in severities:
            if s == 0.0:
                img = base_img
            else:
                img = apply_corruption(base_img, c, s)
            emb = extract_hidden(model, processor, img, prompt)
            embeddings.append(emb)
            labels.append(c if s > 0 else 'clean')
            severity_vals.append(s)
            print(f"  {c} sev={s:.1f}: d={float(cosine(embeddings[0], emb)):.6f}")

    X = np.array(embeddings)  # shape: (n_samples, 4096)
    n = len(X)

    # Clean centroid
    clean_emb = X[0]

    # Part 1: PCA analysis
    print("\n=== Part 1: PCA Analysis ===")
    pca_full = PCA(n_components=min(n, 50))
    X_pca = pca_full.fit_transform(X)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)

    dims_for_90 = int(np.searchsorted(cum_var, 0.90)) + 1
    dims_for_95 = int(np.searchsorted(cum_var, 0.95)) + 1
    dims_for_99 = int(np.searchsorted(cum_var, 0.99)) + 1

    results["pca"] = {
        "explained_variance_ratios": pca_full.explained_variance_ratio_[:20].tolist(),
        "cumulative_variance": cum_var[:20].tolist(),
        "dims_for_90pct": dims_for_90,
        "dims_for_95pct": dims_for_95,
        "dims_for_99pct": dims_for_99,
        "total_components": min(n, 50)
    }
    print(f"  Dims for 90%: {dims_for_90}, 95%: {dims_for_95}, 99%: {dims_for_99}")

    # Part 2: Cluster analysis in PCA space
    print("\n=== Part 2: Cluster Analysis ===")
    # Exclude clean duplicates (severity=0 appears 6 times)
    nonclean_mask = [s > 0 for s in severity_vals]
    X_nonclean = X_pca[nonclean_mask][:, :10]
    labels_nonclean = [l for l, m in zip(labels, nonclean_mask) if m]

    if len(set(labels_nonclean)) > 1:
        sil_score = float(silhouette_score(X_nonclean, labels_nonclean))
    else:
        sil_score = 0.0

    # Within-corruption vs between-corruption distances
    corruption_embs = {}
    for c in corruptions:
        idxs = [i for i, (l, s) in enumerate(zip(labels, severity_vals)) if l == c and s > 0]
        corruption_embs[c] = X[idxs]

    within_dists = {}
    between_dists = {}
    for c in corruptions:
        embs_c = corruption_embs[c]
        within_dists[c] = float(np.mean(cdist(embs_c, embs_c, metric='cosine')))

    for i, c1 in enumerate(corruptions):
        for c2 in corruptions[i+1:]:
            key = f"{c1}_vs_{c2}"
            between_dists[key] = float(np.mean(cdist(corruption_embs[c1], corruption_embs[c2], metric='cosine')))

    results["clustering"] = {
        "silhouette_score": sil_score,
        "within_cluster_distances": within_dists,
        "between_cluster_distances": between_dists
    }
    print(f"  Silhouette score: {sil_score:.4f}")

    # Part 3: Nearest neighbor analysis
    print("\n=== Part 3: Nearest Neighbor Graph ===")
    k = 5
    nn = NearestNeighbors(n_neighbors=k+1, metric='cosine')
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    # For each corruption at severity 1.0, what are its nearest neighbors?
    nn_results = {}
    for c in corruptions:
        idx = [i for i, (l, s) in enumerate(zip(labels, severity_vals)) if l == c and s == 1.0][0]
        neighbors = []
        for j in range(1, k+1):
            ni = indices[idx][j]
            neighbors.append({
                "label": labels[ni],
                "severity": severity_vals[ni],
                "distance": float(distances[idx][j])
            })
        nn_results[c] = neighbors
    results["nearest_neighbors"] = nn_results

    # Part 4: Intrinsic dimensionality estimation (correlation dimension)
    print("\n=== Part 4: Intrinsic Dimensionality ===")
    dist_matrix = cdist(X, X, metric='cosine')
    eps_values = np.percentile(dist_matrix[dist_matrix > 0], [10, 20, 30, 40, 50, 60, 70, 80, 90])

    correlation_dims = []
    for eps in eps_values:
        count = np.sum(dist_matrix < eps) - n  # exclude diagonal
        if count > 0:
            correlation_dims.append({
                "epsilon": float(eps),
                "count": int(count),
                "log_count": float(np.log(count)),
                "log_eps": float(np.log(eps))
            })

    # Estimate dimension from slope
    if len(correlation_dims) > 2:
        log_eps = [c["log_eps"] for c in correlation_dims]
        log_counts = [c["log_count"] for c in correlation_dims]
        from numpy.polynomial import polynomial as P
        coeffs = np.polyfit(log_eps, log_counts, 1)
        intrinsic_dim = float(coeffs[0])
    else:
        intrinsic_dim = 0

    results["intrinsic_dimensionality"] = {
        "correlation_dimension": intrinsic_dim,
        "correlation_data": correlation_dims
    }
    print(f"  Estimated intrinsic dimension: {intrinsic_dim:.2f}")

    # Part 5: Local PCA curvature
    print("\n=== Part 5: Local PCA Curvature ===")
    local_dims = {}
    for c in corruptions:
        embs_c = corruption_embs[c]
        if len(embs_c) >= 3:
            local_pca = PCA(n_components=min(len(embs_c), 10))
            local_pca.fit(embs_c)
            cum_var_local = np.cumsum(local_pca.explained_variance_ratio_)
            dim_90 = int(np.searchsorted(cum_var_local, 0.90)) + 1
            local_dims[c] = {
                "dim_90pct": dim_90,
                "variance_ratios": local_pca.explained_variance_ratio_[:5].tolist(),
                "pc1_variance": float(local_pca.explained_variance_ratio_[0])
            }
            print(f"  {c}: 90% variance in {dim_90}D, PC1={local_pca.explained_variance_ratio_[0]:.4f}")
    results["local_pca"] = local_dims

    # Part 6: Severity trajectory linearity
    print("\n=== Part 6: Severity Trajectory Linearity ===")
    linearity = {}
    for c in corruptions:
        idxs = [i for i, (l, s) in enumerate(zip(labels, severity_vals)) if l == c and s > 0]
        if len(idxs) >= 3:
            embs_c = X[idxs]
            sevs = [severity_vals[i] for i in idxs]
            dists = [float(cosine(clean_emb, e)) for e in embs_c]

            # Linear fit
            coeffs = np.polyfit(sevs, dists, 1)
            predicted = np.polyval(coeffs, sevs)
            residuals = np.array(dists) - predicted
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((np.array(dists) - np.mean(dists))**2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            linearity[c] = {
                "r_squared": float(r_squared),
                "slope": float(coeffs[0]),
                "intercept": float(coeffs[1]),
                "severities": sevs,
                "distances": dists
            }
            print(f"  {c}: R²={r_squared:.4f}, slope={coeffs[0]:.6f}")
    results["linearity"] = linearity

    # Save
    ts = results["timestamp"]
    out_path = f"experiments/topology_{ts}.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
