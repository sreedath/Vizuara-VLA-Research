#!/usr/bin/env python3
"""Experiment 398: Embedding Space Clustering Analysis (Enhanced)

Examines whether corruption types form distinct clusters in the embedding space
and whether unsupervised clustering can separate clean from corrupted embeddings.

Tests:
1. K-means clustering (k=2: clean vs corrupt, k=5: per-corruption)
2. DBSCAN density-based clustering
3. Silhouette scores for cluster quality
4. Inter/intra-cluster distances
5. Corruption type separability (pairwise)
6. Severity gradient within clusters
"""

import json, time, os, sys
import numpy as np
import torch
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor
from collections import Counter

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

def kmeans_simple(X, k, max_iter=100, seed=42):
    """Simple k-means implementation (no sklearn dependency)."""
    rng = np.random.RandomState(seed)
    n = len(X)
    # k-means++ init
    centers = [X[rng.randint(n)]]
    for _ in range(1, k):
        dists = np.array([min(np.sum((x - c)**2) for c in centers) for x in X])
        probs = dists / dists.sum()
        centers.append(X[rng.choice(n, p=probs)])
    centers = np.array(centers)

    for _ in range(max_iter):
        labels = np.array([np.argmin([np.sum((x - c)**2) for c in centers]) for x in X])
        new_centers = np.array([X[labels == j].mean(axis=0) if np.sum(labels == j) > 0
                                else centers[j] for j in range(k)])
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
    return labels, centers

def dbscan_simple(X, eps, min_samples=2):
    """Simple DBSCAN implementation."""
    n = len(X)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt(np.sum((X[i] - X[j])**2))
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    labels = np.full(n, -1)
    cluster_id = 0
    visited = set()

    for i in range(n):
        if i in visited:
            continue
        visited.add(i)
        neighbors = np.where(dist_matrix[i] <= eps)[0]
        if len(neighbors) < min_samples:
            continue
        labels[i] = cluster_id
        seed_set = list(neighbors)
        j = 0
        while j < len(seed_set):
            q = seed_set[j]
            if q not in visited:
                visited.add(q)
                q_neighbors = np.where(dist_matrix[q] <= eps)[0]
                if len(q_neighbors) >= min_samples:
                    seed_set.extend([x for x in q_neighbors if x not in visited])
            if labels[q] == -1:
                labels[q] = cluster_id
            j += 1
        cluster_id += 1
    return labels

def silhouette_score(X, labels):
    """Compute mean silhouette score."""
    unique = np.unique(labels[labels >= 0])
    if len(unique) < 2:
        return 0.0
    scores = []
    for i in range(len(X)):
        if labels[i] < 0:
            continue
        same = X[labels == labels[i]]
        if len(same) <= 1:
            scores.append(0.0)
            continue
        a = np.mean([np.sqrt(np.sum((X[i] - s)**2)) for s in same if not np.array_equal(s, X[i])])
        b = float('inf')
        for c in unique:
            if c == labels[i]:
                continue
            other = X[labels == c]
            b = min(b, np.mean([np.sqrt(np.sum((X[i] - o)**2)) for o in other]))
        scores.append((b - a) / max(a, b))
    return float(np.mean(scores)) if scores else 0.0

def main():
    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    corruptions = ['fog', 'night', 'noise', 'blur']
    severities = [0.3, 0.5, 0.7, 1.0]

    # Generate multiple scenes
    scenes = []
    for seed in [42, 123, 456, 789, 999]:
        scenes.append(Image.fromarray(
            np.random.RandomState(seed).randint(0, 255, (224, 224, 3), dtype=np.uint8)))

    # Collect embeddings
    print("Extracting embeddings...")
    embeddings = []
    labels_true = []  # 0=clean, 1=fog, 2=night, 3=noise, 4=blur
    label_names = ['clean', 'fog', 'night', 'noise', 'blur']

    for si, scene in enumerate(scenes):
        print(f"  Scene {si+1}/5")
        # Clean
        emb = extract_hidden(model, processor, scene, prompt)
        embeddings.append(emb)
        labels_true.append(0)

        # Corrupted
        for ci, c in enumerate(corruptions):
            for sev in severities:
                corrupted = apply_corruption(scene, c, sev)
                emb = extract_hidden(model, processor, corrupted, prompt)
                embeddings.append(emb)
                labels_true.append(ci + 1)

    X = np.array(embeddings)
    labels_true = np.array(labels_true)
    n_total = len(X)
    n_clean = np.sum(labels_true == 0)
    n_corrupt = n_total - n_clean
    print(f"Total embeddings: {n_total} ({n_clean} clean, {n_corrupt} corrupted)")

    # PCA to manageable dimensions for clustering
    X_centered = X - X.mean(axis=0)
    cov = np.cov(X_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]

    results = {}

    # === Test 1: K-means binary (clean vs corrupt) ===
    print("\n=== K-means Binary Clustering ===")
    for n_dims in [2, 5, 10, 50]:
        X_proj = X_centered @ eigvecs[:, :n_dims]
        km_labels, km_centers = kmeans_simple(X_proj, k=2)

        c0_count = np.sum(km_labels == 0)
        c1_count = np.sum(km_labels == 1)

        clean_in_0 = np.sum((km_labels == 0) & (labels_true == 0))
        clean_in_1 = np.sum((km_labels == 1) & (labels_true == 0))

        if clean_in_0 / max(1, c0_count) > clean_in_1 / max(1, c1_count):
            pred_clean = 0
        else:
            pred_clean = 1

        pred_binary = (km_labels != pred_clean).astype(int)
        true_binary = (labels_true > 0).astype(int)
        accuracy = float(np.mean(pred_binary == true_binary))

        sil = silhouette_score(X_proj, km_labels)

        print(f"  {n_dims}D: accuracy={accuracy:.3f}, silhouette={sil:.4f}")
        results[f"kmeans_binary_{n_dims}d"] = {
            "accuracy": accuracy,
            "silhouette": sil,
            "cluster_sizes": [int(c0_count), int(c1_count)]
        }

    # === Test 2: K-means per-corruption (k=5) ===
    print("\n=== K-means 5-class Clustering ===")
    for n_dims in [5, 10, 50]:
        X_proj = X_centered @ eigvecs[:, :n_dims]
        km_labels, km_centers = kmeans_simple(X_proj, k=5)

        total_correct = 0
        for cluster in range(5):
            mask = km_labels == cluster
            if np.sum(mask) == 0:
                continue
            most_common = Counter(labels_true[mask]).most_common(1)[0][1]
            total_correct += most_common
        purity = total_correct / n_total

        sil = silhouette_score(X_proj, km_labels)

        print(f"  {n_dims}D: purity={purity:.3f}, silhouette={sil:.4f}")
        results[f"kmeans_5class_{n_dims}d"] = {
            "purity": float(purity),
            "silhouette": sil
        }

    # === Test 3: DBSCAN on low-dim projections ===
    print("\n=== DBSCAN Clustering ===")
    for n_dims in [2, 5, 10]:
        X_proj = X_centered @ eigvecs[:, :n_dims]
        all_dists = []
        for i in range(min(50, n_total)):
            for j in range(i+1, min(50, n_total)):
                all_dists.append(np.sqrt(np.sum((X_proj[i] - X_proj[j])**2)))
        median_dist = np.median(all_dists)

        for eps_mult in [0.3, 0.5, 0.7]:
            eps = median_dist * eps_mult
            db_labels = dbscan_simple(X_proj, eps=eps, min_samples=3)
            n_clusters = len(set(db_labels) - {-1})
            n_noise = int(np.sum(db_labels == -1))

            if n_clusters >= 2:
                sil = silhouette_score(X_proj, db_labels)
            else:
                sil = 0.0

            print(f"  {n_dims}D eps={eps_mult}: {n_clusters} clusters, {n_noise} noise, sil={sil:.4f}")
            results[f"dbscan_{n_dims}d_eps{eps_mult}"] = {
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "silhouette": sil,
                "eps": float(eps)
            }

    # === Test 4: Inter/Intra cluster distances ===
    print("\n=== Inter/Intra Cluster Distances ===")
    cluster_centers = {}
    for label in range(5):
        mask = labels_true == label
        cluster_centers[label] = X[mask].mean(axis=0)

    intra_dists = {}
    for label in range(5):
        mask = labels_true == label
        dists = [cosine_dist(x, cluster_centers[label]) for x in X[mask]]
        intra_dists[label_names[label]] = {
            "mean": float(np.mean(dists)),
            "std": float(np.std(dists)),
            "max": float(np.max(dists))
        }
        print(f"  {label_names[label]} intra: mean={np.mean(dists):.6f}, max={np.max(dists):.6f}")

    inter_dists = {}
    for i in range(5):
        for j in range(i+1, 5):
            d = cosine_dist(cluster_centers[i], cluster_centers[j])
            key = f"{label_names[i]}_vs_{label_names[j]}"
            inter_dists[key] = float(d)
            print(f"  {key}: {d:.6f}")

    # Separation ratio
    separation_ratios = {}
    for label in range(1, 5):
        inter = cosine_dist(cluster_centers[0], cluster_centers[label])
        intra_clean = intra_dists['clean']['max']
        intra_corrupt = intra_dists[label_names[label]]['max']
        ratio = inter / max(intra_clean + intra_corrupt, 1e-12)
        separation_ratios[label_names[label]] = float(ratio)
        print(f"  clean-{label_names[label]} separation ratio: {ratio:.2f}")

    results["intra_cluster_distances"] = intra_dists
    results["inter_cluster_distances"] = inter_dists
    results["separation_ratios"] = separation_ratios

    # === Test 5: Pairwise corruption separability ===
    print("\n=== Pairwise Corruption Separability ===")
    pairwise = {}
    for i in range(1, 5):
        for j in range(i+1, 5):
            embs_i = X[labels_true == i]
            embs_j = X[labels_true == j]

            center_i = embs_i.mean(axis=0)
            center_j = embs_j.mean(axis=0)
            midpoint = (center_i + center_j) / 2
            direction = center_j - center_i
            dir_norm = np.linalg.norm(direction)
            if dir_norm > 1e-12:
                direction = direction / dir_norm

            proj_i = [np.dot(e - midpoint, direction) for e in embs_i]
            proj_j = [np.dot(e - midpoint, direction) for e in embs_j]

            correct = sum(1 for p in proj_i if p < 0) + sum(1 for p in proj_j if p >= 0)
            total = len(proj_i) + len(proj_j)
            accuracy = correct / total

            centroid_dist = cosine_dist(center_i, center_j)

            key = f"{label_names[i]}_vs_{label_names[j]}"
            pairwise[key] = {
                "linear_accuracy": float(accuracy),
                "centroid_cosine_dist": float(centroid_dist)
            }
            print(f"  {key}: acc={accuracy:.3f}, cos_dist={centroid_dist:.6f}")

    results["pairwise_separability"] = pairwise

    # === Test 6: Severity gradient within clusters ===
    print("\n=== Severity Gradient Analysis ===")
    severity_gradients = {}
    for ci, c in enumerate(corruptions):
        dists_by_sev = {}
        for sev in severities:
            sev_embs = []
            for si in range(len(scenes)):
                idx = si * (1 + len(corruptions) * len(severities)) + 1 + ci * len(severities) + severities.index(sev)
                sev_embs.append(X[idx])

            clean_center = cluster_centers[0]
            dists = [cosine_dist(e, clean_center) for e in sev_embs]
            dists_by_sev[str(sev)] = {
                "mean": float(np.mean(dists)),
                "std": float(np.std(dists))
            }

        means = [dists_by_sev[str(s)]["mean"] for s in severities]
        monotonic = all(means[i] <= means[i+1] for i in range(len(means)-1))

        severity_gradients[c] = {
            "by_severity": dists_by_sev,
            "monotonic": monotonic,
            "gradient": float(means[-1] - means[0])
        }
        grad_str = " -> ".join(f"{m:.6f}" for m in means)
        print(f"  {c}: {grad_str} (monotonic={monotonic})")

    results["severity_gradients"] = severity_gradients
    results["n_embeddings"] = n_total
    results["n_scenes"] = len(scenes)
    results["n_corruptions"] = len(corruptions)
    results["n_severities"] = len(severities)

    # Save
    out_path = "/workspace/Vizuara-VLA-Research/experiments/embedding_clustering_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
