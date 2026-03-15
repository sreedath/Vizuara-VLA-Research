#!/usr/bin/env python3
"""Experiment 309: Embedding Space Visualization & t-SNE/UMAP Analysis
Provides comprehensive embedding space characterization:
1. PCA projection with severity gradient
2. t-SNE clustering quality
3. Corruption centroid geometry (angles, distances)
4. Embedding variance analysis across conditions
5. Nearest-neighbor classification accuracy
"""

import torch
import numpy as np
import json
from datetime import datetime
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor
from scipy.spatial.distance import cosine, pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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
    id_s = np.asarray(id_scores)
    ood_s = np.asarray(ood_scores)
    n_id, n_ood = len(id_s), len(ood_s)
    if n_id == 0 or n_ood == 0: return 0.5
    count = sum(float(np.sum(o > id_s) + 0.5 * np.sum(o == id_s)) for o in ood_s)
    return count / (n_id * n_ood)

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
        "experiment": "embedding_visualization",
        "experiment_number": 309,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    corruptions = ['fog', 'night', 'blur', 'noise']
    severities = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

    # Collect embeddings across all conditions
    print("Collecting embeddings...")
    embeddings = []
    labels = []
    severity_vals = []

    # Clean samples
    for _ in range(5):
        emb = extract_hidden(model, processor, base_img, prompt)
        embeddings.append(emb)
        labels.append('clean')
        severity_vals.append(0.0)

    # Corrupted samples
    for c in corruptions:
        for sev in severities:
            corrupted = apply_corruption(base_img, c, sev)
            emb = extract_hidden(model, processor, corrupted, prompt)
            embeddings.append(emb)
            labels.append(c)
            severity_vals.append(sev)

    embeddings = np.array(embeddings)
    print(f"  Collected {len(embeddings)} embeddings (shape: {embeddings.shape})")

    # Part 1: PCA analysis
    print("\n=== Part 1: PCA Analysis ===")
    pca = PCA(n_components=10)
    pca_coords = pca.fit_transform(embeddings)

    pca_by_condition = {}
    for condition in ['clean'] + corruptions:
        mask = [l == condition for l in labels]
        coords = pca_coords[mask]
        sevs = [severity_vals[i] for i, m in enumerate(mask) if m]
        pca_by_condition[condition] = {
            "coords": [{"pc1": float(coords[j, 0]), "pc2": float(coords[j, 1]),
                        "pc3": float(coords[j, 2]), "severity": sevs[j]}
                       for j in range(len(coords))]
        }

    results["pca"] = {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
        "by_condition": pca_by_condition,
    }

    for condition in ['clean'] + corruptions:
        mask = [l == condition for l in labels]
        coords = pca_coords[mask]
        print(f"  {condition}: PC1 range [{coords[:, 0].min():.4f}, {coords[:, 0].max():.4f}], "
              f"PC2 range [{coords[:, 1].min():.4f}, {coords[:, 1].max():.4f}]")

    print(f"  Explained variance: PC1={pca.explained_variance_ratio_[0]*100:.1f}%, "
          f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%, "
          f"PC3={pca.explained_variance_ratio_[2]*100:.1f}%")

    # Part 2: t-SNE clustering
    print("\n=== Part 2: t-SNE Clustering ===")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(embeddings)-1))
    tsne_coords = tsne.fit_transform(embeddings)

    condition_centroids_tsne = {}
    for condition in ['clean'] + corruptions:
        mask = [l == condition for l in labels]
        condition_centroids_tsne[condition] = tsne_coords[mask].mean(axis=0)

    intra_distances = {}
    inter_distances = {}
    for condition in ['clean'] + corruptions:
        mask = [l == condition for l in labels]
        pts = tsne_coords[mask]
        centroid = condition_centroids_tsne[condition]
        intra_distances[condition] = float(np.mean(np.linalg.norm(pts - centroid, axis=1)))

        other_centroids = [condition_centroids_tsne[c2] for c2 in condition_centroids_tsne if c2 != condition]
        if other_centroids:
            inter_distances[condition] = float(np.min([np.linalg.norm(centroid - oc)
                                                        for oc in other_centroids]))

    results["tsne"] = {
        "intra_distances": intra_distances,
        "inter_distances": inter_distances,
    }

    for condition in ['clean'] + corruptions:
        print(f"  {condition}: intra={intra_distances[condition]:.2f}, "
              f"inter={inter_distances.get(condition, 0):.2f}")

    # Part 3: Corruption centroid geometry
    print("\n=== Part 3: Centroid Geometry ===")
    clean_emb = embeddings[0]

    corruption_centroids = {}
    for c in corruptions:
        mask = [l == c for l in labels]
        corruption_centroids[c] = embeddings[mask].mean(axis=0)

    directions = {}
    for c in corruptions:
        diff = corruption_centroids[c] - clean_emb
        diff_norm = np.linalg.norm(diff)
        if diff_norm > 0:
            directions[c] = diff / diff_norm

    angle_matrix = {}
    for c1 in corruptions:
        for c2 in corruptions:
            if c1 < c2:
                cos_sim = float(np.dot(directions[c1], directions[c2]))
                angle_rad = np.arccos(np.clip(cos_sim, -1, 1))
                angle_deg = float(np.degrees(angle_rad))
                angle_matrix[f"{c1}_vs_{c2}"] = {
                    "angle_degrees": angle_deg,
                    "cosine_similarity": cos_sim,
                }

    centroid_distances = {c: float(cosine(clean_emb, corruption_centroids[c])) for c in corruptions}

    results["centroid_geometry"] = {
        "angles": angle_matrix,
        "distances": centroid_distances,
    }

    for pair, info in angle_matrix.items():
        print(f"  {pair}: {info['angle_degrees']:.1f}° (cos_sim={info['cosine_similarity']:.4f})")
    for c, d in centroid_distances.items():
        print(f"  {c} centroid distance: {d:.6f}")

    # Part 4: Within-condition variance
    print("\n=== Part 4: Within-Condition Variance ===")
    variance_analysis = {}

    for condition in ['clean'] + corruptions:
        mask = [l == condition for l in labels]
        pts = embeddings[mask]

        if len(pts) > 1:
            pairwise = []
            for i in range(len(pts)):
                for j in range(i+1, len(pts)):
                    pairwise.append(float(cosine(pts[i], pts[j])))
            variance_analysis[condition] = {
                "n_samples": len(pts),
                "mean_pairwise_cosine": float(np.mean(pairwise)),
                "max_pairwise_cosine": float(np.max(pairwise)),
                "std_pairwise_cosine": float(np.std(pairwise)),
            }
        else:
            variance_analysis[condition] = {"n_samples": len(pts), "mean_pairwise_cosine": 0}

        v = variance_analysis[condition]
        print(f"  {condition}: n={v['n_samples']}, "
              f"mean_dist={v.get('mean_pairwise_cosine', 0):.8f}")

    results["variance"] = variance_analysis

    # Part 5: Nearest-neighbor classification
    print("\n=== Part 5: Nearest-Neighbor Classification ===")
    correct = 0
    total = 0
    per_class = {c: {"correct": 0, "total": 0} for c in ['clean'] + corruptions}

    for i in range(len(embeddings)):
        min_dist = float('inf')
        nn_label = None
        for j in range(len(embeddings)):
            if i == j:
                continue
            d = float(cosine(embeddings[i], embeddings[j]))
            if d < min_dist:
                min_dist = d
                nn_label = labels[j]

        true_label = labels[i]
        is_correct = nn_label == true_label
        if is_correct:
            correct += 1
        total += 1
        per_class[true_label]["total"] += 1
        if is_correct:
            per_class[true_label]["correct"] += 1

    accuracy = correct / total
    nn_results = {
        "overall_accuracy": accuracy,
        "per_class": {c: v["correct"] / v["total"] if v["total"] > 0 else 0
                      for c, v in per_class.items()},
    }
    results["nn_classification"] = nn_results

    print(f"  Overall 1-NN accuracy: {accuracy*100:.1f}%")
    for c, acc in nn_results["per_class"].items():
        print(f"  {c}: {acc*100:.1f}%")

    # Part 6: Embedding norm distribution
    print("\n=== Part 6: Norm Distribution ===")
    norm_dist = {}
    for condition in ['clean'] + corruptions:
        mask = [l == condition for l in labels]
        norms = [float(np.linalg.norm(embeddings[i])) for i, m in enumerate(mask) if m]
        norm_dist[condition] = {
            "mean": float(np.mean(norms)),
            "std": float(np.std(norms)),
            "min": float(min(norms)),
            "max": float(max(norms)),
        }
        print(f"  {condition}: norm={np.mean(norms):.2f}±{np.std(norms):.2f}")

    results["norm_distribution"] = norm_dist

    # Save
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(v) for v in obj]
        return obj

    ts = results["timestamp"]
    out_path = f"experiments/embedding_viz_{ts}.json"
    with open(out_path, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
