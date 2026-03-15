#!/usr/bin/env python3
"""Experiment 358: Embedding Space Clustering

Can corruption types be clustered in embedding space?
1. K-means clustering of corrupted embeddings
2. Hierarchical clustering dendrogram (cosine linkage)
3. Cluster purity: do corruptions form distinct clusters?
4. Overlap analysis: which corruptions are closest?
5. Severity-dependent cluster separation
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

def kmeans_cosine(embeddings, k, max_iter=100):
    """Simple k-means with cosine distance."""
    n = len(embeddings)
    rng = np.random.RandomState(42)
    # Initialize with random centroids
    indices = rng.choice(n, k, replace=False)
    centroids = [embeddings[i].copy() for i in indices]

    for iteration in range(max_iter):
        # Assign
        assignments = []
        for emb in embeddings:
            dists = [cosine_dist(emb, c) for c in centroids]
            assignments.append(int(np.argmin(dists)))

        # Update centroids
        new_centroids = []
        for ci in range(k):
            members = [embeddings[j] for j in range(n) if assignments[j] == ci]
            if members:
                new_centroids.append(np.mean(members, axis=0))
            else:
                new_centroids.append(centroids[ci])

        # Check convergence
        moved = sum(cosine_dist(c1, c2) for c1, c2 in zip(centroids, new_centroids))
        centroids = new_centroids
        if moved < 1e-10:
            break

    return assignments, centroids

def compute_purity(assignments, labels, k):
    """Compute cluster purity."""
    total = len(assignments)
    correct = 0
    for ci in range(k):
        cluster_labels = [labels[j] for j in range(total) if assignments[j] == ci]
        if cluster_labels:
            most_common = max(set(cluster_labels), key=cluster_labels.count)
            correct += cluster_labels.count(most_common)
    return correct / total

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

    # ========== 1. K-Means Clustering ==========
    print("\n=== K-Means Clustering ===")

    # Combine all embeddings: clean + 4 corruptions
    all_embs = []
    all_labels = []  # 0=clean, 1=fog, 2=night, 3=noise, 4=blur
    label_names = ['clean', 'fog', 'night', 'noise', 'blur']

    for seed in seeds:
        all_embs.append(clean_embs[seed])
        all_labels.append(0)
    for ci, ct in enumerate(ctypes):
        for seed in seeds:
            all_embs.append(corrupt_embs[ct][seed])
            all_labels.append(ci + 1)

    all_embs_arr = np.array(all_embs)

    # Try k=2 (clean vs corrupt), k=4 (corruption types), k=5 (all classes)
    kmeans_results = {}
    for k in [2, 3, 4, 5]:
        assignments, centroids = kmeans_cosine(all_embs, k)
        purity = compute_purity(assignments, all_labels, k)

        # If k=2, check if clean separates from all corrupt
        if k == 2:
            cluster_0_labels = [all_labels[j] for j in range(len(all_labels)) if assignments[j] == 0]
            cluster_1_labels = [all_labels[j] for j in range(len(all_labels)) if assignments[j] == 1]
            clean_in_0 = cluster_0_labels.count(0)
            clean_in_1 = cluster_1_labels.count(0)
            clean_separated = (clean_in_0 == 20 and clean_in_1 == 0) or (clean_in_1 == 20 and clean_in_0 == 0)
        else:
            clean_separated = None

        kmeans_results[str(k)] = {
            'purity': float(purity),
            'clean_separated': clean_separated,
        }
        print(f"  k={k}: purity={purity:.3f}" + (f", clean_separated={clean_separated}" if clean_separated is not None else ""))

    results['kmeans'] = kmeans_results

    # ========== 2. Pairwise Distance Matrix ==========
    print("\n=== Pairwise Class Distances ===")

    class_centroids = {}
    for ci, name in enumerate(label_names):
        mask = [j for j in range(len(all_labels)) if all_labels[j] == ci]
        class_centroids[name] = np.mean([all_embs[j] for j in mask], axis=0)

    pairwise_dists = {}
    for i, n1 in enumerate(label_names):
        for j, n2 in enumerate(label_names):
            if i >= j:
                continue
            d = float(cosine_dist(class_centroids[n1], class_centroids[n2]))
            pairwise_dists[n1 + '_vs_' + n2] = d

    # Find closest and farthest pairs
    closest = min(pairwise_dists.items(), key=lambda x: x[1])
    farthest = max(pairwise_dists.items(), key=lambda x: x[1])

    results['pairwise_distances'] = {
        'distances': pairwise_dists,
        'closest': {'pair': closest[0], 'distance': closest[1]},
        'farthest': {'pair': farthest[0], 'distance': farthest[1]},
    }
    print(f"  Closest: {closest[0]} = {closest[1]:.6f}")
    print(f"  Farthest: {farthest[0]} = {farthest[1]:.6f}")

    # ========== 3. Within-Class vs Between-Class ==========
    print("\n=== Within vs Between Class Distance ===")

    within_class = {}
    for ci, name in enumerate(label_names):
        members = [all_embs[j] for j in range(len(all_labels)) if all_labels[j] == ci]
        pairwise = []
        for a_idx in range(len(members)):
            for b_idx in range(a_idx+1, len(members)):
                pairwise.append(float(cosine_dist(members[a_idx], members[b_idx])))
        within_class[name] = {
            'mean': float(np.mean(pairwise)) if pairwise else 0,
            'max': float(max(pairwise)) if pairwise else 0,
            'std': float(np.std(pairwise)) if pairwise else 0,
        }

    between_class = {}
    for i, n1 in enumerate(label_names):
        for j, n2 in enumerate(label_names):
            if i >= j:
                continue
            members1 = [all_embs[k] for k in range(len(all_labels)) if all_labels[k] == i]
            members2 = [all_embs[k] for k in range(len(all_labels)) if all_labels[k] == j]
            cross_dists = []
            for m1 in members1:
                for m2 in members2:
                    cross_dists.append(float(cosine_dist(m1, m2)))
            between_class[n1 + '_vs_' + n2] = {
                'mean': float(np.mean(cross_dists)),
                'min': float(min(cross_dists)),
            }

    # Compute Fisher's discriminant ratio for each corruption vs clean
    fisher_ratios = {}
    for ct in ctypes:
        within_clean = within_class['clean']['mean']
        within_corrupt = within_class[ct]['mean']
        between = between_class['clean_vs_' + ct]['mean']
        pooled_within = (within_clean + within_corrupt) / 2
        fisher_ratios[ct] = float(between / pooled_within) if pooled_within > 0 else float('inf')

    results['class_separation'] = {
        'within_class': within_class,
        'between_class': between_class,
        'fisher_ratios': fisher_ratios,
    }
    for ct in ctypes:
        print(f"  {ct}: within={within_class[ct]['mean']:.6f}, "
              f"between={between_class['clean_vs_'+ct]['mean']:.6f}, "
              f"Fisher={fisher_ratios[ct]:.2f}")

    # ========== 4. Severity-Dependent Clustering ==========
    print("\n=== Severity-Dependent Clustering ===")

    sev_cluster = {}
    for sev in [0.1, 0.3, 0.5, 0.7, 1.0]:
        sev_embs = []
        sev_labels = []

        for seed in seeds[:10]:
            rng = np.random.RandomState(seed)
            px = rng.randint(50, 200, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(px)

            sev_embs.append(clean_embs[seed])
            sev_labels.append(0)

            for ci, ct in enumerate(ctypes):
                corrupted = apply_corruption(img, ct, sev)
                emb = extract_hidden(model, processor, corrupted, prompt)
                sev_embs.append(emb)
                sev_labels.append(ci + 1)

        assignments, _ = kmeans_cosine(sev_embs, 5)
        purity = compute_purity(assignments, sev_labels, 5)
        sev_cluster[str(sev)] = {'purity': float(purity)}
        print(f"  sev={sev}: k=5 purity={purity:.3f}")

    results['severity_clustering'] = sev_cluster

    # ========== 5. Nearest-Neighbor Classification ==========
    print("\n=== 1-NN Classification ===")

    # Leave-one-out 1-NN
    correct = 0
    total = len(all_labels)
    confusion = {n: {m: 0 for m in label_names} for n in label_names}

    for i in range(total):
        best_dist = float('inf')
        best_label = -1
        for j in range(total):
            if i == j:
                continue
            d = cosine_dist(all_embs[i], all_embs[j])
            if d < best_dist:
                best_dist = d
                best_label = all_labels[j]

        true_name = label_names[all_labels[i]]
        pred_name = label_names[best_label]
        confusion[true_name][pred_name] += 1
        if best_label == all_labels[i]:
            correct += 1

    nn_accuracy = correct / total

    results['nearest_neighbor'] = {
        'accuracy': float(nn_accuracy),
        'confusion': confusion,
        'n_total': total,
        'n_correct': correct,
    }
    print(f"  1-NN accuracy: {nn_accuracy:.3f} ({correct}/{total})")
    for name in label_names:
        row = [str(confusion[name][m]) for m in label_names]
        print(f"    {name}: [{', '.join(row)}]")

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/embedding_clustering_{ts}.json"
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
