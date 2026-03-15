"""
Experiment 234: Corruption Type Clustering
Do different corruption types form distinct clusters in embedding space?
If so, we can not only detect OOD but identify the corruption type.
Tests k-means, silhouette score, and inter/intra-cluster distances.
"""
import torch, json, numpy as np, os
from datetime import datetime
from PIL import Image, ImageFilter

def make_driving_image(w=256, h=256):
    img = Image.new('RGB', (w, h))
    pixels = img.load()
    for y in range(h):
        for x in range(w):
            if y < h // 2:
                b = int(180 + 75 * (1 - y / (h / 2)))
                pixels[x, y] = (100, 150, b)
            else:
                g = int(80 + 40 * ((y - h/2) / (h/2)))
                pixels[x, y] = (g, g + 10, g - 10)
    return img

def apply_corruption(img, name, rng):
    arr = np.array(img, dtype=np.float32)
    if name == 'fog':
        fog = np.full_like(arr, 200)
        arr = arr * 0.4 + fog * 0.6
    elif name == 'night':
        arr = arr * 0.15
    elif name == 'noise':
        arr = arr + rng.normal(0, 30, arr.shape)
    elif name == 'blur':
        return img.filter(ImageFilter.GaussianBlur(radius=5))
    elif name == 'snow':
        snow = rng.random(arr.shape) > 0.97
        arr[snow] = 255
        arr = arr * 0.7 + np.full_like(arr, 200) * 0.3
    elif name == 'rain':
        for _ in range(200):
            x = rng.integers(0, arr.shape[1])
            y_start = rng.integers(0, arr.shape[0] - 20)
            arr[y_start:y_start+20, x, :] = [180, 180, 220]
        arr = arr * 0.85
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def extract_hidden(model, processor, image, prompt, layers):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return {l: fwd.hidden_states[l][0, -1, :].float().cpu().numpy() for l in layers}

def cosine_dist(a, b):
    return 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def kmeans_simple(X, k, max_iter=100, seed=42):
    """Simple k-means implementation (no sklearn dependency)."""
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.choice(n, k, replace=False)
    centroids = X[idx].copy()

    for _ in range(max_iter):
        # Assign
        dists = np.array([[np.linalg.norm(x - c) for c in centroids] for x in X])
        labels = dists.argmin(axis=1)
        # Update
        new_centroids = np.array([X[labels == i].mean(axis=0) if np.sum(labels == i) > 0 else centroids[i] for i in range(k)])
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return labels, centroids

def silhouette_score(X, labels):
    """Compute silhouette score."""
    n = len(X)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0

    scores = []
    for i in range(n):
        same = [j for j in range(n) if labels[j] == labels[i] and j != i]
        if len(same) == 0:
            scores.append(0.0)
            continue
        a_i = np.mean([np.linalg.norm(X[i] - X[j]) for j in same])

        b_i = float('inf')
        for lbl in unique_labels:
            if lbl == labels[i]:
                continue
            others = [j for j in range(n) if labels[j] == lbl]
            if len(others) > 0:
                b_i = min(b_i, np.mean([np.linalg.norm(X[i] - X[j]) for j in others]))

        scores.append((b_i - a_i) / max(a_i, b_i))
    return float(np.mean(scores))

def main():
    print("=" * 60)
    print("Experiment 234: Corruption Type Clustering")
    print("=" * 60)

    from transformers import AutoModelForVision2Seq, AutoProcessor
    print("Loading OpenVLA-7B...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    layers = [3]
    prompt = "In: What action should the robot take to drive forward?\nOut:"
    base_img = make_driving_image()

    # Get centroid
    centroid = extract_hidden(model, processor, base_img, prompt, layers)[3]

    corruption_types = ['fog', 'night', 'noise', 'blur', 'snow', 'rain']
    n_samples = 10  # per corruption type

    # Collect embeddings per type
    embeddings = {}
    for ctype in corruption_types:
        print(f"\n--- {ctype} ---")
        embs = []
        for i in range(n_samples):
            rng = np.random.default_rng(42 + i)
            img = apply_corruption(base_img, ctype, rng)
            h = extract_hidden(model, processor, img, prompt, layers)
            embs.append(h[3])
            if (i+1) % 5 == 0:
                print(f"  Sample {i+1}/{n_samples}")
        embeddings[ctype] = np.array(embs)

    # Compute inter-type and intra-type distances
    print("\n--- Inter/Intra cluster analysis ---")
    cluster_results = {}
    for ctype in corruption_types:
        embs = embeddings[ctype]
        # Intra-cluster: pairwise cosine distances within type
        intra_dists = []
        for i in range(len(embs)):
            for j in range(i+1, len(embs)):
                intra_dists.append(cosine_dist(embs[i], embs[j]))
        cluster_results[ctype] = {
            "intra_mean": round(float(np.mean(intra_dists)), 8) if intra_dists else 0,
            "intra_max": round(float(np.max(intra_dists)), 8) if intra_dists else 0,
            "mean_dist_to_centroid": round(float(np.mean([cosine_dist(e, centroid) for e in embs])), 6),
        }
        print(f"  {ctype}: intra={np.mean(intra_dists):.8f} centroid_dist={cluster_results[ctype]['mean_dist_to_centroid']:.6f}")

    # Inter-cluster distances
    inter_results = {}
    for i, c1 in enumerate(corruption_types):
        for j, c2 in enumerate(corruption_types):
            if i >= j:
                continue
            center1 = embeddings[c1].mean(axis=0)
            center2 = embeddings[c2].mean(axis=0)
            d = cosine_dist(center1, center2)
            inter_results[f"{c1}_vs_{c2}"] = round(d, 8)
            print(f"  {c1} vs {c2}: {d:.8f}")

    # K-means clustering (k = number of corruption types)
    print("\n--- K-means clustering ---")
    all_embs = np.vstack([embeddings[c] for c in corruption_types])
    true_labels = np.concatenate([np.full(n_samples, i) for i in range(len(corruption_types))])

    k = len(corruption_types)
    pred_labels, km_centroids = kmeans_simple(all_embs, k)

    # Silhouette score
    sil = silhouette_score(all_embs, pred_labels)
    print(f"  Silhouette score (k={k}): {sil:.4f}")

    # Also try with true labels
    sil_true = silhouette_score(all_embs, true_labels)
    print(f"  Silhouette score (true labels): {sil_true:.4f}")

    # Confusion matrix: for each predicted cluster, which true type dominates?
    confusion = {}
    for pred_c in range(k):
        mask = pred_labels == pred_c
        true_in_cluster = true_labels[mask]
        type_counts = {}
        for t in true_in_cluster:
            name = corruption_types[t]
            type_counts[name] = type_counts.get(name, 0) + 1
        confusion[f"cluster_{pred_c}"] = type_counts
    print(f"  Cluster composition: {confusion}")

    # Purity: fraction of samples in cluster that belong to dominant type
    purity = 0
    for pred_c in range(k):
        mask = pred_labels == pred_c
        if mask.sum() == 0:
            continue
        true_in_cluster = true_labels[mask]
        most_common = np.bincount(true_in_cluster.astype(int)).max()
        purity += most_common
    purity /= len(all_embs)
    print(f"  Clustering purity: {purity:.4f}")

    # Nearest-centroid classifier accuracy
    print("\n--- Nearest-centroid classifier ---")
    type_centroids = {c: embeddings[c].mean(axis=0) for c in corruption_types}
    correct = 0
    total = 0
    for ctype in corruption_types:
        for emb in embeddings[ctype]:
            dists = {c: cosine_dist(emb, type_centroids[c]) for c in corruption_types}
            predicted = min(dists, key=dists.get)
            if predicted == ctype:
                correct += 1
            total += 1
    nc_accuracy = correct / total
    print(f"  Nearest-centroid accuracy: {nc_accuracy:.4f} ({correct}/{total})")

    # Per-type classification results
    per_type_acc = {}
    for ctype in corruption_types:
        correct_type = 0
        for emb in embeddings[ctype]:
            dists = {c: cosine_dist(emb, type_centroids[c]) for c in corruption_types}
            predicted = min(dists, key=dists.get)
            if predicted == ctype:
                correct_type += 1
        per_type_acc[ctype] = round(correct_type / n_samples, 4)
    print(f"  Per-type accuracy: {per_type_acc}")

    output = {
        "experiment": "clustering",
        "experiment_number": 234,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "layer": 3,
        "n_samples_per_type": n_samples,
        "corruption_types": corruption_types,
        "cluster_results": cluster_results,
        "inter_cluster_distances": inter_results,
        "kmeans_silhouette": round(sil, 4),
        "true_label_silhouette": round(sil_true, 4),
        "kmeans_purity": round(purity, 4),
        "nearest_centroid_accuracy": round(nc_accuracy, 4),
        "per_type_accuracy": per_type_acc,
        "cluster_composition": confusion,
    }

    out_path = f"/workspace/Vizuara-VLA-Research/experiments/clustering_{output['timestamp']}.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
