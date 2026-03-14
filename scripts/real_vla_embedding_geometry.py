"""
Embedding Geometry Analysis.

Studies the geometric structure of ID and OOD embeddings:
- Intrinsic dimensionality via PCA explained variance
- ID cluster compactness vs OOD dispersion
- Inter- and intra-class distances
- Angular distribution on the hypersphere

Experiment 119 in the CalibDrive series.
"""
import os
import json
import datetime
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA

RESULTS_DIR = "/workspace/Vizuara-VLA-Research/experiments"
os.makedirs(RESULTS_DIR, exist_ok=True)
SIZE = (256, 256)


def create_highway(idx):
    rng = np.random.default_rng(idx * 13001)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]
    img[SIZE[0]//2:] = [80, 80, 80]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [255, 255, 255]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_urban(idx):
    rng = np.random.default_rng(idx * 13002)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [135, 206, 235]
    img[SIZE[0]//3:SIZE[0]//2] = [139, 119, 101]
    img[SIZE[0]//2:] = [60, 60, 60]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_noise(idx):
    rng = np.random.default_rng(idx * 13003)
    return rng.integers(0, 256, (*SIZE, 3), dtype=np.uint8)

def create_indoor(idx):
    rng = np.random.default_rng(idx * 13004)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:] = [200, 180, 160]
    img[SIZE[0]//2:, :] = [100, 80, 60]
    img[:SIZE[0]//3, SIZE[1]//3:2*SIZE[1]//3] = [150, 200, 255]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_twilight_highway(idx):
    rng = np.random.default_rng(idx * 13010)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [70, 50, 80]
    img[SIZE[0]//2:] = [60, 60, 60]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [200, 200, 100]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_snow(idx):
    rng = np.random.default_rng(idx * 13014)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [200, 200, 210]
    img[SIZE[0]//2:] = [220, 220, 230]
    img[SIZE[0]//2:, SIZE[1]//2-2:SIZE[1]//2+2] = [180, 180, 190]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def extract_hidden(model, processor, image, prompt):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    if not hasattr(fwd, 'hidden_states') or not fwd.hidden_states:
        return None
    return fwd.hidden_states[-1][0, -1, :].float().cpu().numpy()


def cosine_dist(a, b):
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def main():
    print("=" * 70, flush=True)
    print("EMBEDDING GEOMETRY ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    from transformers import AutoModelForVision2Seq, AutoProcessor
    print("Loading OpenVLA-7B...", flush=True)
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        "openvla/openvla-7b", trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.", flush=True)

    prompt = "In: What action should the robot take to drive forward at 25 m/s safely?\nOut:"

    categories = {
        'highway': (create_highway, 'ID'),
        'urban': (create_urban, 'ID'),
        'noise': (create_noise, 'OOD'),
        'indoor': (create_indoor, 'OOD'),
        'twilight': (create_twilight_highway, 'OOD'),
        'snow': (create_snow, 'OOD'),
    }

    print("\n--- Collecting embeddings ---", flush=True)
    embeddings = {}
    for cat_name, (fn, group) in categories.items():
        print(f"  {cat_name} ({group})...", flush=True)
        embeds = []
        for i in range(15):
            h = extract_hidden(model, processor, Image.fromarray(fn(i + 2000)), prompt)
            if h is not None:
                embeds.append(h)
        embeddings[cat_name] = {'embeds': np.array(embeds), 'group': group}

    # Collect all embeddings
    all_embeds = np.concatenate([d['embeds'] for d in embeddings.values()], axis=0)
    id_embeds = np.concatenate([d['embeds'] for d in embeddings.values() if d['group'] == 'ID'], axis=0)
    ood_embeds = np.concatenate([d['embeds'] for d in embeddings.values() if d['group'] == 'OOD'], axis=0)

    dim = all_embeds.shape[1]
    print(f"\nTotal: {len(all_embeds)}, ID: {len(id_embeds)}, OOD: {len(ood_embeds)}, Dim: {dim}", flush=True)

    # 1. Intrinsic dimensionality via PCA
    print("\n--- Intrinsic Dimensionality ---", flush=True)
    max_comp = min(len(all_embeds), dim) - 1
    n_pca = min(max_comp, 50)
    pca = PCA(n_components=n_pca)
    pca.fit(all_embeds)
    cumvar = np.cumsum(pca.explained_variance_ratio_)

    # Find dimensionality at various thresholds
    thresholds = [0.5, 0.75, 0.9, 0.95, 0.99]
    intrinsic_dims = {}
    for t in thresholds:
        idx = np.searchsorted(cumvar, t)
        intrinsic_dims[str(t)] = int(idx + 1) if idx < len(cumvar) else n_pca
        print(f"  {t*100:.0f}% variance at {intrinsic_dims[str(t)]} dims", flush=True)

    # ID-only and OOD-only PCA
    pca_id = PCA(n_components=min(len(id_embeds)-1, n_pca))
    pca_id.fit(id_embeds)
    id_cumvar = np.cumsum(pca_id.explained_variance_ratio_)

    pca_ood = PCA(n_components=min(len(ood_embeds)-1, n_pca))
    pca_ood.fit(ood_embeds)
    ood_cumvar = np.cumsum(pca_ood.explained_variance_ratio_)

    for t in [0.9, 0.95]:
        id_idx = np.searchsorted(id_cumvar, t)
        ood_idx = np.searchsorted(ood_cumvar, t)
        print(f"  {t*100:.0f}% var: ID at {id_idx+1} dims, OOD at {ood_idx+1} dims", flush=True)

    # 2. Intra-class distances
    print("\n--- Intra-class Distances ---", flush=True)
    intra_distances = {}
    for cat_name, data in embeddings.items():
        dists = []
        for i in range(len(data['embeds'])):
            for j in range(i+1, len(data['embeds'])):
                dists.append(cosine_dist(data['embeds'][i], data['embeds'][j]))
        intra_distances[cat_name] = {
            'mean': float(np.mean(dists)),
            'std': float(np.std(dists)),
            'max': float(np.max(dists)),
            'group': data['group'],
        }
        print(f"  {cat_name}: mean={np.mean(dists):.4f}, std={np.std(dists):.4f}, max={np.max(dists):.4f}", flush=True)

    # 3. Inter-class distances
    print("\n--- Inter-class Distances ---", flush=True)
    inter_distances = {}
    cat_centroids = {c: np.mean(d['embeds'], axis=0) for c, d in embeddings.items()}
    cats = list(embeddings.keys())
    for i in range(len(cats)):
        for j in range(i+1, len(cats)):
            d = cosine_dist(cat_centroids[cats[i]], cat_centroids[cats[j]])
            key = f"{cats[i]}_vs_{cats[j]}"
            inter_distances[key] = {
                'distance': d,
                'groups': f"{embeddings[cats[i]]['group']}_vs_{embeddings[cats[j]]['group']}",
            }
            print(f"  {key}: {d:.4f} ({inter_distances[key]['groups']})", flush=True)

    # 4. Norm statistics
    print("\n--- Norm Statistics ---", flush=True)
    norm_stats = {}
    for cat_name, data in embeddings.items():
        norms = [float(np.linalg.norm(e)) for e in data['embeds']]
        norm_stats[cat_name] = {
            'mean': float(np.mean(norms)),
            'std': float(np.std(norms)),
            'group': data['group'],
        }
        print(f"  {cat_name}: norm={np.mean(norms):.2f}±{np.std(norms):.2f}", flush=True)

    # 5. Angular analysis
    print("\n--- Angular Distribution ---", flush=True)
    id_centroid = np.mean(id_embeds, axis=0)

    angular_stats = {}
    for cat_name, data in embeddings.items():
        angles = [np.arccos(np.clip(1 - cosine_dist(e, id_centroid), -1, 1)) * 180 / np.pi
                  for e in data['embeds']]
        angular_stats[cat_name] = {
            'mean_angle': float(np.mean(angles)),
            'std_angle': float(np.std(angles)),
            'group': data['group'],
        }
        print(f"  {cat_name}: angle={np.mean(angles):.2f}°±{np.std(angles):.2f}°", flush=True)

    # 6. Cluster compactness ratio
    print("\n--- Cluster Compactness ---", flush=True)
    id_intra = np.mean([intra_distances[c]['mean'] for c in embeddings if embeddings[c]['group'] == 'ID'])
    ood_intra = np.mean([intra_distances[c]['mean'] for c in embeddings if embeddings[c]['group'] == 'OOD'])
    id_ood_inter = cosine_dist(np.mean(id_embeds, axis=0), np.mean(ood_embeds, axis=0))
    compactness = id_ood_inter / (id_intra + 1e-10)
    print(f"  ID intra-distance: {id_intra:.4f}", flush=True)
    print(f"  OOD intra-distance: {ood_intra:.4f}", flush=True)
    print(f"  ID-OOD inter-distance: {id_ood_inter:.4f}", flush=True)
    print(f"  Compactness ratio (inter/intra): {compactness:.2f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'embedding_geometry',
        'experiment_number': 119,
        'timestamp': timestamp,
        'dim': dim,
        'n_id': len(id_embeds),
        'n_ood': len(ood_embeds),
        'intrinsic_dims': intrinsic_dims,
        'pca_cumvar': cumvar[:20].tolist(),
        'id_pca_cumvar': id_cumvar[:20].tolist(),
        'ood_pca_cumvar': ood_cumvar[:20].tolist(),
        'intra_distances': intra_distances,
        'inter_distances': inter_distances,
        'norm_stats': norm_stats,
        'angular_stats': angular_stats,
        'compactness': {
            'id_intra': id_intra,
            'ood_intra': ood_intra,
            'id_ood_inter': id_ood_inter,
            'ratio': compactness,
        },
    }
    output_path = os.path.join(RESULTS_DIR, f"embedding_geometry_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
