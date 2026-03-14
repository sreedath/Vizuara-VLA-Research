"""
Embedding Space Geometry Analysis.

Analyzes the intrinsic dimensionality, cluster structure, and
geometric properties of ID vs OOD hidden state embeddings. Tests
whether ID and OOD occupy distinct manifolds and measures their
intrinsic dimensionality via PCA eigenspectrum analysis.

Experiment 94 in the CalibDrive series.
"""
import os
import json
import datetime
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

RESULTS_DIR = "/workspace/Vizuara-VLA-Research/experiments"
os.makedirs(RESULTS_DIR, exist_ok=True)
SIZE = (256, 256)


def create_highway(idx):
    rng = np.random.default_rng(idx * 5001)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]
    img[SIZE[0]//2:] = [80, 80, 80]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [255, 255, 255]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_urban(idx):
    rng = np.random.default_rng(idx * 5002)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [135, 206, 235]
    img[SIZE[0]//3:SIZE[0]//2] = [139, 119, 101]
    img[SIZE[0]//2:] = [60, 60, 60]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_noise(idx):
    rng = np.random.default_rng(idx * 5003)
    return rng.integers(0, 256, (*SIZE, 3), dtype=np.uint8)

def create_indoor(idx):
    rng = np.random.default_rng(idx * 5004)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:] = [200, 180, 160]
    img[SIZE[0]//2:, :] = [100, 80, 60]
    img[:SIZE[0]//3, SIZE[1]//3:2*SIZE[1]//3] = [150, 200, 255]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_twilight_highway(idx):
    rng = np.random.default_rng(idx * 5010)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [70, 50, 80]
    img[SIZE[0]//2:] = [60, 60, 60]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [200, 200, 100]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_snow(idx):
    rng = np.random.default_rng(idx * 5014)
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
    if hasattr(fwd, 'hidden_states') and fwd.hidden_states:
        return fwd.hidden_states[-1][0, -1, :].float().cpu().numpy()
    return None


def cosine_dist(a, b):
    return 1.0 - float(np.dot(a / (np.linalg.norm(a) + 1e-10),
                               b / (np.linalg.norm(b) + 1e-10)))


def intrinsic_dim_pca(embeddings, threshold=0.95):
    """Estimate intrinsic dimensionality as PCA dims to reach threshold variance."""
    pca = PCA(n_components=min(len(embeddings)-1, 50), random_state=42)
    pca.fit(embeddings)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    dim_95 = int(np.searchsorted(cumvar, threshold) + 1)
    return dim_95, pca.explained_variance_ratio_, cumvar


def main():
    print("=" * 70, flush=True)
    print("EMBEDDING SPACE GEOMETRY ANALYSIS", flush=True)
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

    # Collect per-category embeddings
    categories = {
        'highway': (create_highway, 'ID'),
        'urban': (create_urban, 'ID'),
        'noise': (create_noise, 'OOD'),
        'indoor': (create_indoor, 'OOD'),
        'twilight': (create_twilight_highway, 'OOD'),
        'snow': (create_snow, 'OOD'),
    }

    embeddings = {}
    for name, (fn, group) in categories.items():
        print(f"\n  Collecting {name}...", flush=True)
        embs = []
        for i in range(15):
            h = extract_hidden(model, processor,
                              Image.fromarray(fn(i + 500)), prompt)
            if h is not None:
                embs.append(h)
        embeddings[name] = np.array(embs)
        print(f"    {name}: {len(embs)} embeddings, shape {embs[0].shape}", flush=True)

    # 1. Per-category statistics
    print("\n--- Per-category statistics ---", flush=True)
    category_stats = {}
    for name, embs in embeddings.items():
        norms = np.linalg.norm(embs, axis=1)
        # Intra-category pairwise cosine distances
        intra_dists = []
        for i in range(len(embs)):
            for j in range(i+1, len(embs)):
                intra_dists.append(cosine_dist(embs[i], embs[j]))
        intra_dists = np.array(intra_dists)

        category_stats[name] = {
            'n': len(embs),
            'norm_mean': float(norms.mean()),
            'norm_std': float(norms.std()),
            'intra_cos_mean': float(intra_dists.mean()),
            'intra_cos_std': float(intra_dists.std()),
        }
        print(f"  {name:<10}: norm={norms.mean():.1f}±{norms.std():.1f}, "
              f"intra_cos={intra_dists.mean():.4f}±{intra_dists.std():.4f}", flush=True)

    # 2. Inter-category distances
    print("\n--- Inter-category distances ---", flush=True)
    centroids = {name: embs.mean(axis=0) for name, embs in embeddings.items()}
    inter_dists = {}
    for n1 in sorted(centroids.keys()):
        for n2 in sorted(centroids.keys()):
            if n1 < n2:
                d = cosine_dist(centroids[n1], centroids[n2])
                inter_dists[f"{n1}_vs_{n2}"] = float(d)
                print(f"  {n1} vs {n2}: {d:.4f}", flush=True)

    # 3. Intrinsic dimensionality
    print("\n--- Intrinsic dimensionality ---", flush=True)
    id_embs = np.concatenate([embeddings['highway'], embeddings['urban']])
    ood_embs = np.concatenate([embeddings[n] for n in ['noise', 'indoor', 'twilight', 'snow']])
    all_embs = np.concatenate([id_embs, ood_embs])

    dim_results = {}
    for name, emb_set in [('id', id_embs), ('ood', ood_embs), ('all', all_embs)]:
        dim_95, var_ratios, cumvar = intrinsic_dim_pca(emb_set, 0.95)
        dim_90, _, _ = intrinsic_dim_pca(emb_set, 0.90)
        dim_99, _, _ = intrinsic_dim_pca(emb_set, 0.99)
        dim_results[name] = {
            'dim_90': int(dim_90),
            'dim_95': int(dim_95),
            'dim_99': int(dim_99),
            'top_5_var': [float(v) for v in var_ratios[:5]],
            'cumvar_10': float(cumvar[min(9, len(cumvar)-1)]),
        }
        print(f"  {name}: dim_90={dim_90}, dim_95={dim_95}, dim_99={dim_99}", flush=True)

    # 4. ID-OOD separability in PCA space
    print("\n--- PCA separability ---", flush=True)
    pca_seps = {}
    for n_comp in [2, 3, 4, 8, 16]:
        if n_comp > min(all_embs.shape[0]-1, 50):
            continue
        pca = PCA(n_components=n_comp, random_state=42)
        all_pca = pca.fit_transform(all_embs)

        id_pca = all_pca[:len(id_embs)]
        ood_pca = all_pca[len(id_embs):]

        id_centroid = id_pca.mean(axis=0)
        id_scores = [cosine_dist(h, id_centroid) for h in id_pca]
        ood_scores = [cosine_dist(h, id_centroid) for h in ood_pca]
        labels = [0]*len(id_scores) + [1]*len(ood_scores)
        auroc = roc_auc_score(labels, id_scores + ood_scores)

        pca_seps[f'pca_{n_comp}'] = {
            'auroc': float(auroc),
            'explained_var': float(sum(pca.explained_variance_ratio_)),
        }
        print(f"  PCA-{n_comp}: AUROC={auroc:.3f}, var={sum(pca.explained_variance_ratio_):.3f}", flush=True)

    # 5. Cluster compactness ratio (intra/inter)
    print("\n--- Compactness ratio ---", flush=True)
    id_centroid = id_embs.mean(axis=0)
    ood_centroid = ood_embs.mean(axis=0)

    id_intra = np.mean([cosine_dist(h, id_centroid) for h in id_embs])
    ood_intra = np.mean([cosine_dist(h, ood_centroid) for h in ood_embs])
    inter = cosine_dist(id_centroid, ood_centroid)

    compactness = {
        'id_intra': float(id_intra),
        'ood_intra': float(ood_intra),
        'inter': float(inter),
        'ratio_id': float(inter / (id_intra + 1e-10)),
        'ratio_ood': float(inter / (ood_intra + 1e-10)),
    }
    print(f"  ID intra: {id_intra:.4f}, OOD intra: {ood_intra:.4f}, Inter: {inter:.4f}", flush=True)
    print(f"  Ratio (inter/id_intra): {inter/id_intra:.2f}", flush=True)

    # 6. Per-category 2D PCA projection coordinates (for visualization)
    pca_2d = PCA(n_components=2, random_state=42)
    all_2d = pca_2d.fit_transform(all_embs)
    coords_2d = {}
    offset = 0
    for name in ['highway', 'urban', 'noise', 'indoor', 'twilight', 'snow']:
        n = len(embeddings[name])
        coords_2d[name] = all_2d[offset:offset+n].tolist()
        offset += n

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'embedding_geometry',
        'experiment_number': 94,
        'timestamp': timestamp,
        'category_stats': category_stats,
        'inter_category_distances': inter_dists,
        'intrinsic_dimensionality': dim_results,
        'pca_separability': pca_seps,
        'compactness': compactness,
        'coords_2d': coords_2d,
        'pca_2d_explained_var': float(sum(pca_2d.explained_variance_ratio_)),
    }
    output_path = os.path.join(RESULTS_DIR, f"embedding_geometry_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
