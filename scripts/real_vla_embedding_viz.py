"""
Embedding Space Visualization for Paper Figures.

Extracts hidden states from calibration + test images and saves them
for offline PCA/t-SNE visualization. This produces the key geometric
figure showing ID/OOD cluster separation.

Also computes:
1. PCA explained variance ratios
2. Inter-cluster and intra-cluster distances
3. Silhouette scores for ID vs OOD

Experiment 55 in the CalibDrive series.
"""
import os
import json
import datetime
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.decomposition import PCA

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
    img[SIZE[0]//2:] = [139, 90, 43]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_inverted(idx):
    return 255 - create_highway(idx + 3000)

def create_blackout(idx):
    return np.zeros((*SIZE, 3), dtype=np.uint8)

def create_blank(idx):
    rng = np.random.default_rng(idx * 5005)
    val = rng.integers(200, 256)
    return np.full((*SIZE, 3), val, dtype=np.uint8)


def extract_hidden(model, processor, image, prompt):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=7, do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
    if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
        last_step = outputs.hidden_states[-1]
        if isinstance(last_step, tuple):
            hidden = last_step[-1][0, -1, :].float().cpu().numpy()
        else:
            hidden = last_step[0, -1, :].float().cpu().numpy()
    else:
        hidden = np.zeros(4096)
    return hidden


def cosine_dist(a, b):
    return 1.0 - float(np.dot(a / (np.linalg.norm(a) + 1e-10),
                               b / (np.linalg.norm(b) + 1e-10)))


def main():
    print("=" * 70, flush=True)
    print("EMBEDDING SPACE VISUALIZATION DATA", flush=True)
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

    # Collect hidden states from all scenarios
    scenarios = {
        'highway': (create_highway, 'ID', 12),
        'urban': (create_urban, 'ID', 12),
        'noise': (create_noise, 'OOD', 8),
        'indoor': (create_indoor, 'OOD', 8),
        'inverted': (create_inverted, 'OOD', 8),
        'blackout': (create_blackout, 'OOD', 8),
        'blank': (create_blank, 'OOD', 8),
    }

    all_hidden = []
    all_labels = []
    all_scenarios = []
    all_is_ood = []
    cnt = 0
    total = sum(v[2] for v in scenarios.values())

    for scene, (fn, label, n) in scenarios.items():
        for i in range(n):
            cnt += 1
            h = extract_hidden(model, processor,
                               Image.fromarray(fn(i + 200)), prompt)
            all_hidden.append(h)
            all_labels.append(label)
            all_scenarios.append(scene)
            all_is_ood.append(label == 'OOD')
            if cnt % 10 == 0:
                print(f"  [{cnt}/{total}] {scene}_{i}", flush=True)

    hidden_matrix = np.array(all_hidden)
    print(f"\n  Total: {len(all_hidden)} samples, dim={hidden_matrix.shape[1]}", flush=True)

    # PCA
    print("\nComputing PCA...", flush=True)
    pca = PCA(n_components=50)
    pca_50 = pca.fit_transform(hidden_matrix)
    pca_2d = pca_50[:, :2]

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    print(f"  PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% variance", flush=True)
    print(f"  PC1+PC2 explains {cumvar[1]*100:.1f}% variance", flush=True)
    print(f"  Top 10 PCs explain {cumvar[9]*100:.1f}% variance", flush=True)
    print(f"  Top 50 PCs explain {cumvar[49]*100:.1f}% variance", flush=True)

    # Centroid analysis
    centroid = np.mean(hidden_matrix, axis=0)
    id_hidden = hidden_matrix[np.array(all_is_ood) == False]
    ood_hidden = hidden_matrix[np.array(all_is_ood) == True]
    id_centroid = np.mean(id_hidden, axis=0)
    ood_centroid = np.mean(ood_hidden, axis=0)

    # Inter/intra cluster distances
    id_intra = np.mean([cosine_dist(h, id_centroid) for h in id_hidden])
    ood_intra = np.mean([cosine_dist(h, ood_centroid) for h in ood_hidden])
    inter = cosine_dist(id_centroid, ood_centroid)

    print(f"\n  ID intra-cluster distance: {id_intra:.4f}", flush=True)
    print(f"  OOD intra-cluster distance: {ood_intra:.4f}", flush=True)
    print(f"  Inter-cluster distance: {inter:.4f}", flush=True)
    print(f"  Separation ratio (inter/max_intra): {inter/max(id_intra, ood_intra):.2f}", flush=True)

    # Silhouette score
    binary_labels = [0 if not ood else 1 for ood in all_is_ood]
    sil = silhouette_score(pca_50[:, :10], binary_labels)
    print(f"  Silhouette score (10-d PCA): {sil:.3f}", flush=True)

    # Per-scenario centroids in PCA space
    print("\n  Per-scenario PCA centroids:", flush=True)
    scenario_centroids_2d = {}
    for scene in sorted(set(all_scenarios)):
        mask = [s == scene for s in all_scenarios]
        scene_pca = pca_2d[mask]
        centroid_2d = scene_pca.mean(axis=0)
        scenario_centroids_2d[scene] = centroid_2d.tolist()
        is_ood = all_is_ood[all_scenarios.index(scene)]
        marker = "OOD" if is_ood else "ID"
        print(f"    {scene} ({marker}): ({centroid_2d[0]:.2f}, {centroid_2d[1]:.2f})", flush=True)

    # AUROC using cosine to ID centroid
    cos_dists = [cosine_dist(h, id_centroid) for h in hidden_matrix]
    auroc = roc_auc_score(binary_labels, cos_dists)
    print(f"\n  AUROC (cosine to ID centroid): {auroc:.3f}", flush=True)

    # Save data for visualization
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'embedding_viz',
        'experiment_number': 55,
        'timestamp': timestamp,
        'n_samples': len(all_hidden),
        'dim': int(hidden_matrix.shape[1]),
        'pca_2d': pca_2d.tolist(),
        'scenarios': all_scenarios,
        'is_ood': [bool(x) for x in all_is_ood],
        'pca_explained_variance': pca.explained_variance_ratio_[:10].tolist(),
        'cumulative_variance_50': float(cumvar[49]),
        'id_intra_dist': float(id_intra),
        'ood_intra_dist': float(ood_intra),
        'inter_dist': float(inter),
        'silhouette_score': float(sil),
        'auroc': float(auroc),
        'scenario_centroids_2d': scenario_centroids_2d,
    }
    output_path = os.path.join(RESULTS_DIR, f"embedding_viz_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
