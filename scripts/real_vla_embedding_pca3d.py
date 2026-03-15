#!/usr/bin/env python3
"""Experiment 276: 3D Embedding Geometry Visualization
Projects clean and corrupted embeddings (4 corruptions × 5 severities + clean)
into 3D PCA space to visualize the geometric structure of the detection problem.
Tests whether corruptions form distinct rays from the clean cluster.
"""
import torch, json, numpy as np
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor
from datetime import datetime

def apply_corruption(img, ctype, severity=1.0):
    arr = np.array(img).astype(np.float32) / 255.0
    if ctype == 'fog':
        arr = arr * (1 - 0.6 * severity) + 0.6 * severity
    elif ctype == 'night':
        arr = arr * max(0.01, 1.0 - 0.95 * severity)
    elif ctype == 'noise':
        arr = arr + np.random.RandomState(42).randn(*arr.shape) * 0.3 * severity
        arr = np.clip(arr, 0, 1)
    elif ctype == 'blur':
        return img.filter(ImageFilter.GaussianBlur(radius=10 * severity))
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))

def extract_hidden(model, processor, image, prompt, layers):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return {l: fwd.hidden_states[l][0, -1, :].float().cpu().numpy() for l in layers}

print("=" * 60)
print("Experiment 276: 3D Embedding Geometry Visualization")
print("=" * 60)

print("Loading OpenVLA-7B...")
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", torch_dtype=torch.bfloat16,
    device_map="auto", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
model.eval()

prompt = "In: What action should the robot take to pick up the object?\nOut:"

# Generate 3 different scene images for richer visualization
scenes = {}
for i, seed in enumerate([42, 123, 456]):
    np.random.seed(seed)
    pixels = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    scenes[f'scene_{i}'] = Image.fromarray(pixels)

corruptions = ['fog', 'night', 'noise', 'blur']
severities = [0.1, 0.3, 0.5, 0.7, 1.0]
layers = [3, 15, 31]

all_embeddings = []
all_labels = []
all_metadata = []

for scene_name, scene_img in scenes.items():
    print(f"\n--- {scene_name} ---")

    # Clean embedding
    h_clean = extract_hidden(model, processor, scene_img, prompt, layers)
    for l in layers:
        all_embeddings.append(h_clean[l])
        all_labels.append(f'{scene_name}_clean_L{l}')
        all_metadata.append({'scene': scene_name, 'corruption': 'clean', 'severity': 0.0, 'layer': l})

    # Corrupted embeddings
    for ctype in corruptions:
        for sev in severities:
            cimg = apply_corruption(scene_img, ctype, sev)
            h_corr = extract_hidden(model, processor, cimg, prompt, layers)
            for l in layers:
                all_embeddings.append(h_corr[l])
                all_labels.append(f'{scene_name}_{ctype}_{sev}_L{l}')
                all_metadata.append({'scene': scene_name, 'corruption': ctype, 'severity': float(sev), 'layer': l})
            print(f"  {ctype} sev={sev:.1f} done")

# PCA per layer
from numpy.linalg import svd

pca_results = {}
for l in layers:
    # Filter embeddings for this layer
    layer_mask = [m['layer'] == l for m in all_metadata]
    layer_embs = np.array([e for e, m in zip(all_embeddings, all_metadata) if m['layer'] == l])
    layer_meta = [m for m in all_metadata if m['layer'] == l]

    # Center the data
    mean_emb = layer_embs.mean(axis=0)
    centered = layer_embs - mean_emb

    # SVD for PCA
    U, S, Vt = svd(centered, full_matrices=False)

    # Project to 3D
    proj_3d = centered @ Vt[:3].T

    # Explained variance
    explained_var = (S[:10] ** 2) / (S ** 2).sum()

    # Compute distances in original space
    clean_embs = {}
    for i, m in enumerate(layer_meta):
        if m['corruption'] == 'clean':
            clean_embs[m['scene']] = layer_embs[i]

    distances = []
    for i, m in enumerate(layer_meta):
        if m['corruption'] != 'clean':
            clean = clean_embs[m['scene']]
            d = 1.0 - np.dot(layer_embs[i], clean) / (np.linalg.norm(layer_embs[i]) * np.linalg.norm(clean))
            distances.append({'label': f"{m['scene']}_{m['corruption']}_{m['severity']}", 'distance': float(d)})

    # Pairwise cosine similarity between corruption directions
    direction_sims = {}
    for scene_name in scenes:
        clean = clean_embs[scene_name]
        dirs = {}
        for ctype in corruptions:
            # Get max severity embedding
            for i, m in enumerate(layer_meta):
                if m['scene'] == scene_name and m['corruption'] == ctype and m['severity'] == 1.0:
                    diff = layer_embs[i] - clean
                    dirs[ctype] = diff / (np.linalg.norm(diff) + 1e-10)
                    break

        # Pairwise cosine similarities between corruption directions
        for c1 in corruptions:
            for c2 in corruptions:
                if c1 < c2 and c1 in dirs and c2 in dirs:
                    sim = float(np.dot(dirs[c1], dirs[c2]))
                    direction_sims[f'{scene_name}_{c1}_vs_{c2}'] = sim

    pca_results[f'L{l}'] = {
        'explained_variance_top10': [float(v) for v in explained_var],
        'cumulative_var_3d': float(explained_var[:3].sum()),
        'projections': [
            {
                'label': f"{m['scene']}_{m['corruption']}_{m['severity']}",
                'x': float(proj_3d[i, 0]),
                'y': float(proj_3d[i, 1]),
                'z': float(proj_3d[i, 2]),
                'corruption': m['corruption'],
                'severity': float(m['severity']),
                'scene': m['scene']
            }
            for i, m in enumerate(layer_meta)
        ],
        'distances': distances,
        'direction_similarities': direction_sims
    }

    print(f"\nL{l}: top-3 explained variance = {explained_var[:3].sum():.3f}")
    print(f"  Variance per PC: {explained_var[:5]}")

# Check if corruptions form distinct rays
print("\n=== CORRUPTION DIRECTION ANALYSIS ===")
for l in layers:
    print(f"\nLayer {l}:")
    for key, sim in pca_results[f'L{l}']['direction_similarities'].items():
        print(f"  {key}: cos_sim = {sim:.4f}")

# Check severity scaling (do embeddings move linearly along rays?)
print("\n=== SEVERITY-DISTANCE LINEARITY ===")
linearity_results = {}
for l in layers:
    layer_meta_l = [m for m in all_metadata if m['layer'] == l]
    layer_embs_l = np.array([e for e, m in zip(all_embeddings, all_metadata) if m['layer'] == l])

    clean_embs_l = {}
    for i, m in enumerate(layer_meta_l):
        if m['corruption'] == 'clean':
            clean_embs_l[m['scene']] = layer_embs_l[i]

    for scene_name in scenes:
        for ctype in corruptions:
            sevs_list = []
            dists_list = []
            for i, m in enumerate(layer_meta_l):
                if m['scene'] == scene_name and m['corruption'] == ctype:
                    clean = clean_embs_l[m['scene']]
                    d = 1.0 - np.dot(layer_embs_l[i], clean) / (np.linalg.norm(layer_embs_l[i]) * np.linalg.norm(clean))
                    sevs_list.append(m['severity'])
                    dists_list.append(float(d))
            if len(sevs_list) >= 3:
                corr = float(np.corrcoef(sevs_list, dists_list)[0, 1])
                linearity_results[f'{scene_name}_{ctype}_L{l}'] = {
                    'correlation': corr,
                    'severities': sevs_list,
                    'distances': dists_list
                }
                print(f"  {scene_name} {ctype} L{l}: r = {corr:.4f}")

pca_results['linearity'] = linearity_results

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
out = {
    'experiment': 'embedding_pca_3d',
    'experiment_number': 276,
    'timestamp': ts,
    'n_scenes': len(scenes),
    'n_corruptions': len(corruptions),
    'n_severities': len(severities),
    'layers': layers,
    'results': pca_results
}

path = f'/workspace/Vizuara-VLA-Research/experiments/pca3d_{ts}.json'
with open(path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {path}")
