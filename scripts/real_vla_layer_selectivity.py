#!/usr/bin/env python3
"""Experiment 264: Layer-Wise Corruption Selectivity
Measures which layers are most selective (discriminative) for specific
corruption types. Tests whether different layers specialize for different
corruptions by computing per-layer AUROC for each corruption type using
diverse-scene calibration.
"""
import torch, json, numpy as np
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor
from datetime import datetime

def extract_hidden(model, processor, image, prompt, layers):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return {l: fwd.hidden_states[l][0, -1, :].float().cpu().numpy() for l in layers}

def cosine_distance(a, b):
    return 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

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

def compute_auroc(id_scores, ood_scores):
    id_scores = np.asarray(id_scores)
    ood_scores = np.asarray(ood_scores)
    n_id, n_ood = len(id_scores), len(ood_scores)
    if n_id == 0 or n_ood == 0: return 0.5
    count = sum(float(np.sum(o > id_scores) + 0.5 * np.sum(o == id_scores)) for o in ood_scores)
    return count / (n_id * n_ood)

print("=" * 60)
print("Experiment 264: Layer-Wise Corruption Selectivity")
print("=" * 60)

print("Loading OpenVLA-7B...")
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", torch_dtype=torch.bfloat16,
    device_map="auto", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
model.eval()

prompt = "In: What action should the robot take to pick up the object?\nOut:"

# Test layers across the full range
layers = [1, 3, 7, 11, 15, 19, 23, 27, 31]
corruptions = ['fog', 'night', 'noise', 'blur']

# Generate 5 diverse scenes
scenes = []
for seed in range(5):
    np.random.seed(seed * 100)
    pixels = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    scenes.append(Image.fromarray(pixels))

# Extract clean embeddings for all scenes
print("Extracting clean embeddings for 5 scenes...")
clean_embeds = {l: [] for l in layers}
for si, scene in enumerate(scenes):
    h = extract_hidden(model, processor, scene, prompt, layers)
    for l in layers:
        clean_embeds[l].append(h[l])

# Compute clean centroid per layer (average across scenes)
centroids = {l: np.mean(clean_embeds[l], axis=0) for l in layers}

# Compute ID distances (each clean scene to centroid)
id_distances = {}
for l in layers:
    id_distances[l] = [cosine_distance(clean_embeds[l][i], centroids[l]) for i in range(5)]

results = {}

for ctype in corruptions:
    print(f"\n--- {ctype} ---")
    results[ctype] = {}

    for sev in [0.5, 1.0]:
        ood_distances = {l: [] for l in layers}

        for si, scene in enumerate(scenes):
            corrupted = apply_corruption(scene, ctype, sev)
            h = extract_hidden(model, processor, corrupted, prompt, layers)
            for l in layers:
                d = cosine_distance(h[l], centroids[l])
                ood_distances[l].append(d)

        layer_aurocs = {}
        layer_sep_ratios = {}
        for l in layers:
            auroc = compute_auroc(id_distances[l], ood_distances[l])
            id_max = max(id_distances[l])
            ood_min = min(ood_distances[l])
            sep = ood_min / (id_max + 1e-15)
            layer_aurocs[l] = auroc
            layer_sep_ratios[l] = sep
            print(f"  L{l} sev={sev}: AUROC={auroc:.3f}, ID_max={id_max:.6f}, OOD_min={ood_min:.6f}, sep={sep:.2f}")

        results[ctype][f'sev_{sev}'] = {
            'aurocs': {str(l): layer_aurocs[l] for l in layers},
            'separation_ratios': {str(l): layer_sep_ratios[l] for l in layers},
            'id_distances': {str(l): [float(x) for x in id_distances[l]] for l in layers},
            'ood_distances': {str(l): [float(x) for x in ood_distances[l]] for l in layers}
        }

# Find most selective layer per corruption
print("\n\n=== SELECTIVITY ANALYSIS ===")
for ctype in corruptions:
    for sev_key in ['sev_0.5', 'sev_1.0']:
        aurocs = results[ctype][sev_key]['aurocs']
        best_layer = max(aurocs, key=aurocs.get)
        worst_layer = min(aurocs, key=aurocs.get)
        print(f"{ctype} {sev_key}: best=L{best_layer} (AUROC={aurocs[best_layer]:.3f}), worst=L{worst_layer} (AUROC={aurocs[worst_layer]:.3f})")

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
out = {
    'experiment': 'layer_selectivity',
    'experiment_number': 264,
    'timestamp': ts,
    'layers': layers,
    'n_scenes': 5,
    'results': results
}

path = f'/workspace/Vizuara-VLA-Research/experiments/layer_selectivity_{ts}.json'
with open(path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {path}")
