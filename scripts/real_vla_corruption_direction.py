#!/usr/bin/env python3
"""Experiment 268: Corruption Direction Consistency
Tests whether each corruption type shifts embeddings in a consistent
direction across different images. If directions are consistent, a
corruption-specific detector (not just distance) could identify
WHICH corruption is present.
"""
import torch, json, numpy as np
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor
from datetime import datetime

def extract_hidden(model, processor, image, prompt, layer=3):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return fwd.hidden_states[layer][0, -1, :].float().cpu().numpy()

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

print("=" * 60)
print("Experiment 268: Corruption Direction Consistency")
print("=" * 60)

print("Loading OpenVLA-7B...")
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", torch_dtype=torch.bfloat16,
    device_map="auto", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
model.eval()

prompt = "In: What action should the robot take to pick up the object?\nOut:"
corruptions = ['fog', 'night', 'noise', 'blur']

# Generate 8 different scenes
scenes = []
for seed in range(8):
    np.random.seed(seed * 100 + 7)
    pixels = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    scenes.append(Image.fromarray(pixels))

results = {}

for layer in [3, 15, 31]:
    print(f"\n=== Layer {layer} ===")
    layer_results = {}

    # Extract clean embeddings
    clean_embeds = []
    for scene in scenes:
        h = extract_hidden(model, processor, scene, prompt, layer=layer)
        clean_embeds.append(h)

    # For each corruption, compute shift vectors
    for ctype in corruptions:
        shifts = []
        for si, scene in enumerate(scenes):
            corrupted = apply_corruption(scene, ctype)
            h_corr = extract_hidden(model, processor, corrupted, prompt, layer=layer)
            shift = h_corr - clean_embeds[si]
            # Normalize shift
            norm = np.linalg.norm(shift)
            if norm > 0:
                shifts.append(shift / norm)

        # Compute pairwise cosine similarity of shift directions
        n = len(shifts)
        sim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                sim_matrix[i, j] = float(np.dot(shifts[i], shifts[j]))

        # Average off-diagonal similarity
        mask = ~np.eye(n, dtype=bool)
        mean_sim = float(sim_matrix[mask].mean())
        std_sim = float(sim_matrix[mask].std())
        min_sim = float(sim_matrix[mask].min())

        layer_results[ctype] = {
            'mean_direction_similarity': mean_sim,
            'std_direction_similarity': std_sim,
            'min_direction_similarity': min_sim,
            'n_scenes': n
        }

        print(f"  {ctype}: direction similarity = {mean_sim:.4f} ± {std_sim:.4f} (min={min_sim:.4f})")

    # Cross-corruption direction similarity
    cross_sims = {}
    for ci, c1 in enumerate(corruptions):
        for cj, c2 in enumerate(corruptions):
            if ci >= cj:
                continue
            # Use scene 0 shifts
            h1 = extract_hidden(model, processor, apply_corruption(scenes[0], c1), prompt, layer=layer)
            h2 = extract_hidden(model, processor, apply_corruption(scenes[0], c2), prompt, layer=layer)
            s1 = h1 - clean_embeds[0]
            s2 = h2 - clean_embeds[0]
            s1 = s1 / (np.linalg.norm(s1) + 1e-10)
            s2 = s2 / (np.linalg.norm(s2) + 1e-10)
            sim = float(np.dot(s1, s2))
            cross_sims[f"{c1}_vs_{c2}"] = sim
            print(f"  Cross: {c1} vs {c2} = {sim:.4f}")

    layer_results['cross_corruption'] = cross_sims
    results[str(layer)] = layer_results

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
out = {
    'experiment': 'corruption_direction',
    'experiment_number': 268,
    'timestamp': ts,
    'n_scenes': 8,
    'results': results
}

path = f'/workspace/Vizuara-VLA-Research/experiments/direction_{ts}.json'
with open(path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {path}")
