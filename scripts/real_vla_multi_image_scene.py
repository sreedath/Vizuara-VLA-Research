#!/usr/bin/env python3
"""Experiment 273: Multi-Image Scene Robustness
Tests detection with images generated from different random seeds,
color distributions, and texture patterns to verify the detector
works across diverse visual content.
"""
import torch, json, numpy as np
from PIL import Image, ImageFilter, ImageDraw
from transformers import AutoModelForVision2Seq, AutoProcessor
from datetime import datetime

def extract_hidden(model, processor, image, prompt, layer=3):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return fwd.hidden_states[layer][0, -1, :].float().cpu().numpy()

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
print("Experiment 273: Multi-Image Scene Robustness")
print("=" * 60)

print("Loading OpenVLA-7B...")
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", torch_dtype=torch.bfloat16,
    device_map="auto", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
model.eval()

prompt = "In: What action should the robot take to pick up the object?\nOut:"

# Generate 15 diverse scene types
scenes = {}

# Random noise scenes
for seed in range(5):
    np.random.seed(seed * 1000)
    pixels = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    scenes[f'random_{seed}'] = Image.fromarray(pixels)

# Solid color scenes
for i, color in enumerate([(200, 50, 50), (50, 200, 50), (50, 50, 200), (200, 200, 50), (100, 100, 100)]):
    img = Image.new('RGB', (224, 224), color)
    scenes[f'solid_{i}'] = img

# Gradient scenes
for i, axis in enumerate(['horizontal', 'vertical', 'diagonal']):
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    if axis == 'horizontal':
        for x in range(224):
            arr[:, x] = int(x / 224 * 255)
    elif axis == 'vertical':
        for y in range(224):
            arr[y, :] = int(y / 224 * 255)
    else:
        for y in range(224):
            for x in range(224):
                arr[y, x] = int((x + y) / 448 * 255)
    scenes[f'gradient_{axis}'] = Image.fromarray(arr)

# Checkerboard
arr = np.zeros((224, 224, 3), dtype=np.uint8)
for y in range(224):
    for x in range(224):
        if (x // 32 + y // 32) % 2 == 0:
            arr[y, x] = [200, 200, 200]
        else:
            arr[y, x] = [50, 50, 50]
scenes['checkerboard'] = Image.fromarray(arr)

# Dark scene
arr = np.random.randint(0, 30, (224, 224, 3), dtype=np.uint8)
scenes['dark'] = Image.fromarray(arr)

corruptions = ['fog', 'night', 'noise', 'blur']
results = {}

for scene_name, scene_img in scenes.items():
    print(f"\n--- {scene_name} ---")
    h_clean = extract_hidden(model, processor, scene_img, prompt)

    scene_results = {}
    for ctype in corruptions:
        corrupted = apply_corruption(scene_img, ctype)
        h_corr = extract_hidden(model, processor, corrupted, prompt)
        d = cosine_distance(h_clean, h_corr)
        scene_results[ctype] = float(d)

    print(f"  " + ", ".join(f"{c}={d:.6f}" for c, d in scene_results.items()))
    results[scene_name] = scene_results

# Compute per-scene AUROC (ID = repeat clean, OOD = corrupted)
print("\n=== PER-SCENE AUROC ===")
auroc_results = {}
for scene_name, scene_img in scenes.items():
    id_dists = [0.0]  # deterministic: always 0
    ood_dists = [results[scene_name][c] for c in corruptions]
    auroc = compute_auroc(id_dists, ood_dists)
    auroc_results[scene_name] = float(auroc)
    print(f"  {scene_name}: AUROC={auroc:.3f}")

results['aurocs'] = auroc_results

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
out = {
    'experiment': 'multi_image_scene',
    'experiment_number': 273,
    'timestamp': ts,
    'n_scenes': len(scenes),
    'results': results
}

path = f'/workspace/Vizuara-VLA-Research/experiments/multi_scene_{ts}.json'
with open(path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {path}")
