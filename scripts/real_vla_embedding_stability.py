#!/usr/bin/env python3
"""Experiment 265: Embedding Stability Under Input Perturbation
Tests how stable the embeddings are to tiny, semantically-irrelevant
perturbations (1-bit, 1-pixel changes) vs actual corruption.
Quantifies the model's input sensitivity threshold.
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

def cosine_distance(a, b):
    return 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

print("=" * 60)
print("Experiment 265: Embedding Stability Under Input Perturbation")
print("=" * 60)

print("Loading OpenVLA-7B...")
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", torch_dtype=torch.bfloat16,
    device_map="auto", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
model.eval()

prompt = "In: What action should the robot take to pick up the object?\nOut:"
np.random.seed(42)
pixels = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
img = Image.fromarray(pixels)

h_clean = extract_hidden(model, processor, img, prompt)

results = {}

# 1. Single pixel change
print("\n--- Single pixel changes ---")
pixel_distances = []
for i in range(20):
    arr = pixels.copy()
    r, c = np.random.randint(0, 224, 2)
    ch = np.random.randint(0, 3)
    arr[r, c, ch] = min(255, arr[r, c, ch] + 1)  # +1 to one channel
    perturbed = Image.fromarray(arr)
    h = extract_hidden(model, processor, perturbed, prompt)
    d = cosine_distance(h_clean, h)
    pixel_distances.append(d)
    if i < 5:
        print(f"  1-pixel change {i}: d={d:.10f}")

results['single_pixel'] = {
    'mean': float(np.mean(pixel_distances)),
    'max': float(np.max(pixel_distances)),
    'min': float(np.min(pixel_distances)),
    'std': float(np.std(pixel_distances)),
    'n_zero': int(sum(1 for d in pixel_distances if d == 0.0))
}
print(f"  Mean: {np.mean(pixel_distances):.10f}, Max: {np.max(pixel_distances):.10f}")
print(f"  Exactly zero: {results['single_pixel']['n_zero']}/20")

# 2. N-pixel random changes (1, 5, 10, 50, 100, 500, 1000, 5000)
print("\n--- N-pixel random changes ---")
n_pixels_list = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000]
npixel_results = {}
for n_px in n_pixels_list:
    distances = []
    for trial in range(5):
        arr = pixels.copy()
        for _ in range(n_px):
            r, c = np.random.randint(0, 224, 2)
            ch = np.random.randint(0, 3)
            arr[r, c, ch] = np.random.randint(0, 256)
        perturbed = Image.fromarray(arr)
        h = extract_hidden(model, processor, perturbed, prompt)
        d = cosine_distance(h_clean, h)
        distances.append(d)
    mean_d = float(np.mean(distances))
    npixel_results[n_px] = {
        'mean': mean_d,
        'std': float(np.std(distances)),
        'distances': [float(x) for x in distances]
    }
    print(f"  {n_px:6d} pixels changed: d={mean_d:.8f}")

results['n_pixel'] = npixel_results

# 3. Uniform brightness shift (+1, +2, +5, +10, +20, +50)
print("\n--- Uniform brightness shift ---")
brightness_results = {}
for shift in [1, 2, 5, 10, 20, 50]:
    arr = np.clip(pixels.astype(np.int16) + shift, 0, 255).astype(np.uint8)
    perturbed = Image.fromarray(arr)
    h = extract_hidden(model, processor, perturbed, prompt)
    d = cosine_distance(h_clean, h)
    brightness_results[shift] = float(d)
    print(f"  +{shift:3d} brightness: d={d:.8f}")

results['brightness_shift'] = brightness_results

# 4. JPEG compression at various qualities
print("\n--- JPEG compression ---")
jpeg_results = {}
import io
for quality in [95, 90, 80, 70, 50, 30, 10]:
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    jpeg_img = Image.open(buf).convert('RGB')
    h = extract_hidden(model, processor, jpeg_img, prompt)
    d = cosine_distance(h_clean, h)
    jpeg_results[quality] = float(d)
    print(f"  JPEG q={quality:3d}: d={d:.8f}")

results['jpeg_compression'] = jpeg_results

# 5. Compare with actual corruption distances
print("\n--- Corruption reference distances ---")
from PIL import ImageFilter
corruption_refs = {}
for ctype, sev in [('fog', 0.01), ('fog', 0.05), ('fog', 0.1), ('fog', 1.0),
                    ('night', 0.01), ('night', 0.1), ('night', 1.0),
                    ('noise', 0.01), ('noise', 0.1)]:
    arr = pixels.astype(np.float32) / 255.0
    if ctype == 'fog':
        arr = arr * (1 - 0.6*sev) + 0.6*sev
    elif ctype == 'night':
        arr = arr * max(0.01, 1.0 - 0.95*sev)
    elif ctype == 'noise':
        arr = arr + np.random.RandomState(42).randn(*arr.shape) * 0.3 * sev
        arr = np.clip(arr, 0, 1)
    corrupted = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    h = extract_hidden(model, processor, corrupted, prompt)
    d = cosine_distance(h_clean, h)
    key = f"{ctype}_{sev}"
    corruption_refs[key] = float(d)
    print(f"  {key}: d={d:.8f}")

results['corruption_refs'] = corruption_refs

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
out = {
    'experiment': 'embedding_stability',
    'experiment_number': 265,
    'timestamp': ts,
    'results': results
}

path = f'/workspace/Vizuara-VLA-Research/experiments/stability_{ts}.json'
with open(path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {path}")
