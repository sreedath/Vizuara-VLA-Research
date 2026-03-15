#!/usr/bin/env python3
"""Experiment 282: Empirical Detection Bound Analysis
Determines the theoretical minimum detectable corruption by:
1. Measuring the noise floor (smallest nonzero distance for benign perturbations)
2. Testing ultra-fine severity increments (0.001 to 0.01) for each corruption
3. Establishing the minimum corruption that produces d > noise_floor
4. Computing the theoretical detection limit
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

def extract_hidden(model, processor, image, prompt, layer=3):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return fwd.hidden_states[layer][0, -1, :].float().cpu().numpy()

print("=" * 60)
print("Experiment 282: Empirical Detection Bound")
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

centroid = extract_hidden(model, processor, img, prompt)

def compute_distance(emb, centroid):
    return float(1.0 - np.dot(emb, centroid) / (np.linalg.norm(emb) * np.linalg.norm(centroid) + 1e-30))

# Step 1: Measure noise floor (re-run same clean image multiple times)
print("\n=== NOISE FLOOR ===")
noise_floor_distances = []
for i in range(10):
    h = extract_hidden(model, processor, img, prompt)
    d = compute_distance(h, centroid)
    noise_floor_distances.append(d)
    print(f"  Run {i}: d={d:.15f}")

noise_floor = max(noise_floor_distances)
print(f"  Noise floor: {noise_floor:.15e}")

# Step 2: Ultra-fine severity sweep
print("\n=== ULTRA-FINE SEVERITY SWEEP ===")
fine_severities = [0.0005, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1]

fine_results = {}
for ctype in ['fog', 'night', 'noise', 'blur']:
    print(f"\n--- {ctype} ---")
    cresults = {}
    first_detected = None
    for sev in fine_severities:
        frame = apply_corruption(img, ctype, sev)
        h = extract_hidden(model, processor, frame, prompt)
        d = compute_distance(h, centroid)
        above_noise = d > noise_floor
        cresults[str(sev)] = {'distance': d, 'above_noise_floor': above_noise}
        if above_noise and first_detected is None:
            first_detected = sev
        print(f"  sev={sev:.4f}: d={d:.10f} {'> noise' if above_noise else '<= noise'}")
    
    fine_results[ctype] = {
        'distances': cresults,
        'first_detected_severity': first_detected,
        'detection_bound': first_detected
    }
    print(f"  First detected at severity: {first_detected}")

# Step 3: Single-pixel perturbation sweep
print("\n=== SINGLE-PIXEL PERTURBATION ===")
pixel_results = []
rng = np.random.RandomState(42)
for n_pixels in [1, 2, 5, 10, 20, 50, 100, 500]:
    arr = np.array(img).copy()
    for _ in range(n_pixels):
        x, y = rng.randint(0, 224), rng.randint(0, 224)
        arr[y, x] = rng.randint(0, 256, 3).astype(np.uint8)
    frame = Image.fromarray(arr)
    h = extract_hidden(model, processor, frame, prompt)
    d = compute_distance(h, centroid)
    pixel_results.append({'n_pixels': n_pixels, 'distance': d, 'above_noise': d > noise_floor})
    print(f"  {n_pixels:4d} pixels: d={d:.10f} {'> noise' if d > noise_floor else '<= noise'}")

# Step 4: Bit-depth reduction
print("\n=== BIT-DEPTH REDUCTION ===")
bitdepth_results = []
for bits in [8, 7, 6, 5, 4, 3, 2, 1]:
    arr = np.array(img).copy()
    shift = 8 - bits
    arr = ((arr >> shift) << shift).astype(np.uint8)
    frame = Image.fromarray(arr)
    h = extract_hidden(model, processor, frame, prompt)
    d = compute_distance(h, centroid)
    bitdepth_results.append({'bits': bits, 'distance': d, 'above_noise': d > noise_floor})
    print(f"  {bits} bits: d={d:.10f}")

# Step 5: Uniform brightness shift (1-level increments)
print("\n=== MINIMAL BRIGHTNESS SHIFT ===")
brightness_results = []
for shift in [1, 2, 3, 5, 10, 20]:
    arr = np.array(img).astype(np.int16) + shift
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    frame = Image.fromarray(arr)
    h = extract_hidden(model, processor, frame, prompt)
    d = compute_distance(h, centroid)
    brightness_results.append({'shift': shift, 'distance': d, 'above_noise': d > noise_floor})
    print(f"  +{shift:2d}: d={d:.10f}")

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
out = {
    'experiment': 'detection_bound',
    'experiment_number': 282,
    'timestamp': ts,
    'results': {
        'noise_floor': {
            'distances': noise_floor_distances,
            'max': noise_floor,
            'mean': float(np.mean(noise_floor_distances)),
        },
        'fine_severity': fine_results,
        'pixel_perturbation': pixel_results,
        'bitdepth': bitdepth_results,
        'brightness': brightness_results
    }
}

path = f'/workspace/Vizuara-VLA-Research/experiments/bound_{ts}.json'
with open(path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {path}")
