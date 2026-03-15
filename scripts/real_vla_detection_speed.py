#!/usr/bin/env python3
"""Experiment 270: Detection Speed Scaling
Benchmarks detection latency as a function of:
1. Number of calibration centroids (1-100)
2. Embedding dimensionality (32, 128, 512, 4096)
3. Distance metric computation
Measures total detection overhead per frame.
"""
import torch, json, numpy as np, time
from PIL import Image
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
print("Experiment 270: Detection Speed Scaling")
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

# Extract embedding for benchmarking
h = extract_hidden(model, processor, img, prompt)

results = {}

# 1. Full pipeline latency (extraction + comparison)
print("\n--- Full Pipeline Latency ---")
pipeline_times = []
for trial in range(20):
    t0 = time.time()
    h_test = extract_hidden(model, processor, img, prompt)
    d = cosine_distance(h_test, h)
    t1 = time.time()
    pipeline_times.append((t1 - t0) * 1000)

results['full_pipeline_ms'] = {
    'mean': float(np.mean(pipeline_times)),
    'std': float(np.std(pipeline_times)),
    'min': float(np.min(pipeline_times)),
    'max': float(np.max(pipeline_times)),
    'p50': float(np.percentile(pipeline_times, 50)),
    'p99': float(np.percentile(pipeline_times, 99)),
}
print(f"  Full pipeline: {np.mean(pipeline_times):.2f} ± {np.std(pipeline_times):.2f} ms")

# 2. Extraction-only latency
print("\n--- Extraction-Only Latency ---")
extract_times = []
for trial in range(20):
    t0 = time.time()
    h_test = extract_hidden(model, processor, img, prompt)
    t1 = time.time()
    extract_times.append((t1 - t0) * 1000)

results['extraction_ms'] = {
    'mean': float(np.mean(extract_times)),
    'std': float(np.std(extract_times)),
}
print(f"  Extraction: {np.mean(extract_times):.2f} ± {np.std(extract_times):.2f} ms")

# 3. Distance computation scaling with N centroids
print("\n--- Distance vs N Centroids ---")
centroid_results = {}
for n_centroids in [1, 5, 10, 50, 100, 500, 1000]:
    centroids = [np.random.randn(4096).astype(np.float32) for _ in range(n_centroids)]
    times = []
    for trial in range(100):
        t0 = time.time()
        min_d = min(cosine_distance(h, c) for c in centroids)
        t1 = time.time()
        times.append((t1 - t0) * 1000)
    mean_t = float(np.mean(times))
    centroid_results[n_centroids] = mean_t
    print(f"  {n_centroids:5d} centroids: {mean_t:.4f} ms")

results['centroid_scaling'] = centroid_results

# 4. Dimensionality scaling
print("\n--- Distance vs Dimensionality ---")
dim_results = {}
for dim in [32, 64, 128, 256, 512, 1024, 2048, 4096]:
    a = np.random.randn(dim).astype(np.float32)
    b = np.random.randn(dim).astype(np.float32)
    times = []
    for trial in range(1000):
        t0 = time.time()
        d = cosine_distance(a, b)
        t1 = time.time()
        times.append((t1 - t0) * 1000)
    mean_t = float(np.mean(times))
    dim_results[dim] = mean_t
    print(f"  {dim:5d}D: {mean_t:.6f} ms")

results['dim_scaling'] = dim_results

# 5. Random projection latency
print("\n--- Random Projection ---")
proj_results = {}
for target_dim in [32, 64, 128, 256]:
    proj_matrix = np.random.randn(target_dim, 4096).astype(np.float32) / np.sqrt(target_dim)
    times = []
    for trial in range(100):
        t0 = time.time()
        h_proj = proj_matrix @ h
        t1 = time.time()
        times.append((t1 - t0) * 1000)
    mean_t = float(np.mean(times))
    proj_results[target_dim] = mean_t
    print(f"  4096 -> {target_dim}D projection: {mean_t:.4f} ms")

results['projection_ms'] = proj_results

# 6. Inference-only time (no hidden state extraction)
print("\n--- Inference-Only Time ---")
inference_times = []
inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)
for trial in range(10):
    t0 = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    t1 = time.time()
    inference_times.append((t1 - t0) * 1000)

results['inference_ms'] = {
    'mean': float(np.mean(inference_times)),
    'std': float(np.std(inference_times)),
}
print(f"  Inference only: {np.mean(inference_times):.2f} ± {np.std(inference_times):.2f} ms")

# Detection overhead
det_overhead = np.mean(pipeline_times) - np.mean(inference_times)
results['detection_overhead_ms'] = float(det_overhead)
results['overhead_pct'] = float(det_overhead / np.mean(inference_times) * 100)
print(f"\n  Detection overhead: {det_overhead:.2f} ms ({det_overhead/np.mean(inference_times)*100:.2f}%)")

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
out = {
    'experiment': 'detection_speed',
    'experiment_number': 270,
    'timestamp': ts,
    'results': results
}

path = f'/workspace/Vizuara-VLA-Research/experiments/speed_{ts}.json'
with open(path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {path}")
