#!/usr/bin/env python3
"""Experiment 271: Feature Importance for OOD Detection
Identifies which embedding dimensions are most important for
distinguishing clean from corrupted inputs. Tests whether a small
subset of dimensions can achieve the same detection performance.
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
print("Experiment 271: Feature Importance for OOD Detection")
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

corruptions = ['fog', 'night', 'noise', 'blur']
shift_vectors = {}
for ctype in corruptions:
    corrupted = apply_corruption(img, ctype)
    h_corr = extract_hidden(model, processor, corrupted, prompt)
    shift_vectors[ctype] = h_corr - h_clean

results = {}

# 1. Per-dimension shift magnitude
print("\n--- Per-Dimension Shift Magnitude ---")
for ctype in corruptions:
    shift = shift_vectors[ctype]
    abs_shift = np.abs(shift)
    top_dims = np.argsort(abs_shift)[::-1][:20]
    total_energy = np.sum(abs_shift ** 2)
    cumulative = np.cumsum(np.sort(abs_shift ** 2)[::-1]) / total_energy

    # How many dims for 50%, 90%, 99% of energy?
    n_50 = int(np.searchsorted(cumulative, 0.5)) + 1
    n_90 = int(np.searchsorted(cumulative, 0.9)) + 1
    n_99 = int(np.searchsorted(cumulative, 0.99)) + 1

    results[ctype] = {
        'top20_dims': [int(d) for d in top_dims],
        'top20_magnitudes': [float(abs_shift[d]) for d in top_dims],
        'n_dims_50pct': n_50,
        'n_dims_90pct': n_90,
        'n_dims_99pct': n_99,
        'total_energy': float(total_energy),
        'max_dim_contribution': float(abs_shift[top_dims[0]] ** 2 / total_energy * 100),
    }
    print(f"  {ctype}: 50%={n_50}, 90%={n_90}, 99%={n_99} dims")
    print(f"    Top dim {top_dims[0]} contributes {abs_shift[top_dims[0]]**2/total_energy*100:.1f}%")

# 2. Dimension overlap across corruptions
print("\n--- Dimension Overlap ---")
top100 = {}
for ctype in corruptions:
    abs_shift = np.abs(shift_vectors[ctype])
    top100[ctype] = set(np.argsort(abs_shift)[::-1][:100])

overlap_matrix = {}
for c1 in corruptions:
    for c2 in corruptions:
        if c1 >= c2:
            continue
        overlap = len(top100[c1] & top100[c2])
        key = f"{c1}_vs_{c2}"
        overlap_matrix[key] = overlap
        print(f"  {c1} vs {c2}: {overlap}/100 overlap")

results['dim_overlap'] = overlap_matrix

# 3. Subset detection: use only top-K dims
print("\n--- Subset Detection ---")
subset_results = {}
for k in [10, 50, 100, 200, 500, 1000, 2000, 4096]:
    # Use top-K dims from combined shift magnitude
    combined_shift = sum(np.abs(shift_vectors[c]) for c in corruptions)
    top_k_dims = np.argsort(combined_shift)[::-1][:k]

    distances = {}
    for ctype in corruptions:
        h_corr = h_clean + shift_vectors[ctype]
        # Use only top-K dims
        h_clean_sub = h_clean[top_k_dims]
        h_corr_sub = h_corr[top_k_dims]
        d = cosine_distance(h_clean_sub, h_corr_sub)
        distances[ctype] = float(d)

    subset_results[k] = distances
    print(f"  K={k:5d}: " + ", ".join(f"{c}={d:.6f}" for c, d in distances.items()))

results['subset_detection'] = subset_results

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
out = {
    'experiment': 'feature_importance',
    'experiment_number': 271,
    'timestamp': ts,
    'results': results
}

path = f'/workspace/Vizuara-VLA-Research/experiments/feature_importance_{ts}.json'
with open(path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {path}")
