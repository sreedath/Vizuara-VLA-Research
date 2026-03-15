"""Experiment 251: PCA of Corruption Subspace
Analyzes the principal components of corruption shift vectors to find the
intrinsic dimensionality of the corruption subspace.
"""
import torch, json, numpy as np
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image, ImageFilter
import time

print("=" * 60)
print("Experiment 251: PCA of Corruption Subspace")
print("=" * 60)

print("Loading OpenVLA-7B...")
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", torch_dtype=torch.bfloat16,
    device_map="auto", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
model.eval()

prompt = "In: What action should the robot take to pick up the object?\nOut:"
base_img = Image.fromarray(np.random.RandomState(42).randint(0, 256, (256, 256, 3), dtype=np.uint8))

def extract_hidden(image, layer=3):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return fwd.hidden_states[layer][0, -1, :].float().cpu().numpy()

# Corruptions at multiple severities to get diverse shift vectors
def apply_fog(img, s):
    arr = np.array(img).astype(np.float32)
    return Image.fromarray((arr * (1-s*0.5) + 120*s*0.5).clip(0,255).astype(np.uint8))

def apply_night(img, s):
    return Image.fromarray((np.array(img).astype(np.float32) * (1-s*0.85)).clip(0,255).astype(np.uint8))

def apply_noise(img, s):
    arr = np.array(img).astype(np.float32)
    arr += np.random.RandomState(99).randn(*arr.shape) * s * 50
    return Image.fromarray(arr.clip(0,255).astype(np.uint8))

def apply_blur(img, s):
    r = s * 5
    return img.filter(ImageFilter.GaussianBlur(radius=max(r, 0.1))) if r > 0.1 else img

def apply_snow(img, s):
    arr = np.array(img).astype(np.float32)
    mask = np.random.RandomState(55).random(arr.shape[:2]) < s * 0.3
    arr[mask] = 255
    return Image.fromarray(arr.clip(0,255).astype(np.uint8))

def apply_rain(img, s):
    arr = np.array(img).astype(np.float32)
    for i in range(int(s * 100)):
        x = np.random.RandomState(i+200).randint(0, arr.shape[1])
        y0 = np.random.RandomState(i+300).randint(0, arr.shape[0]-20)
        arr[y0:y0+15, max(0,x-1):x+1, :] = arr[y0:y0+15, max(0,x-1):x+1, :] * 0.7 + 200 * 0.3
    return Image.fromarray(arr.clip(0,255).astype(np.uint8))

ref = extract_hidden(base_img)
print(f"Reference extracted (4096D)")

# Collect shift vectors: 6 types × 5 severities = 30 vectors
corruptions = {'fog': apply_fog, 'night': apply_night, 'noise': apply_noise,
               'blur': apply_blur, 'snow': apply_snow, 'rain': apply_rain}
severities = [0.2, 0.4, 0.6, 0.8, 1.0]

shift_vectors = []
labels = []

for name, fn in corruptions.items():
    for sev in severities:
        h = extract_hidden(fn(base_img, sev))
        shift = h - ref
        shift_vectors.append(shift)
        labels.append(f"{name}_{sev}")
        print(f"  {name} sev={sev}: ||shift||={np.linalg.norm(shift):.4f}")

shift_matrix = np.array(shift_vectors)  # 30 × 4096
print(f"\nShift matrix: {shift_matrix.shape}")

# PCA
from numpy.linalg import svd
U, S, Vt = svd(shift_matrix, full_matrices=False)
explained_var = S**2 / np.sum(S**2)
cumulative_var = np.cumsum(explained_var)

print("\n--- PCA Results ---")
for i in range(min(15, len(S))):
    print(f"  PC{i+1}: singular={S[i]:.4f}, var_explained={explained_var[i]:.4f} ({cumulative_var[i]:.4f} cumulative)")

# Find dimensionality at 90%, 95%, 99%
dim_90 = int(np.searchsorted(cumulative_var, 0.90) + 1)
dim_95 = int(np.searchsorted(cumulative_var, 0.95) + 1)
dim_99 = int(np.searchsorted(cumulative_var, 0.99) + 1)
print(f"\n  90% variance: {dim_90} components")
print(f"  95% variance: {dim_95} components")
print(f"  99% variance: {dim_99} components")

# Project onto top 2 PCs for visualization
projections_2d = U[:, :2] * S[:2]
proj_by_type = {}
for i, label in enumerate(labels):
    ctype = label.split('_')[0]
    if ctype not in proj_by_type:
        proj_by_type[ctype] = []
    proj_by_type[ctype].append([round(float(projections_2d[i, 0]), 6),
                                  round(float(projections_2d[i, 1]), 6)])

output = {
    "experiment": "pca_subspace",
    "experiment_number": 251,
    "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    "n_vectors": len(shift_vectors),
    "n_types": len(corruptions),
    "n_severities": len(severities),
    "results": {
        "singular_values": [round(float(s), 6) for s in S[:15]],
        "explained_variance": [round(float(v), 6) for v in explained_var[:15]],
        "cumulative_variance": [round(float(v), 6) for v in cumulative_var[:15]],
        "dim_90": dim_90,
        "dim_95": dim_95,
        "dim_99": dim_99,
        "projections_2d": proj_by_type,
        "labels": labels
    }
}

path = f"/workspace/Vizuara-VLA-Research/experiments/pca_subspace_{output['timestamp']}.json"
with open(path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {path}")
