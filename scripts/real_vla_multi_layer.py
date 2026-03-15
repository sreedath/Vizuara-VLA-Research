"""Experiment 255: Multi-Layer Fusion
Tests whether combining distances from multiple layers (L3+L15+L31)
improves detection, especially for diverse calibration scenarios.
"""
import torch, json, numpy as np
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image, ImageFilter
import time

print("=" * 60)
print("Experiment 255: Multi-Layer Fusion")
print("=" * 60)

print("Loading OpenVLA-7B...")
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", torch_dtype=torch.bfloat16,
    device_map="auto", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
model.eval()

prompt = "In: What action should the robot take to pick up the object?\nOut:"
base_img = Image.fromarray(np.random.RandomState(42).randint(0, 256, (256, 256, 3), dtype=np.uint8))

layers = [3, 7, 15, 23, 31]

def extract_multi(image):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return {l: fwd.hidden_states[l][0, -1, :].float().cpu().numpy() for l in layers}

def cosine_dist(a, b):
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def compute_auroc(id_scores, ood_scores):
    id_scores = np.asarray(id_scores)
    ood_scores = np.asarray(ood_scores)
    n_id, n_ood = len(id_scores), len(ood_scores)
    if n_id == 0 or n_ood == 0: return 0.5
    count = sum(float(np.sum(o > id_scores) + 0.5 * np.sum(o == id_scores)) for o in ood_scores)
    return count / (n_id * n_ood)

# Corruptions
def apply_fog(img): return Image.fromarray((np.array(img).astype(np.float32)*0.6+120*0.4).clip(0,255).astype(np.uint8))
def apply_night(img): return Image.fromarray((np.array(img).astype(np.float32)*0.15).clip(0,255).astype(np.uint8))
def apply_noise(img):
    a = np.array(img).astype(np.float32); a += np.random.RandomState(99).randn(*a.shape)*50
    return Image.fromarray(a.clip(0,255).astype(np.uint8))
def apply_blur(img): return img.filter(ImageFilter.GaussianBlur(radius=5))

# Reference
ref = extract_multi(base_img)

# Diverse calibration: 10 images
print("Diverse calibration...")
id_dists = {l: [] for l in layers}
id_fused_sum = []
id_fused_max = []
id_fused_norm = []

for seed in range(100, 110):
    img = Image.fromarray(np.random.RandomState(seed).randint(0, 256, (256, 256, 3), dtype=np.uint8))
    h = extract_multi(img)
    per_layer = {}
    for l in layers:
        d = cosine_dist(ref[l], h[l])
        id_dists[l].append(d)
        per_layer[l] = d
    
    # Fusion strategies
    vals = list(per_layer.values())
    id_fused_sum.append(sum(vals))
    id_fused_max.append(max(vals))
    # Normalized: divide by per-layer natural variation
    id_fused_norm.append(sum(vals))  # same for now, normalize later

print("  ID diverse calibration stats:")
for l in layers:
    print(f"    L{l}: mean={np.mean(id_dists[l]):.6f}, max={max(id_dists[l]):.6f}")
print(f"    SUM: mean={np.mean(id_fused_sum):.6f}, max={max(id_fused_sum):.6f}")
print(f"    MAX: mean={np.mean(id_fused_max):.6f}, max={max(id_fused_max):.6f}")

# Test corruptions
corruptions = {'fog': apply_fog, 'night': apply_night, 'noise': apply_noise, 'blur': apply_blur}
results = {}

for name, fn in corruptions.items():
    cimg = fn(base_img)
    h = extract_multi(cimg)
    
    layer_results = {}
    per_layer_d = {}
    for l in layers:
        d = cosine_dist(ref[l], h[l])
        auroc = compute_auroc(id_dists[l], [d])
        layer_results[str(l)] = {"distance": round(d, 8), "auroc": round(float(auroc), 4)}
        per_layer_d[l] = d
    
    # Fused
    fused_sum = sum(per_layer_d.values())
    fused_max = max(per_layer_d.values())
    auroc_sum = compute_auroc(id_fused_sum, [fused_sum])
    auroc_max = compute_auroc(id_fused_max, [fused_max])
    
    results[name] = {
        "per_layer": layer_results,
        "fused_sum": {"value": round(fused_sum, 8), "auroc": round(float(auroc_sum), 4)},
        "fused_max": {"value": round(fused_max, 8), "auroc": round(float(auroc_max), 4)}
    }
    
    print(f"\n  {name}:")
    for l in layers:
        print(f"    L{l}: d={per_layer_d[l]:.6f}, AUROC={layer_results[str(l)]['auroc']:.4f}")
    print(f"    SUM: d={fused_sum:.6f}, AUROC={auroc_sum:.4f}")
    print(f"    MAX: d={fused_max:.6f}, AUROC={auroc_max:.4f}")

output = {
    "experiment": "multi_layer_fusion",
    "experiment_number": 255,
    "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    "layers": layers,
    "results": results,
    "id_stats": {
        str(l): {"mean": round(np.mean(id_dists[l]), 8), "max": round(max(id_dists[l]), 8)} for l in layers
    }
}

path = f"/workspace/Vizuara-VLA-Research/experiments/multi_layer_{output['timestamp']}.json"
with open(path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {path}")
