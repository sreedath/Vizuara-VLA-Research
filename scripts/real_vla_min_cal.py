"""Experiment 257: Minimum Calibration Images
Tests how many per-scene images are needed for robust calibration.
With 1 image: zero variance. With N images: what's the centroid stability?
"""
import torch, json, numpy as np
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image, ImageFilter
import time

print("=" * 60)
print("Experiment 257: Minimum Calibration Images")
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

# Collect 20 embeddings of the SAME scene (same image, different forward passes)
# Since model is deterministic, these should be identical
print("Collecting 20 same-scene embeddings...")
same_scene_embs = []
for i in range(20):
    h = extract_hidden(base_img)
    same_scene_embs.append(h)

# Test with N=1, 2, 5, 10, 20 calibration images
cal_sizes = [1, 2, 5, 10, 20]
results = {}

for n_cal in cal_sizes:
    centroid = np.mean(same_scene_embs[:n_cal], axis=0)
    
    # Compute ID distances (remaining images from same scene)
    id_dists = []
    for i in range(n_cal, 20):
        d = cosine_dist(centroid, same_scene_embs[i])
        id_dists.append(d)
    
    # Compute OOD distances
    fog_d = cosine_dist(centroid, extract_hidden(apply_fog(base_img)))
    night_d = cosine_dist(centroid, extract_hidden(apply_night(base_img)))
    noise_d = cosine_dist(centroid, extract_hidden(apply_noise(base_img)))
    
    # AUROC for each corruption
    fog_auroc = compute_auroc(id_dists, [fog_d]) if len(id_dists) > 0 else 1.0
    night_auroc = compute_auroc(id_dists, [night_d]) if len(id_dists) > 0 else 1.0
    noise_auroc = compute_auroc(id_dists, [noise_d]) if len(id_dists) > 0 else 1.0
    
    results[str(n_cal)] = {
        "id_distances": [round(d, 10) for d in id_dists],
        "id_mean": round(float(np.mean(id_dists)), 10) if id_dists else 0,
        "id_max": round(float(max(id_dists)), 10) if id_dists else 0,
        "fog": {"distance": round(fog_d, 8), "auroc": round(float(fog_auroc), 4)},
        "night": {"distance": round(night_d, 8), "auroc": round(float(night_auroc), 4)},
        "noise": {"distance": round(noise_d, 8), "auroc": round(float(noise_auroc), 4)}
    }
    print(f"\n  n_cal={n_cal}:")
    print(f"    ID: mean={np.mean(id_dists) if id_dists else 0:.2e}, max={max(id_dists) if id_dists else 0:.2e}")
    print(f"    Fog: d={fog_d:.6f}, AUROC={fog_auroc:.4f}")
    print(f"    Night: d={night_d:.6f}, AUROC={night_auroc:.4f}")
    print(f"    Noise: d={noise_d:.6f}, AUROC={noise_auroc:.4f}")

# Also test: centroid of N different scenes
print("\n--- Multi-Scene Centroid ---")
multi_scene_embs = []
for seed in range(42, 52):
    img = Image.fromarray(np.random.RandomState(seed).randint(0,256,(256,256,3),dtype=np.uint8))
    multi_scene_embs.append(extract_hidden(img))

multi_centroid = np.mean(multi_scene_embs, axis=0)
ms_fog_d = cosine_dist(multi_centroid, extract_hidden(apply_fog(base_img)))
ms_night_d = cosine_dist(multi_centroid, extract_hidden(apply_night(base_img)))
ms_noise_d = cosine_dist(multi_centroid, extract_hidden(apply_noise(base_img)))
ms_id_dists = [cosine_dist(multi_centroid, e) for e in multi_scene_embs]

results["multi_scene_10"] = {
    "id_distances": [round(d, 8) for d in ms_id_dists],
    "id_mean": round(float(np.mean(ms_id_dists)), 8),
    "id_max": round(float(max(ms_id_dists)), 8),
    "fog": {"distance": round(ms_fog_d, 8), "auroc": round(float(compute_auroc(ms_id_dists, [ms_fog_d])), 4)},
    "night": {"distance": round(ms_night_d, 8), "auroc": round(float(compute_auroc(ms_id_dists, [ms_night_d])), 4)},
    "noise": {"distance": round(ms_noise_d, 8), "auroc": round(float(compute_auroc(ms_id_dists, [ms_noise_d])), 4)}
}
print(f"\n  Multi-scene (10 scenes):")
print(f"    ID: mean={np.mean(ms_id_dists):.6f}, max={max(ms_id_dists):.6f}")
print(f"    Fog: d={ms_fog_d:.6f}, Night: d={ms_night_d:.6f}, Noise: d={ms_noise_d:.6f}")

output = {
    "experiment": "min_calibration",
    "experiment_number": 257,
    "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    "layer": 3,
    "results": results
}

path = f"/workspace/Vizuara-VLA-Research/experiments/min_cal_{output['timestamp']}.json"
with open(path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {path}")
