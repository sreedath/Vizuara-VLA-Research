"""Experiment 253: Cross-Corruption Transfer
Tests if a detector calibrated on one corruption type can detect other types.
Measures AUROC when using single-corruption centroids vs multi-corruption centroids.
"""
import torch, json, numpy as np
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image, ImageFilter
import time

print("=" * 60)
print("Experiment 253: Cross-Corruption Transfer")
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

# Corruption functions
def apply_fog(img, s=0.4):
    arr = np.array(img).astype(np.float32)
    return Image.fromarray((arr*(1-s)+120*s).clip(0,255).astype(np.uint8))

def apply_night(img, f=0.15):
    return Image.fromarray((np.array(img).astype(np.float32)*f).clip(0,255).astype(np.uint8))

def apply_noise(img, std=50):
    arr = np.array(img).astype(np.float32)
    arr += np.random.RandomState(99).randn(*arr.shape)*std
    return Image.fromarray(arr.clip(0,255).astype(np.uint8))

def apply_blur(img, r=5):
    return img.filter(ImageFilter.GaussianBlur(radius=r))

def apply_snow(img, density=0.3):
    arr = np.array(img).astype(np.float32)
    mask = np.random.RandomState(55).random(arr.shape[:2]) < density
    arr[mask] = 255
    return Image.fromarray(arr.clip(0,255).astype(np.uint8))

corruptions = {
    'fog': apply_fog(base_img),
    'night': apply_night(base_img),
    'noise': apply_noise(base_img),
    'blur': apply_blur(base_img),
    'snow': apply_snow(base_img)
}

# Get clean and corrupt embeddings
ref = extract_hidden(base_img)
corrupt_embeddings = {name: extract_hidden(img) for name, img in corruptions.items()}

# Get clean variation (multiple seeds) for ID scores
id_distances = []
for seed in range(100, 110):
    img = Image.fromarray(np.random.RandomState(seed).randint(0, 256, (256, 256, 3), dtype=np.uint8))
    h = extract_hidden(img)
    d = cosine_dist(ref, h)
    id_distances.append(d)

print(f"Clean ID distances: mean={np.mean(id_distances):.6f}, max={max(id_distances):.6f}")

# Compute OOD distances for each corruption
ood_distances = {}
for name, h in corrupt_embeddings.items():
    d = cosine_dist(ref, h)
    ood_distances[name] = d
    print(f"  {name}: d={d:.6f}")

# Cross-corruption transfer: train on one, test on all
print("\n--- Cross-Corruption Transfer Matrix ---")
transfer_matrix = {}

for train_type in list(corruptions.keys()):
    transfer_matrix[train_type] = {}
    # The "detector" just uses clean centroid (ref) with threshold from train_type
    train_dist = ood_distances[train_type]
    threshold = train_dist * 0.5  # midpoint threshold
    
    for test_type in list(corruptions.keys()):
        test_dist = ood_distances[test_type]
        # Simple detection: is test_dist > threshold?
        detected = test_dist > threshold
        # AUROC: can we separate clean from test using the clean centroid?
        auroc = compute_auroc(id_distances, [test_dist])
        transfer_matrix[train_type][test_type] = {
            "detected": bool(detected),
            "threshold": round(threshold, 8),
            "test_distance": round(test_dist, 8),
            "auroc": round(float(auroc), 4)
        }
        print(f"  Train={train_type}, Test={test_type}: detected={detected}, auroc={auroc:.4f}")

# The key insight: ALL use same clean centroid, so cross-transfer is trivially 100%
# The real question is about thresholds from different corruptions

# Now test: use corruption-specific centroid (midpoint of clean-corrupt)
print("\n--- Corruption-Specific Centroid Analysis ---")
specific_results = {}
for name, h in corrupt_embeddings.items():
    mid = (ref + h) / 2  # midpoint centroid
    mid_norm = mid / (np.linalg.norm(mid) + 1e-10)
    
    # Distance from clean to midpoint
    clean_mid_dist = cosine_dist(ref, mid)
    # Distance from corrupt to midpoint
    corrupt_mid_dist = cosine_dist(h, mid)
    # Distance from other corruptions to midpoint
    other_dists = {}
    for other_name, other_h in corrupt_embeddings.items():
        if other_name != name:
            other_dists[other_name] = round(cosine_dist(other_h, mid), 8)
    
    specific_results[name] = {
        "clean_to_mid": round(clean_mid_dist, 8),
        "corrupt_to_mid": round(corrupt_mid_dist, 8),
        "other_distances": other_dists
    }
    print(f"  {name}: clean→mid={clean_mid_dist:.6f}, corrupt→mid={corrupt_mid_dist:.6f}")

output = {
    "experiment": "cross_transfer",
    "experiment_number": 253,
    "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    "layer": 3,
    "results": {
        "id_distances": [round(d, 8) for d in id_distances],
        "ood_distances": {k: round(v, 8) for k, v in ood_distances.items()},
        "transfer_matrix": transfer_matrix,
        "specific_centroids": specific_results
    }
}

path = f"/workspace/Vizuara-VLA-Research/experiments/cross_transfer_{output['timestamp']}.json"
with open(path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {path}")
