"""Experiment 254: Image Token Aggregation
Compares last-token detection vs mean-image-token detection for OOD.
Tests whether aggregating image tokens improves detection with diverse calibration.
"""
import torch, json, numpy as np
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image, ImageFilter
import time

print("=" * 60)
print("Experiment 254: Image Token Aggregation")
print("=" * 60)

print("Loading OpenVLA-7B...")
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", torch_dtype=torch.bfloat16,
    device_map="auto", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
model.eval()

prompt = "In: What action should the robot take to pick up the object?\nOut:"
base_img = Image.fromarray(np.random.RandomState(42).randint(0, 256, (256, 256, 3), dtype=np.uint8))

def extract_embeddings(image, layer=3):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    h = fwd.hidden_states[layer][0, :, :].float().cpu().numpy()  # (n_tokens, 4096)
    n_tokens = h.shape[0]
    # Image tokens are positions 1 to ~256 (after BOS)
    # Text tokens start at ~257
    last_token = h[-1]
    mean_img = h[1:257].mean(axis=0)  # Mean of image tokens
    max_img = h[1:257].max(axis=0)  # Max of image tokens
    return last_token, mean_img, max_img, n_tokens

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

# Get reference embeddings
print("Extracting reference...")
ref_last, ref_mean_img, ref_max_img, n_tok = extract_embeddings(base_img)
print(f"  n_tokens={n_tok}")

# Diverse calibration: 10 different images
print("\nDiverse calibration (10 images)...")
id_last, id_mean, id_max = [], [], []
for seed in range(100, 110):
    img = Image.fromarray(np.random.RandomState(seed).randint(0, 256, (256, 256, 3), dtype=np.uint8))
    lt, mi, mx, _ = extract_embeddings(img)
    id_last.append(cosine_dist(ref_last, lt))
    id_mean.append(cosine_dist(ref_mean_img, mi))
    id_max.append(cosine_dist(ref_max_img, mx))

print(f"  ID last-token: mean={np.mean(id_last):.6f}, max={max(id_last):.6f}")
print(f"  ID mean-image: mean={np.mean(id_mean):.6f}, max={max(id_mean):.6f}")
print(f"  ID max-image:  mean={np.mean(id_max):.6f}, max={max(id_max):.6f}")

# OOD detection
corruptions = {'fog': apply_fog, 'night': apply_night, 'noise': apply_noise, 'blur': apply_blur}
results = {}

for name, fn in corruptions.items():
    cimg = fn(base_img)
    lt, mi, mx, _ = extract_embeddings(cimg)
    d_last = cosine_dist(ref_last, lt)
    d_mean = cosine_dist(ref_mean_img, mi)
    d_max = cosine_dist(ref_max_img, mx)
    
    auroc_last = compute_auroc(id_last, [d_last])
    auroc_mean = compute_auroc(id_mean, [d_mean])
    auroc_max = compute_auroc(id_max, [d_max])
    
    results[name] = {
        "last_token": {"distance": round(d_last, 8), "auroc_diverse": round(float(auroc_last), 4)},
        "mean_image": {"distance": round(d_mean, 8), "auroc_diverse": round(float(auroc_mean), 4)},
        "max_image": {"distance": round(d_max, 8), "auroc_diverse": round(float(auroc_max), 4)}
    }
    print(f"\n  {name}:")
    print(f"    last-token: d={d_last:.6f}, AUROC={auroc_last:.4f}")
    print(f"    mean-image: d={d_mean:.6f}, AUROC={auroc_mean:.4f}")
    print(f"    max-image:  d={d_max:.6f}, AUROC={auroc_max:.4f}")

output = {
    "experiment": "token_aggregation",
    "experiment_number": 254,
    "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    "layer": 3,
    "n_tokens": n_tok,
    "id_stats": {
        "last_token": {"mean": round(np.mean(id_last), 8), "max": round(max(id_last), 8)},
        "mean_image": {"mean": round(np.mean(id_mean), 8), "max": round(max(id_mean), 8)},
        "max_image": {"mean": round(np.mean(id_max), 8), "max": round(max(id_max), 8)}
    },
    "results": results
}

path = f"/workspace/Vizuara-VLA-Research/experiments/token_agg_{output['timestamp']}.json"
with open(path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {path}")
