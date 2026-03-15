"""Experiment 258: Corruption Interpolation
Linearly interpolates between clean and corrupted images in pixel space
to track the embedding trajectory and test linearity of the path.
"""
import torch, json, numpy as np
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image, ImageFilter
import time

print("=" * 60)
print("Experiment 258: Corruption Interpolation")
print("=" * 60)

print("Loading OpenVLA-7B...")
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", torch_dtype=torch.bfloat16,
    device_map="auto", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
model.eval()

prompt = "In: What action should the robot take to pick up the object?\nOut:"
base_img = Image.fromarray(np.random.RandomState(42).randint(0, 256, (256, 256, 3), dtype=np.uint8))
base_arr = np.array(base_img).astype(np.float32)

def extract_hidden(image, layer=3):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return fwd.hidden_states[layer][0, -1, :].float().cpu().numpy()

def cosine_dist(a, b):
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

# Create corrupted arrays
fog_arr = base_arr * 0.6 + 120 * 0.4
night_arr = base_arr * 0.15
noise_arr = base_arr + np.random.RandomState(99).randn(*base_arr.shape) * 50
noise_arr = noise_arr.clip(0, 255)

ref = extract_hidden(base_img)

# Interpolation: clean → corruption at 11 steps
alphas = np.linspace(0, 1, 11)
results = {}

for name, corrupt_arr in [('fog', fog_arr), ('night', night_arr), ('noise', noise_arr)]:
    print(f"\n--- Interpolation: clean → {name} ---")
    distances = []
    embeddings = []
    
    for alpha in alphas:
        interp_arr = (1 - alpha) * base_arr + alpha * corrupt_arr
        interp_img = Image.fromarray(interp_arr.clip(0, 255).astype(np.uint8))
        h = extract_hidden(interp_img)
        d = cosine_dist(ref, h)
        distances.append(round(d, 8))
        embeddings.append(h)
        print(f"  α={alpha:.1f}: d={d:.8f}")
    
    # Check linearity of embedding path
    # Measure if intermediate embeddings lie on the line between clean and corrupt
    clean_emb = embeddings[0]
    corrupt_emb = embeddings[-1]
    direction = corrupt_emb - clean_emb
    dir_norm = np.linalg.norm(direction)
    
    deviations = []
    for i, h in enumerate(embeddings):
        if i == 0 or i == len(embeddings) - 1:
            continue
        # Project onto clean-corrupt line
        diff = h - clean_emb
        proj_len = np.dot(diff, direction) / (dir_norm + 1e-10)
        proj = clean_emb + (proj_len / dir_norm) * direction
        deviation = np.linalg.norm(h - proj) / np.linalg.norm(diff + 1e-10)
        deviations.append(round(float(deviation), 6))
    
    results[name] = {
        "distances": distances,
        "max_deviation": round(float(max(deviations)), 6) if deviations else 0,
        "mean_deviation": round(float(np.mean(deviations)), 6) if deviations else 0,
        "r_squared": round(float(np.corrcoef(alphas, distances)[0, 1] ** 2), 4)
    }
    print(f"  R² (distance vs α): {results[name]['r_squared']:.4f}")
    print(f"  Mean path deviation: {results[name]['mean_deviation']:.6f}")

# Also: interpolation between two corruptions (fog → night)
print(f"\n--- Interpolation: fog → night ---")
fog2night_d = []
for alpha in alphas:
    interp_arr = (1 - alpha) * fog_arr + alpha * night_arr
    interp_img = Image.fromarray(interp_arr.clip(0, 255).astype(np.uint8))
    h = extract_hidden(interp_img)
    d = cosine_dist(ref, h)
    fog2night_d.append(round(d, 8))
    print(f"  α={alpha:.1f}: d={d:.8f}")

results["fog_to_night"] = {
    "distances": fog2night_d,
    "r_squared": round(float(np.corrcoef(alphas, fog2night_d)[0, 1] ** 2), 4)
}

output = {
    "experiment": "corruption_interpolation",
    "experiment_number": 258,
    "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    "layer": 3,
    "n_steps": 11,
    "results": results
}

path = f"/workspace/Vizuara-VLA-Research/experiments/interpolation_{output['timestamp']}.json"
with open(path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {path}")
