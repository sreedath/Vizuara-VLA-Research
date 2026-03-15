"""Experiment 249: Full Layer Distance Profile
Measures cosine distance at ALL 32 transformer layers (L1-L32) to find
the complete layer-wise OOD sensitivity profile.
"""
import torch, json, numpy as np
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import time

print("=" * 60)
print("Experiment 249: Full Layer Distance Profile")
print("=" * 60)

print("Loading OpenVLA-7B...")
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", torch_dtype=torch.bfloat16,
    device_map="auto", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
model.eval()

prompt = "In: What action should the robot take to pick up the object?\nOut:"
base_img = Image.fromarray(np.random.RandomState(42).randint(0, 256, (256, 256, 3), dtype=np.uint8))

# All 33 hidden states (embedding + 32 layers)
all_layers = list(range(33))

def extract_all_hidden(image):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return {l: fwd.hidden_states[l][0, -1, :].float().cpu().numpy() for l in all_layers}

def cosine_dist(a, b):
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

# Corruptions
def apply_fog(img, intensity=0.4):
    arr = np.array(img).astype(np.float32)
    arr = arr * (1 - intensity) + 120 * intensity
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))

def apply_night(img, factor=0.15):
    arr = (np.array(img).astype(np.float32) * factor).clip(0, 255)
    return Image.fromarray(arr.astype(np.uint8))

def apply_noise(img, std=50):
    arr = np.array(img).astype(np.float32)
    arr += np.random.RandomState(99).randn(*arr.shape) * std
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))

def apply_blur(img, radius=5):
    from PIL import ImageFilter
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

corruptions = {
    'fog': apply_fog(base_img),
    'night': apply_night(base_img),
    'noise': apply_noise(base_img),
    'blur': apply_blur(base_img)
}

# Extract clean reference
print("Extracting clean reference (all 33 layers)...")
clean_h = extract_all_hidden(base_img)

results = {}
for name, img in corruptions.items():
    print(f"\n--- {name} ---")
    h = extract_all_hidden(img)
    distances = {}
    for l in all_layers:
        d = cosine_dist(clean_h[l], h[l])
        distances[str(l)] = round(d, 8)
    results[name] = distances
    
    # Print summary
    dists = list(distances.values())
    max_l = max(distances, key=distances.get)
    min_l = min(distances, key=distances.get)
    print(f"  Max distance: L{max_l} = {distances[max_l]:.6f}")
    print(f"  Min distance: L{min_l} = {distances[min_l]:.6f}")
    print(f"  Mean distance: {np.mean(dists):.6f}")
    
    # Print all layers
    for l in all_layers:
        print(f"  L{l}: {distances[str(l)]:.8f}")

output = {
    "experiment": "layer_profile",
    "experiment_number": 249,
    "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    "n_layers": 33,
    "results": results
}

path = f"/workspace/Vizuara-VLA-Research/experiments/layer_profile_{output['timestamp']}.json"
with open(path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {path}")
