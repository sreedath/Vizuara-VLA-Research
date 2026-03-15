"""Experiment 248: Temporal Drift Analysis
Measures how hidden state embeddings drift over extended clean sequences (50 frames)
to establish the true temporal noise floor and assess calibration stability.
"""
import torch, json, numpy as np
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import time

print("=" * 60)
print("Experiment 248: Temporal Drift Analysis")
print("=" * 60)

print("Loading OpenVLA-7B...")
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", torch_dtype=torch.bfloat16,
    device_map="auto", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
model.eval()

prompt = "In: What action should the robot take to pick up the object?\nOut:"
base_img = Image.fromarray(np.random.RandomState(42).randint(0, 256, (256, 256, 3), dtype=np.uint8))

def extract_hidden(image, layers=[3, 15, 31]):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return {l: fwd.hidden_states[l][0, -1, :].float().cpu().numpy() for l in layers}

def cosine_dist(a, b):
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

layers = [3, 15, 31]
n_frames = 50
results = {l: [] for l in layers}

# Get reference embedding (frame 0)
ref = extract_hidden(base_img, layers)
print(f"Reference embedding extracted")

# Run 50 frames with identical image and measure drift
for i in range(n_frames):
    # Recreate image from same seed each time (identical content)
    img = Image.fromarray(np.random.RandomState(42).randint(0, 256, (256, 256, 3), dtype=np.uint8))
    h = extract_hidden(img, layers)
    for l in layers:
        d = cosine_dist(ref[l], h[l])
        results[l].append(round(d, 10))
    if (i + 1) % 10 == 0:
        print(f"  Frame {i+1}/{n_frames}: L3={results[3][-1]:.2e}, L15={results[15][-1]:.2e}, L31={results[31][-1]:.2e}")

# Also test with slightly different images (variation seeds)
print("\n--- Variation test (different random seeds) ---")
variation_results = {l: [] for l in layers}
for seed in range(100, 120):
    img = Image.fromarray(np.random.RandomState(seed).randint(0, 256, (256, 256, 3), dtype=np.uint8))
    h = extract_hidden(img, layers)
    for l in layers:
        d = cosine_dist(ref[l], h[l])
        variation_results[l].append(round(d, 8))
    print(f"  Seed {seed}: L3={variation_results[3][-1]:.6f}, L15={variation_results[15][-1]:.6f}, L31={variation_results[31][-1]:.6f}")

output = {
    "experiment": "temporal_drift",
    "experiment_number": 248,
    "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    "n_frames": n_frames,
    "layers": layers,
    "results": {
        "same_image_drift": {
            str(l): {
                "distances": results[l],
                "max": round(float(max(results[l])), 10),
                "mean": round(float(np.mean(results[l])), 10),
                "std": round(float(np.std(results[l])), 10),
                "monotonic": all(results[l][i] <= results[l][i+1] for i in range(len(results[l])-1))
            } for l in layers
        },
        "different_images": {
            str(l): {
                "distances": variation_results[l],
                "max": round(float(max(variation_results[l])), 8),
                "min": round(float(min(variation_results[l])), 8),
                "mean": round(float(np.mean(variation_results[l])), 8),
                "std": round(float(np.std(variation_results[l])), 8)
            } for l in layers
        }
    }
}

path = f"/workspace/Vizuara-VLA-Research/experiments/temporal_drift_{output['timestamp']}.json"
with open(path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {path}")
