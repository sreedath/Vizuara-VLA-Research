"""Experiment 259: Full Action Dimension Analysis
Decodes all 7 action dimensions for clean and corrupted conditions
to understand exactly how corruption affects the complete action vector.
"""
import torch, json, numpy as np
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image, ImageFilter
import time

print("=" * 60)
print("Experiment 259: Full Action Dimension Analysis")
print("=" * 60)

print("Loading OpenVLA-7B...")
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", torch_dtype=torch.bfloat16,
    device_map="auto", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
model.eval()

prompt = "In: What action should the robot take to pick up the object?\nOut:"
base_img = Image.fromarray(np.random.RandomState(42).randint(0, 256, (256, 256, 3), dtype=np.uint8))

def get_actions(image, n_tokens=7):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=n_tokens, do_sample=False)
    # Get last n_tokens
    tokens = out[0, -n_tokens:].cpu().tolist()
    return tokens

# Corruptions
def apply_fog(img): return Image.fromarray((np.array(img).astype(np.float32)*0.6+120*0.4).clip(0,255).astype(np.uint8))
def apply_night(img): return Image.fromarray((np.array(img).astype(np.float32)*0.15).clip(0,255).astype(np.uint8))
def apply_noise(img):
    a = np.array(img).astype(np.float32); a += np.random.RandomState(99).randn(*a.shape)*50
    return Image.fromarray(a.clip(0,255).astype(np.uint8))
def apply_blur(img): return img.filter(ImageFilter.GaussianBlur(radius=5))

# Get clean actions
print("Clean actions...")
clean_tokens = get_actions(base_img)
print(f"  Clean: {clean_tokens}")

# Action token range: 31744-31999 (256 bins)
# Token ID - 31744 = bin number (0-255)
clean_bins = [t - 31744 for t in clean_tokens]
print(f"  Clean bins: {clean_bins}")

# Get corrupted actions
conditions = {
    'fog': apply_fog(base_img),
    'night': apply_night(base_img),
    'noise': apply_noise(base_img),
    'blur': apply_blur(base_img)
}

results = {"clean": {"tokens": clean_tokens, "bins": clean_bins}}

for name, img in conditions.items():
    tokens = get_actions(img)
    bins = [t - 31744 for t in tokens]
    diffs = [b - c for b, c in zip(bins, clean_bins)]
    n_changed = sum(1 for d in diffs if d != 0)
    
    results[name] = {
        "tokens": tokens,
        "bins": bins,
        "diffs": diffs,
        "n_changed": n_changed,
        "total_deviation": sum(abs(d) for d in diffs),
        "max_deviation": max(abs(d) for d in diffs)
    }
    print(f"\n  {name}:")
    print(f"    Tokens: {tokens}")
    print(f"    Bins: {bins}")
    print(f"    Diffs: {diffs}")
    print(f"    Changed: {n_changed}/7, total deviation: {results[name]['total_deviation']}")

# Also test with multiple runs for consistency
print("\n--- Consistency test (5 runs each) ---")
consistency = {}
for name, img in conditions.items():
    runs = []
    for _ in range(5):
        tokens = get_actions(img)
        runs.append(tokens)
    # Check if all runs produce same tokens
    consistent = all(r == runs[0] for r in runs)
    consistency[name] = {
        "consistent": consistent,
        "runs": runs
    }
    print(f"  {name}: consistent={consistent}")

results["consistency"] = consistency

output = {
    "experiment": "full_action",
    "experiment_number": 259,
    "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    "results": results
}

path = f"/workspace/Vizuara-VLA-Research/experiments/full_action_{output['timestamp']}.json"
with open(path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {path}")
