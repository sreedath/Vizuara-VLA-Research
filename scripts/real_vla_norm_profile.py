"""Experiment 261: L2 Norm Profile Across All Layers
Measures the L2 norm of the hidden state at each of the 33 layers
for clean and corrupted images.
"""
import torch, json, numpy as np
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image, ImageFilter
import time

print("=" * 60)
print("Experiment 261: L2 Norm Profile")
print("=" * 60)

print("Loading OpenVLA-7B...")
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", torch_dtype=torch.bfloat16,
    device_map="auto", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
model.eval()

prompt = "In: What action should the robot take to pick up the object?\nOut:"
base_img = Image.fromarray(np.random.RandomState(42).randint(0, 256, (256, 256, 3), dtype=np.uint8))

all_layers = list(range(33))

def extract_norms(image):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return {l: round(float(torch.norm(fwd.hidden_states[l][0, -1, :].float()).item()), 4) for l in all_layers}

# Corruptions
def apply_fog(img): return Image.fromarray((np.array(img).astype(np.float32)*0.6+120*0.4).clip(0,255).astype(np.uint8))
def apply_night(img): return Image.fromarray((np.array(img).astype(np.float32)*0.15).clip(0,255).astype(np.uint8))
def apply_noise(img):
    a = np.array(img).astype(np.float32); a += np.random.RandomState(99).randn(*a.shape)*50
    return Image.fromarray(a.clip(0,255).astype(np.uint8))
def apply_blur(img): return img.filter(ImageFilter.GaussianBlur(radius=5))

# Get norms
print("Clean norms...")
clean_norms = extract_norms(base_img)
for l in all_layers:
    print(f"  L{l}: {clean_norms[l]:.4f}")

corruptions = {'fog': apply_fog, 'night': apply_night, 'noise': apply_noise, 'blur': apply_blur}
results = {"clean": clean_norms}

for name, fn in corruptions.items():
    print(f"\n--- {name} ---")
    norms = extract_norms(fn(base_img))
    results[name] = norms
    # Print differences
    for l in all_layers:
        diff = norms[l] - clean_norms[l]
        pct = (diff / clean_norms[l] * 100) if clean_norms[l] > 0 else 0
        if abs(pct) > 1:
            print(f"  L{l}: {norms[l]:.4f} ({pct:+.2f}%)")

output = {
    "experiment": "norm_profile",
    "experiment_number": 261,
    "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    "n_layers": 33,
    "results": results
}

path = f"/workspace/Vizuara-VLA-Research/experiments/norm_profile_{output['timestamp']}.json"
with open(path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {path}")
