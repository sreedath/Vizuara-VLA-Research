"""Experiment 252: Multi-Token Position Analysis
Analyzes how OOD signal distributes across different token positions
(BOS, image tokens, text tokens) at layers 3, 15, and 31.
"""
import torch, json, numpy as np
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import time

print("=" * 60)
print("Experiment 252: Multi-Token Position Analysis")
print("=" * 60)

print("Loading OpenVLA-7B...")
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", torch_dtype=torch.bfloat16,
    device_map="auto", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
model.eval()

prompt = "In: What action should the robot take to pick up the object?\nOut:"
base_img = Image.fromarray(np.random.RandomState(42).randint(0, 256, (256, 256, 3), dtype=np.uint8))

def apply_fog(img, s=0.4):
    arr = np.array(img).astype(np.float32)
    return Image.fromarray((arr * (1-s) + 120*s).clip(0,255).astype(np.uint8))

def apply_night(img, f=0.15):
    return Image.fromarray((np.array(img).astype(np.float32) * f).clip(0,255).astype(np.uint8))

def apply_noise(img, std=50):
    arr = np.array(img).astype(np.float32)
    arr += np.random.RandomState(99).randn(*arr.shape) * std
    return Image.fromarray(arr.clip(0,255).astype(np.uint8))

layers = [3, 15, 31]

def extract_full_hidden(image):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    n_tokens = fwd.hidden_states[0].shape[1]
    result = {}
    for l in layers:
        result[l] = fwd.hidden_states[l][0, :, :].float().cpu().numpy()  # (n_tokens, 4096)
    return result, n_tokens

def cosine_dist(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(1 - np.dot(a, b) / (na * nb))

# Get clean reference
print("Extracting clean reference...")
clean_h, n_tokens = extract_full_hidden(base_img)
print(f"  Total tokens: {n_tokens}")

# Test positions: first, middle, last, and sampled image/text positions
# Typically: BOS(0), image tokens (1 to ~256), text tokens (~257+)
positions_to_test = [0, 1, 2, 5, 10, 50, 100, 128, 200, 255, 256, -5, -3, -2, -1]

corruptions = {
    'fog': apply_fog(base_img),
    'night': apply_night(base_img),
    'noise': apply_noise(base_img)
}

results = {}
for cname, cimg in corruptions.items():
    print(f"\n--- {cname} ---")
    corrupt_h, _ = extract_full_hidden(cimg)
    
    layer_results = {}
    for l in layers:
        pos_results = {}
        for p in positions_to_test:
            actual_p = p if p >= 0 else n_tokens + p
            if actual_p >= n_tokens or actual_p < 0:
                continue
            d = cosine_dist(clean_h[l][actual_p], corrupt_h[l][actual_p])
            pos_results[str(p)] = round(d, 8)
            print(f"  L{l} pos={p} (actual={actual_p}): d={d:.8f}")
        layer_results[str(l)] = pos_results
    results[cname] = layer_results

output = {
    "experiment": "token_positions",
    "experiment_number": 252,
    "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    "n_tokens": n_tokens,
    "layers": layers,
    "positions_tested": positions_to_test,
    "results": results
}

path = f"/workspace/Vizuara-VLA-Research/experiments/token_positions_{output['timestamp']}.json"
with open(path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {path}")
