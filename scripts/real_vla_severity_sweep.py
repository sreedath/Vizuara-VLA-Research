"""Experiment 250: Fine-Grained Severity Sweep
Maps the exact relationship between corruption severity (0-100%) and cosine distance
at 20 granularity levels for fog, night, noise, and blur.
"""
import torch, json, numpy as np
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image, ImageFilter
import time

print("=" * 60)
print("Experiment 250: Fine-Grained Severity Sweep")
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

# Severity levels: 0%, 5%, 10%, ..., 100%
severity_levels = [i/20 for i in range(21)]  # 0.0, 0.05, 0.10, ..., 1.0

def apply_fog(img, severity):
    arr = np.array(img).astype(np.float32)
    arr = arr * (1 - severity * 0.5) + 120 * severity * 0.5  # max 50% fog
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))

def apply_night(img, severity):
    factor = 1.0 - severity * 0.85  # range: 1.0 (clean) to 0.15 (full night)
    arr = (np.array(img).astype(np.float32) * factor).clip(0, 255)
    return Image.fromarray(arr.astype(np.uint8))

def apply_noise(img, severity):
    arr = np.array(img).astype(np.float32)
    std = severity * 50  # range: 0 to 50
    arr += np.random.RandomState(99).randn(*arr.shape) * std
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))

def apply_blur(img, severity):
    radius = severity * 5  # range: 0 to 5
    if radius < 0.1:
        return img
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

# Get reference
ref = extract_hidden(base_img)
print(f"Reference extracted")

results = {}
for name, apply_fn in [('fog', apply_fog), ('night', apply_night), ('noise', apply_noise), ('blur', apply_blur)]:
    print(f"\n--- {name} ---")
    distances = []
    action_tokens = []
    for sev in severity_levels:
        img = apply_fn(base_img, sev)
        h = extract_hidden(img)
        d = cosine_dist(ref, h)
        
        # Also get action token
        inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=1, do_sample=False)
        token = int(out[0, -1].item())
        
        distances.append(round(d, 8))
        action_tokens.append(token)
        print(f"  severity={sev:.2f}: d={d:.8f}, token={token}")
    
    # Find first severity where action changes
    clean_token = action_tokens[0]
    first_action_change = None
    for i, t in enumerate(action_tokens):
        if t != clean_token:
            first_action_change = round(severity_levels[i], 2)
            break
    
    results[name] = {
        "distances": distances,
        "action_tokens": action_tokens,
        "first_action_change": first_action_change,
        "clean_token": clean_token,
        "max_distance": round(max(distances), 8),
        "r_squared": None  # will compute
    }
    
    # Compute R² for linearity
    sev_arr = np.array(severity_levels)
    d_arr = np.array(distances)
    if d_arr.std() > 0:
        corr = np.corrcoef(sev_arr, d_arr)[0, 1]
        results[name]["r_squared"] = round(float(corr**2), 4)

output = {
    "experiment": "severity_sweep",
    "experiment_number": 250,
    "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    "n_levels": 21,
    "layer": 3,
    "results": results
}

path = f"/workspace/Vizuara-VLA-Research/experiments/severity_sweep_{output['timestamp']}.json"
with open(path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {path}")
