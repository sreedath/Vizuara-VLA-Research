#!/usr/bin/env python3
"""Experiment 267: Prompt Sensitivity Analysis
Tests how much the OOD detection signal depends on the specific prompt text.
Uses 10 diverse prompts spanning different tasks and phrasings.
"""
import torch, json, numpy as np
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor
from datetime import datetime

def extract_hidden(model, processor, image, prompt, layer=3):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return fwd.hidden_states[layer][0, -1, :].float().cpu().numpy()

def cosine_distance(a, b):
    return 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def apply_corruption(img, ctype, severity=1.0):
    arr = np.array(img).astype(np.float32) / 255.0
    if ctype == 'fog':
        arr = arr * (1 - 0.6 * severity) + 0.6 * severity
    elif ctype == 'night':
        arr = arr * max(0.01, 1.0 - 0.95 * severity)
    elif ctype == 'noise':
        arr = arr + np.random.RandomState(42).randn(*arr.shape) * 0.3 * severity
        arr = np.clip(arr, 0, 1)
    elif ctype == 'blur':
        return img.filter(ImageFilter.GaussianBlur(radius=10 * severity))
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))

print("=" * 60)
print("Experiment 267: Prompt Sensitivity Analysis")
print("=" * 60)

print("Loading OpenVLA-7B...")
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", torch_dtype=torch.bfloat16,
    device_map="auto", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
model.eval()

prompts = [
    "In: What action should the robot take to pick up the object?\nOut:",
    "In: What action should the robot take to move forward?\nOut:",
    "In: What action should the robot take to turn left?\nOut:",
    "In: What action should the robot take to grasp the cup?\nOut:",
    "In: What action should the robot take to open the drawer?\nOut:",
    "In: What action should the robot take to push the button?\nOut:",
    "In: What action should the robot take to place the block?\nOut:",
    "In: What action should the robot take to stack the objects?\nOut:",
    "In: What action should the robot take to wipe the surface?\nOut:",
    "In: What action should the robot take to close the lid?\nOut:",
]

np.random.seed(42)
pixels = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
img = Image.fromarray(pixels)

corruptions = ['fog', 'night', 'noise', 'blur']
layers = [3, 15, 31]

results = {}

for li in layers:
    print(f"\n=== Layer {li} ===")
    layer_results = {}

    for pi, prompt in enumerate(prompts):
        prompt_key = f"p{pi}"
        h_clean = extract_hidden(model, processor, img, prompt, layer=li)

        distances = {}
        for ctype in corruptions:
            corrupted = apply_corruption(img, ctype)
            h_corr = extract_hidden(model, processor, corrupted, prompt, layer=li)
            d = cosine_distance(h_clean, h_corr)
            distances[ctype] = float(d)

        layer_results[prompt_key] = {
            'prompt': prompt[:60],
            'distances': distances
        }

        if pi < 3:
            print(f"  P{pi}: " + ", ".join(f"{c}={d:.6f}" for c, d in distances.items()))

    # Compute cross-prompt consistency
    for ctype in corruptions:
        dists = [layer_results[f"p{pi}"]["distances"][ctype] for pi in range(len(prompts))]
        cv = np.std(dists) / np.mean(dists) if np.mean(dists) > 0 else 0
        print(f"  L{li} {ctype}: mean={np.mean(dists):.6f}, std={np.std(dists):.6f}, CV={cv:.4f}")

    results[str(li)] = layer_results

# Cross-prompt centroid test
print("\n=== CROSS-PROMPT CENTROID TEST ===")
cross_prompt = {}
for li in [3]:
    for pi_cal in range(3):
        h_cal = extract_hidden(model, processor, img, prompts[pi_cal], layer=li)
        for pi_test in range(3):
            for ctype in ['fog', 'night']:
                corrupted = apply_corruption(img, ctype)
                h_test = extract_hidden(model, processor, corrupted, prompts[pi_test], layer=li)
                d = cosine_distance(h_cal, h_test)
                key = f"cal_p{pi_cal}_test_p{pi_test}_{ctype}"
                cross_prompt[key] = float(d)
                if pi_cal != pi_test:
                    print(f"  Cross: cal=P{pi_cal}, test=P{pi_test}, {ctype}: d={d:.6f}")

results['cross_prompt'] = cross_prompt

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
out = {
    'experiment': 'prompt_sensitivity',
    'experiment_number': 267,
    'timestamp': ts,
    'n_prompts': len(prompts),
    'results': results
}

path = f'/workspace/Vizuara-VLA-Research/experiments/prompt_sensitivity_{ts}.json'
with open(path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {path}")
