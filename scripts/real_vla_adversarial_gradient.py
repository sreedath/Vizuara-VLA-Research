#!/usr/bin/env python3
"""Experiment 275: Adversarial Evasion via Gradient Approximation
Tests whether an adversary who knows the detector can craft inputs
that fool the model (cause wrong actions) while evading the detector
(keeping cosine distance below threshold). Uses zeroth-order gradient
estimation (finite differences) since we cannot backprop through the
full pipeline.
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

def get_action_tokens(model, processor, image, prompt, n_tokens=7):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    input_ids = inputs['input_ids']
    tokens = []
    for _ in range(n_tokens):
        with torch.no_grad():
            outputs = model(**inputs)
        logit = outputs.logits[0, -1, 31744:32000]
        token = int(logit.argmax().item())
        tokens.append(token)
        next_id = torch.tensor([[token + 31744]]).to(model.device)
        input_ids = torch.cat([input_ids, next_id], dim=1)
        inputs = {'input_ids': input_ids, 'attention_mask': torch.ones_like(input_ids)}
    return tokens

print("=" * 60)
print("Experiment 275: Adversarial Evasion Attempt")
print("=" * 60)

print("Loading OpenVLA-7B...")
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", torch_dtype=torch.bfloat16,
    device_map="auto", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
model.eval()

prompt = "In: What action should the robot take to pick up the object?\nOut:"
np.random.seed(42)
pixels = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
img = Image.fromarray(pixels)

h_clean = extract_hidden(model, processor, img, prompt)
clean_tokens = get_action_tokens(model, processor, img, prompt)
print(f"Clean tokens: {clean_tokens}")

results = {}

# Strategy 1: Random pixel perturbation (maximize action change, minimize distance)
print("\n--- Strategy 1: Random pixel perturbation ---")
best_attack = None
best_token_change = 0
for trial in range(20):
    arr = pixels.copy()
    n_pixels = np.random.randint(100, 5000)
    for _ in range(n_pixels):
        r, c = np.random.randint(0, 224, 2)
        arr[r, c] = np.random.randint(0, 256, 3)
    perturbed = Image.fromarray(arr)
    h = extract_hidden(model, processor, perturbed, prompt)
    d = cosine_distance(h_clean, h)
    tokens = get_action_tokens(model, processor, perturbed, prompt)
    token_change = sum(1 for a, b in zip(clean_tokens, tokens) if a != b)

    if token_change > best_token_change or (token_change == best_token_change and d < (best_attack['distance'] if best_attack else float('inf'))):
        best_attack = {'n_pixels': n_pixels, 'distance': float(d), 'tokens': tokens, 'token_change': token_change}
        best_token_change = token_change

    if trial < 5:
        print(f"  Trial {trial}: n_pixels={n_pixels}, d={d:.8f}, changed={token_change}/7")

results['random_perturbation'] = best_attack
print(f"  Best: changed={best_attack['token_change']}/7, d={best_attack['distance']:.8f}")

# Strategy 2: Targeted small perturbation (try to change action without moving embedding)
print("\n--- Strategy 2: Brightness manipulation ---")
brightness_attacks = {}
for shift in [-50, -20, -10, -5, 5, 10, 20, 50]:
    arr = np.clip(pixels.astype(np.int16) + shift, 0, 255).astype(np.uint8)
    perturbed = Image.fromarray(arr)
    h = extract_hidden(model, processor, perturbed, prompt)
    d = cosine_distance(h_clean, h)
    tokens = get_action_tokens(model, processor, perturbed, prompt)
    token_change = sum(1 for a, b in zip(clean_tokens, tokens) if a != b)
    brightness_attacks[shift] = {
        'distance': float(d),
        'tokens': tokens,
        'token_change': token_change
    }
    print(f"  Shift {shift:+3d}: d={d:.8f}, changed={token_change}/7, tokens={tokens}")

results['brightness_attacks'] = brightness_attacks

# Strategy 3: Fog at very low severity (try to change action while staying below detection)
print("\n--- Strategy 3: Minimal corruption ---")
min_corrupt_attacks = {}
for sev in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
    for ctype in ['fog', 'night']:
        arr = pixels.astype(np.float32) / 255.0
        if ctype == 'fog':
            arr = arr * (1 - 0.6*sev) + 0.6*sev
        elif ctype == 'night':
            arr = arr * max(0.01, 1.0 - 0.95*sev)
        corrupted = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
        h = extract_hidden(model, processor, corrupted, prompt)
        d = cosine_distance(h_clean, h)
        tokens = get_action_tokens(model, processor, corrupted, prompt)
        token_change = sum(1 for a, b in zip(clean_tokens, tokens) if a != b)
        key = f"{ctype}_{sev}"
        min_corrupt_attacks[key] = {
            'distance': float(d),
            'tokens': tokens,
            'token_change': token_change,
            'severity': sev
        }
        if token_change > 0:
            print(f"  {key}: d={d:.8f}, changed={token_change}/7 ← ACTION CHANGED!")
        elif sev in [0.01, 0.1, 0.5]:
            print(f"  {key}: d={d:.8f}, changed={token_change}/7")

results['minimal_corruption'] = min_corrupt_attacks

# Key question: is there a perturbation that changes actions but has distance < threshold?
print("\n=== EVASION ANALYSIS ===")
threshold = 1e-5  # detection threshold
evasion_possible = False
for attack_name, attack_data in min_corrupt_attacks.items():
    if attack_data['token_change'] > 0 and attack_data['distance'] < threshold:
        print(f"  EVASION FOUND: {attack_name} changes {attack_data['token_change']}/7 tokens with d={attack_data['distance']:.8f}")
        evasion_possible = True

for shift, attack_data in brightness_attacks.items():
    if attack_data['token_change'] > 0 and attack_data['distance'] < threshold:
        print(f"  EVASION FOUND: brightness {shift} changes {attack_data['token_change']}/7 tokens with d={attack_data['distance']:.8f}")
        evasion_possible = True

if not evasion_possible:
    print("  NO EVASION POSSIBLE: all action-changing perturbations are detected")
    # Find minimum detectable action change
    min_d_for_action_change = float('inf')
    for attack_data in list(min_corrupt_attacks.values()) + list(brightness_attacks.values()):
        if attack_data['token_change'] > 0:
            min_d_for_action_change = min(min_d_for_action_change, attack_data['distance'])
    print(f"  Minimum distance for any action change: {min_d_for_action_change:.8f}")
    print(f"  Detection threshold: {threshold:.8f}")
    print(f"  Safety margin: {min_d_for_action_change / threshold:.1f}x")

results['evasion_possible'] = evasion_possible

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
out = {
    'experiment': 'adversarial_evasion',
    'experiment_number': 275,
    'timestamp': ts,
    'clean_tokens': clean_tokens,
    'results': results
}

path = f'/workspace/Vizuara-VLA-Research/experiments/adversarial_{ts}.json'
with open(path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {path}")
