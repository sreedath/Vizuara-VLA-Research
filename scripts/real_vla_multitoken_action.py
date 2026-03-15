#!/usr/bin/env python3
"""Experiment 285: Multi-Token Action Sequence Detection
Tests whether OOD detection can be performed at each step of the 7-token
action sequence generation, not just before generation starts.
Compares detection at each autoregressive step.
"""
import torch, json, numpy as np
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor
from datetime import datetime

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
print("Experiment 285: Multi-Token Action Sequence Detection")
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

corruptions = {
    'fog': apply_corruption(img, 'fog'),
    'night': apply_corruption(img, 'night'),
    'noise': apply_corruption(img, 'noise'),
    'blur': apply_corruption(img, 'blur'),
}

# Generate 7 tokens autoregressively, extracting hidden states at each step
results = {}

for cname, cimg in [('clean', img)] + list(corruptions.items()):
    print(f"\n--- {cname} ---")
    inputs = processor(prompt, cimg).to(model.device, dtype=torch.bfloat16)
    input_ids = inputs['input_ids']
    
    step_data = []
    current_ids = input_ids.clone()
    
    for step in range(7):
        with torch.no_grad():
            fwd = model(input_ids=current_ids, attention_mask=torch.ones_like(current_ids),
                       pixel_values=inputs.get('pixel_values'),
                       output_hidden_states=True)
        
        # Hidden state at last position
        h_l3 = fwd.hidden_states[3][0, -1, :].float().cpu().numpy()
        
        # Next token (greedy)
        logits = fwd.logits[0, -1, :]
        next_token = int(logits.argmax())
        top1_prob = float(torch.softmax(logits.float(), dim=0).max())
        
        # Entropy
        probs = torch.softmax(logits.float(), dim=0)
        entropy = float(-torch.sum(probs * torch.log(probs + 1e-10)))
        
        step_data.append({
            'step': step,
            'token': next_token,
            'top1_prob': top1_prob,
            'entropy': entropy,
            'hidden_norm': float(np.linalg.norm(h_l3)),
            'hidden_l3': h_l3
        })
        
        # Append token for next step
        current_ids = torch.cat([current_ids, torch.tensor([[next_token]], device=current_ids.device)], dim=1)
        
        print(f"  Step {step}: token={next_token}, prob={top1_prob:.4f}, entropy={entropy:.3f}")
    
    results[cname] = step_data

# Compute per-step cosine distances
print("\n=== PER-STEP DETECTION ===")
step_distances = {}
for ctype in corruptions:
    dists = []
    for step in range(7):
        clean_h = results['clean'][step]['hidden_l3']
        corr_h = results[ctype][step]['hidden_l3']
        d = 1.0 - np.dot(clean_h, corr_h) / (np.linalg.norm(clean_h) * np.linalg.norm(corr_h) + 1e-30)
        dists.append(float(d))
    step_distances[ctype] = dists
    print(f"  {ctype}: {['%.6f' % d for d in dists]}")

# Token comparison
print("\n=== TOKEN COMPARISON ===")
token_comparison = {}
for ctype in corruptions:
    clean_tokens = [s['token'] for s in results['clean']]
    corr_tokens = [s['token'] for s in results[ctype]]
    matches = sum(1 for c, r in zip(clean_tokens, corr_tokens) if c == r)
    token_comparison[ctype] = {
        'clean_tokens': clean_tokens,
        'corrupted_tokens': corr_tokens,
        'matches': matches,
        'total': 7
    }
    print(f"  {ctype}: clean={clean_tokens}, corrupt={corr_tokens}, match={matches}/7")

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
# Clean up for serialization
clean_results = {}
for cname, steps in results.items():
    clean_results[cname] = [{k: v for k, v in s.items() if k != 'hidden_l3'} for s in steps]

out = {
    'experiment': 'multitoken_action',
    'experiment_number': 285,
    'timestamp': ts,
    'results': {
        'step_data': clean_results,
        'step_distances': step_distances,
        'token_comparison': token_comparison
    }
}

path = f'/workspace/Vizuara-VLA-Research/experiments/multitoken_{ts}.json'
with open(path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {path}")
