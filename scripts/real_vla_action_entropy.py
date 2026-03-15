#!/usr/bin/env python3
"""Experiment 269: Action Token Entropy Across Dimensions
Measures per-dimension action token entropy and confidence under
corruption. Tests whether some action dimensions are more sensitive
to corruption than others in terms of output uncertainty.
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
print("Experiment 269: Action Token Entropy Across Dimensions")
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

ACTION_START = 31744
ACTION_END = 32000
N_DIMS = 7

conditions = {
    'clean': img,
    'fog': apply_corruption(img, 'fog'),
    'night': apply_corruption(img, 'night'),
    'noise': apply_corruption(img, 'noise'),
    'blur': apply_corruption(img, 'blur'),
}

results = {}

for cname, cimg in conditions.items():
    print(f"\n--- {cname} ---")
    inputs = processor(prompt, cimg).to(model.device, dtype=torch.bfloat16)
    input_ids = inputs['input_ids']

    dim_stats = {}
    generated_tokens = []

    for dim in range(N_DIMS):
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, ACTION_START:ACTION_END].float().cpu().numpy()

        # Softmax
        logits_shifted = logits - logits.max()
        probs = np.exp(logits_shifted) / np.exp(logits_shifted).sum()

        # Entropy
        entropy = -np.sum(probs[probs > 0] * np.log(probs[probs > 0]))
        max_prob = float(probs.max())
        top1_token = int(np.argmax(probs))
        top5_tokens = list(np.argsort(probs)[::-1][:5])
        top5_probs = [float(probs[t]) for t in top5_tokens]

        dim_stats[dim] = {
            'entropy': float(entropy),
            'max_prob': max_prob,
            'top1_token': top1_token,
            'top5_tokens': top5_tokens,
            'top5_probs': top5_probs
        }

        print(f"  dim{dim}: entropy={entropy:.3f}, top1={top1_token} (p={max_prob:.3f})")

        # Append predicted token for autoregressive generation
        next_token = torch.tensor([[top1_token + ACTION_START]]).to(model.device)
        generated_tokens.append(top1_token)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        inputs = {'input_ids': input_ids, 'attention_mask': torch.ones_like(input_ids)}

    results[cname] = {
        'dim_stats': dim_stats,
        'generated_tokens': generated_tokens
    }

# Compare entropies
print("\n=== ENTROPY COMPARISON ===")
for dim in range(N_DIMS):
    clean_ent = results['clean']['dim_stats'][dim]['entropy']
    for cname in ['fog', 'night', 'noise', 'blur']:
        corr_ent = results[cname]['dim_stats'][dim]['entropy']
        pct = (corr_ent - clean_ent) / clean_ent * 100
        print(f"  dim{dim} {cname}: clean={clean_ent:.3f}, corrupt={corr_ent:.3f}, change={pct:+.1f}%")

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
out = {
    'experiment': 'action_entropy',
    'experiment_number': 269,
    'timestamp': ts,
    'results': results
}

path = f'/workspace/Vizuara-VLA-Research/experiments/action_entropy_{ts}.json'
with open(path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {path}")
