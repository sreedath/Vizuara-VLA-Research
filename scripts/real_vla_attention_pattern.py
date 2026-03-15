#!/usr/bin/env python3
"""Experiment 266: Attention Pattern Analysis Under Corruption
Examines how attention patterns change under corruption by computing
attention entropy, max attention weight, and attention distribution
statistics across heads and layers.
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
print("Experiment 266: Attention Pattern Analysis")
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

    with torch.no_grad():
        fwd = model(**inputs, output_attentions=True, output_hidden_states=True)

    # fwd.attentions is a tuple of (n_layers,) each with shape (batch, n_heads, seq_len, seq_len)
    n_layers = len(fwd.attentions)
    n_heads = fwd.attentions[0].shape[1]
    seq_len = fwd.attentions[0].shape[2]

    print(f"  n_layers={n_layers}, n_heads={n_heads}, seq_len={seq_len}")

    layer_stats = {}
    for li in [0, 3, 7, 15, 23, 31]:
        if li >= n_layers:
            continue
        attn = fwd.attentions[li][0].float().cpu().numpy()  # (n_heads, seq_len, seq_len)

        # Focus on last token's attention (what the last token attends to)
        last_token_attn = attn[:, -1, :]  # (n_heads, seq_len)

        # Entropy per head
        entropies = []
        for h in range(n_heads):
            p = last_token_attn[h]
            p = p[p > 0]  # avoid log(0)
            ent = -np.sum(p * np.log(p))
            entropies.append(float(ent))

        # Max attention weight per head
        max_weights = [float(last_token_attn[h].max()) for h in range(n_heads)]

        # Attention to image tokens (positions 1 to ~257) vs text tokens
        # Image tokens are roughly positions 1-256
        img_attn_frac = [float(last_token_attn[h, 1:257].sum()) for h in range(n_heads)]

        layer_stats[li] = {
            'mean_entropy': float(np.mean(entropies)),
            'std_entropy': float(np.std(entropies)),
            'min_entropy': float(np.min(entropies)),
            'max_entropy': float(np.max(entropies)),
            'mean_max_weight': float(np.mean(max_weights)),
            'mean_img_attn_frac': float(np.mean(img_attn_frac)),
            'entropies': entropies[:8],  # first 8 heads
            'max_weights': max_weights[:8]
        }

        print(f"  L{li}: entropy={np.mean(entropies):.3f}±{np.std(entropies):.3f}, "
              f"max_wt={np.mean(max_weights):.4f}, img_frac={np.mean(img_attn_frac):.4f}")

    results[cname] = layer_stats

# Compute attention divergence (clean vs corrupted)
print("\n=== ATTENTION DIVERGENCE ===")
divergences = {}
for cname in ['fog', 'night', 'noise', 'blur']:
    div_by_layer = {}
    for li in [0, 3, 7, 15, 23, 31]:
        if str(li) not in str(results['clean'].keys()):
            continue
        clean_ent = results['clean'][li]['mean_entropy']
        corr_ent = results[cname][li]['mean_entropy']
        pct_change = (corr_ent - clean_ent) / clean_ent * 100
        div_by_layer[li] = {
            'entropy_pct_change': float(pct_change),
            'clean_entropy': float(clean_ent),
            'corrupted_entropy': float(corr_ent)
        }
        print(f"  {cname} L{li}: entropy change = {pct_change:+.1f}%")
    divergences[cname] = div_by_layer

results['divergences'] = divergences

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
out = {
    'experiment': 'attention_pattern',
    'experiment_number': 266,
    'timestamp': ts,
    'results': results
}

path = f'/workspace/Vizuara-VLA-Research/experiments/attention_{ts}.json'
with open(path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {path}")
