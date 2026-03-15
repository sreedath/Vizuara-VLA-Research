#!/usr/bin/env python3
"""Experiment 279: Token Position Trajectory Analysis
Tracks how the OOD signal varies across ALL token positions (not just last)
at L3, creating a complete spatial map of where corruption information
resides in the sequence. Tests image tokens, text tokens, and BOS.
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
print("Experiment 279: Token Position Trajectory Analysis")
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

# Get full hidden states for ALL token positions at layers 1, 3, 7, 15
target_layers = [1, 3, 7, 15]

results = {}

# First: clean
print("\n--- Clean ---")
inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)
with torch.no_grad():
    fwd_clean = model(**inputs, output_hidden_states=True)

seq_len = fwd_clean.hidden_states[0].shape[1]
print(f"  Sequence length: {seq_len}")

# Store clean hidden states for target layers (all positions)
clean_hidden = {}
for l in target_layers:
    clean_hidden[l] = fwd_clean.hidden_states[l][0].float().cpu().numpy()  # (seq_len, 4096)
    print(f"  L{l}: shape={clean_hidden[l].shape}")

# For each corruption, compute per-position cosine distance
for ctype, cimg in corruptions.items():
    print(f"\n--- {ctype} ---")
    inputs = processor(prompt, cimg).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)

    pos_distances = {}
    for l in target_layers:
        corr_hidden = fwd.hidden_states[l][0].float().cpu().numpy()  # (seq_len, 4096)

        # Compute cosine distance at each position
        distances = []
        for pos in range(seq_len):
            c = clean_hidden[l][pos]
            r = corr_hidden[pos]
            d = 1.0 - np.dot(c, r) / (np.linalg.norm(c) * np.linalg.norm(r) + 1e-30)
            distances.append(float(d))

        pos_distances[f'L{l}'] = distances

        # Summary stats
        img_start, img_end = 1, min(257, seq_len)
        img_dists = distances[img_start:img_end]
        text_dists = distances[img_end:]
        bos_d = distances[0]

        print(f"  L{l}: BOS={bos_d:.6f}, img_mean={np.mean(img_dists):.6f}, "
              f"img_max={np.max(img_dists):.6f}, text_mean={np.mean(text_dists):.6f}, "
              f"last={distances[-1]:.6f}")

    results[ctype] = pos_distances

# Analyze signal propagation from image tokens to text tokens
print("\n=== SIGNAL PROPAGATION ===")
propagation = {}
for ctype in corruptions:
    prop = {}
    for l in target_layers:
        dists = results[ctype][f'L{l}']
        img_dists = dists[1:min(257, seq_len)]
        text_dists = dists[min(257, seq_len):]

        # Image-to-text signal ratio
        img_mean = np.mean(img_dists)
        text_mean = np.mean(text_dists) if text_dists else 0
        ratio = img_mean / (text_mean + 1e-30)

        # Find position with max distance
        max_pos = int(np.argmax(dists))

        # Signal uniformity across image tokens
        img_std = np.std(img_dists)
        img_cv = img_std / (img_mean + 1e-30)

        prop[f'L{l}'] = {
            'img_mean': float(img_mean),
            'img_std': float(img_std),
            'img_cv': float(img_cv),
            'img_max': float(np.max(img_dists)),
            'text_mean': float(text_mean),
            'img_to_text_ratio': float(ratio),
            'max_position': max_pos,
            'bos_distance': float(dists[0]),
            'last_token_distance': float(dists[-1])
        }
        print(f"  {ctype} L{l}: img/text ratio={ratio:.1f}×, "
              f"img CV={img_cv:.3f}, max_pos={max_pos}")

    propagation[ctype] = prop

# Spatial structure within image tokens
print("\n=== SPATIAL STRUCTURE IN IMAGE TOKENS ===")
spatial = {}
for ctype in corruptions:
    l3_dists = results[ctype]['L3'][1:min(257, seq_len)]
    n_img = len(l3_dists)

    # Image tokens are typically arranged as 16x16 patches
    if n_img == 256:
        grid = np.array(l3_dists).reshape(16, 16)
        row_means = grid.mean(axis=1).tolist()
        col_means = grid.mean(axis=0).tolist()
        spatial[ctype] = {
            'grid_row_means': [float(v) for v in row_means],
            'grid_col_means': [float(v) for v in col_means],
            'center_mean': float(grid[4:12, 4:12].mean()),
            'border_mean': float(np.concatenate([grid[0,:], grid[-1,:], grid[1:-1,0], grid[1:-1,-1]]).mean()),
            'center_vs_border_ratio': float(grid[4:12, 4:12].mean() / (np.concatenate([grid[0,:], grid[-1,:], grid[1:-1,0], grid[1:-1,-1]]).mean() + 1e-30))
        }
        print(f"  {ctype}: center={grid[4:12, 4:12].mean():.6f}, border={spatial[ctype]['border_mean']:.6f}")
    else:
        # Just store quartiles
        q1 = l3_dists[:n_img//4]
        q4 = l3_dists[3*n_img//4:]
        spatial[ctype] = {
            'first_quarter_mean': float(np.mean(q1)),
            'last_quarter_mean': float(np.mean(q4)),
            'n_image_tokens': n_img
        }

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
out = {
    'experiment': 'token_trajectory',
    'experiment_number': 279,
    'timestamp': ts,
    'seq_len': seq_len,
    'target_layers': target_layers,
    'results': {
        'per_position_distances': results,
        'propagation': propagation,
        'spatial': spatial
    }
}

path = f'/workspace/Vizuara-VLA-Research/experiments/trajectory_{ts}.json'
with open(path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {path}")
