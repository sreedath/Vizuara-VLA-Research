#!/usr/bin/env python3
"""Experiment 303: Activation Pattern Analysis
Examines internal activation patterns under corruption:
1. ReLU/activation sparsity changes per layer
2. Activation magnitude statistics
3. Dead neuron detection under corruption
4. Activation cosine similarity between clean and corrupted
5. Layer-wise activation norm profiles
"""

import torch
import numpy as np
import json
from datetime import datetime
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor
from scipy.spatial.distance import cosine

def apply_corruption(image, ctype, severity=1.0):
    arr = np.array(image).astype(np.float32) / 255.0
    if ctype == 'fog':
        arr = arr * (1 - 0.6 * severity) + 0.6 * severity
    elif ctype == 'night':
        arr = arr * max(0.01, 1.0 - 0.95 * severity)
    elif ctype == 'noise':
        arr = arr + np.random.RandomState(42).randn(*arr.shape) * 0.3 * severity
        arr = np.clip(arr, 0, 1)
    elif ctype == 'blur':
        return image.filter(ImageFilter.GaussianBlur(radius=10 * severity))
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))

def main():
    print("Loading OpenVLA-7B...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    np.random.seed(42)
    pixels = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    base_img = Image.fromarray(pixels)

    results = {
        "experiment": "activation_patterns",
        "experiment_number": 303,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    corruptions = ['fog', 'night', 'blur', 'noise']
    target_layers = [0, 3, 7, 15, 31]

    # Get clean hidden states
    print("Getting clean activations...")
    inputs = processor(prompt, base_img).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)

    n_layers = len(fwd.hidden_states)
    clean_states = {}
    clean_stats = {}

    for li in target_layers:
        if li < n_layers:
            # Full sequence hidden states
            h = fwd.hidden_states[li][0].float().cpu().numpy()  # (seq_len, hidden_dim)
            clean_states[li] = h
            
            # Last token stats
            last = h[-1]
            clean_stats[f"L{li}"] = {
                "mean": float(np.mean(last)),
                "std": float(np.std(last)),
                "min": float(np.min(last)),
                "max": float(np.max(last)),
                "norm": float(np.linalg.norm(last)),
                "sparsity": float(np.mean(np.abs(last) < 0.01)),
                "positive_frac": float(np.mean(last > 0)),
                "negative_frac": float(np.mean(last < 0)),
            }

    results["clean_stats"] = clean_stats

    # Part 1: Per-layer activation comparison
    print("\n=== Part 1: Activation Comparison ===")
    activation_comparison = {}

    for c in corruptions:
        print(f"  {c}...")
        corrupted = apply_corruption(base_img, c, 1.0)
        inputs = processor(prompt, corrupted).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            fwd_c = model(**inputs, output_hidden_states=True)

        activation_comparison[c] = {}
        for li in target_layers:
            if li < n_layers:
                h_corr = fwd_c.hidden_states[li][0].float().cpu().numpy()
                last_clean = clean_states[li][-1]
                last_corr = h_corr[-1]

                # Comparison metrics
                cos_d = float(cosine(last_clean, last_corr))
                l2_d = float(np.linalg.norm(last_clean - last_corr))

                # Sparsity change
                clean_sparse = np.mean(np.abs(last_clean) < 0.01)
                corr_sparse = np.mean(np.abs(last_corr) < 0.01)

                # Sign flips
                sign_flips = np.mean(np.sign(last_clean) != np.sign(last_corr))

                # Magnitude change
                mag_ratio = float(np.linalg.norm(last_corr) / (np.linalg.norm(last_clean) + 1e-30))

                # Per-dimension correlation
                corr_coef = float(np.corrcoef(last_clean, last_corr)[0, 1])

                activation_comparison[c][f"L{li}"] = {
                    "cosine_distance": cos_d,
                    "l2_distance": l2_d,
                    "sparsity_clean": float(clean_sparse),
                    "sparsity_corrupted": float(corr_sparse),
                    "sparsity_change": float(corr_sparse - clean_sparse),
                    "sign_flip_rate": float(sign_flips),
                    "magnitude_ratio": mag_ratio,
                    "correlation": corr_coef,
                    "corr_mean": float(np.mean(last_corr)),
                    "corr_std": float(np.std(last_corr))
                }

        # Print L3 stats
        l3 = activation_comparison[c].get("L3", {})
        print(f"    L3: cos_d={l3.get('cosine_distance', 0):.6f}, sign_flips={l3.get('sign_flip_rate', 0):.3f}, "
              f"corr={l3.get('correlation', 0):.6f}")

    results["activation_comparison"] = activation_comparison

    # Part 2: Sequence-level analysis (image vs text tokens)
    print("\n=== Part 2: Sequence Position Analysis ===")
    seq_analysis = {}

    for c in ['fog', 'night']:
        corrupted = apply_corruption(base_img, c, 1.0)
        inputs = processor(prompt, corrupted).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            fwd_c = model(**inputs, output_hidden_states=True)

        seq_analysis[c] = {}
        for li in [3, 15, 31]:
            if li < n_layers:
                clean_seq = clean_states[li]  # (seq_len, hidden_dim)
                corr_seq = fwd_c.hidden_states[li][0].float().cpu().numpy()

                # Per-position cosine distance
                seq_len = min(clean_seq.shape[0], corr_seq.shape[0])
                pos_dists = []
                for pos in range(seq_len):
                    d = float(cosine(clean_seq[pos], corr_seq[pos]))
                    pos_dists.append(d)

                # Segment analysis
                # First 10 tokens (text), middle (image), last 10 (text)
                first10 = np.mean(pos_dists[:10])
                mid = np.mean(pos_dists[10:-10]) if len(pos_dists) > 20 else 0
                last10 = np.mean(pos_dists[-10:])

                seq_analysis[c][f"L{li}"] = {
                    "mean_dist": float(np.mean(pos_dists)),
                    "max_dist": float(np.max(pos_dists)),
                    "min_dist": float(np.min(pos_dists)),
                    "first10_mean": float(first10),
                    "middle_mean": float(mid),
                    "last10_mean": float(last10),
                    "max_pos": int(np.argmax(pos_dists)),
                    "n_positions": len(pos_dists)
                }
                print(f"  {c} L{li}: first10={first10:.6f}, middle={mid:.6f}, last10={last10:.6f}")

    results["sequence_analysis"] = seq_analysis

    # Part 3: Activation distribution shift
    print("\n=== Part 3: Distribution Shift ===")
    dist_shift = {}

    for c in corruptions:
        corrupted = apply_corruption(base_img, c, 1.0)
        inputs = processor(prompt, corrupted).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            fwd_c = model(**inputs, output_hidden_states=True)

        dist_shift[c] = {}
        for li in [3, 15, 31]:
            if li < n_layers:
                clean_h = clean_states[li][-1]
                corr_h = fwd_c.hidden_states[li][0, -1, :].float().cpu().numpy()

                # Moment comparison
                moments = {
                    "mean_shift": float(np.mean(corr_h) - np.mean(clean_h)),
                    "std_shift": float(np.std(corr_h) - np.std(clean_h)),
                    "skew_clean": float(np.mean(((clean_h - clean_h.mean()) / (clean_h.std() + 1e-10))**3)),
                    "skew_corrupted": float(np.mean(((corr_h - corr_h.mean()) / (corr_h.std() + 1e-10))**3)),
                    "kurtosis_clean": float(np.mean(((clean_h - clean_h.mean()) / (clean_h.std() + 1e-10))**4) - 3),
                    "kurtosis_corrupted": float(np.mean(((corr_h - corr_h.mean()) / (corr_h.std() + 1e-10))**4) - 3),
                }
                dist_shift[c][f"L{li}"] = moments

        print(f"  {c} L3: mean_shift={dist_shift[c]['L3']['mean_shift']:.6f}, "
              f"std_shift={dist_shift[c]['L3']['std_shift']:.6f}")

    results["distribution_shift"] = dist_shift

    # Save
    ts = results["timestamp"]
    out_path = f"experiments/activation_{ts}.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
