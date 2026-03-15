#!/usr/bin/env python3
"""Experiment 315: Architecture Probe — Why Does Zero-Variance Emerge?
Probes OpenVLA's architecture to understand the zero-variance phenomenon:
1. Layer-by-layer variance profile (all 33 layers)
2. Token-type variance (BOS, image, text separately)
3. Vision encoder vs language model contribution
4. Attention pattern determinism per layer
5. Weight norm and gradient structure at each layer
6. Intermediate activation statistics under corruption
"""

import torch
import numpy as np
import json
import time
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
        "experiment": "architecture_probe",
        "experiment_number": 315,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    # Part 1: Full layer profile — all hidden states
    print("=== Part 1: Full Layer Profile (Clean) ===")
    inputs = processor(prompt, base_img).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)

    n_layers = len(fwd.hidden_states)
    input_ids = inputs['input_ids'][0].cpu().numpy()
    seq_len = len(input_ids)

    print(f"  {n_layers} hidden states, seq_len={seq_len}")

    # Get token type boundaries
    # In OpenVLA: input_ids contains text tokens and image tokens
    # Image tokens are typically in the middle
    layer_profiles = {}

    for layer_idx in range(n_layers):
        hs = fwd.hidden_states[layer_idx][0]  # [seq_len, hidden_dim]
        layer_profiles[layer_idx] = {
            "shape": list(hs.shape),
            "mean_norm": float(hs.float().norm(dim=-1).mean().cpu()),
            "std_norm": float(hs.float().norm(dim=-1).std().cpu()),
            "first_token_norm": float(hs[0].float().norm().cpu()),
            "last_token_norm": float(hs[-1].float().norm().cpu()),
        }
        if layer_idx % 5 == 0:
            print(f"  Layer {layer_idx}: shape={hs.shape}, mean_norm={layer_profiles[layer_idx]['mean_norm']:.2f}")

    results["layer_profiles_clean"] = layer_profiles

    # Part 2: Layer-by-layer corruption distance (ALL layers)
    print("\n=== Part 2: Corruption Distance per Layer ===")
    corruptions = {'fog': 0.5, 'night': 0.5, 'blur': 0.5, 'noise': 0.5}
    layer_distances = {}

    # Get all clean embeddings first
    clean_hidden = {}
    for layer_idx in range(n_layers):
        clean_hidden[layer_idx] = fwd.hidden_states[layer_idx][0, -1, :].float().cpu().numpy()

    for c, sev in corruptions.items():
        print(f"  {c} (sev={sev})...")
        corrupted = apply_corruption(base_img, c, sev)
        inputs_c = processor(prompt, corrupted).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            fwd_c = model(**inputs_c, output_hidden_states=True)

        dists = []
        for layer_idx in range(n_layers):
            emb_c = fwd_c.hidden_states[layer_idx][0, -1, :].float().cpu().numpy()
            d = float(cosine(clean_hidden[layer_idx], emb_c))
            dists.append(d)

        layer_distances[c] = dists
        # Find first layer with d > 0
        first_nonzero = next((i for i, d in enumerate(dists) if d > 0), -1)
        max_layer = int(np.argmax(dists))
        print(f"    First nonzero: layer {first_nonzero}, max: layer {max_layer} (d={max(dists):.6f})")

    results["layer_distances"] = layer_distances

    # Part 3: Token-position variance analysis
    print("\n=== Part 3: Token Position Variance ===")
    # Compare first, middle, last token across 3 clean passes
    token_variance = {}
    clean_runs = []

    for i in range(3):
        inputs_i = processor(prompt, base_img).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            fwd_i = model(**inputs_i, output_hidden_states=True)
        clean_runs.append(fwd_i)

    for layer_idx in [0, 1, 3, 7, 15, 31, n_layers-1]:
        if layer_idx >= n_layers:
            continue
        embeddings = [run.hidden_states[layer_idx][0].float().cpu().numpy() for run in clean_runs]
        # Check variance across runs for each token position
        var_per_position = np.var(embeddings, axis=0).mean(axis=-1)  # [seq_len]
        token_variance[layer_idx] = {
            "max_variance": float(var_per_position.max()),
            "mean_variance": float(var_per_position.mean()),
            "all_zero": bool(var_per_position.max() == 0),
            "n_positions": int(len(var_per_position)),
        }
        print(f"  Layer {layer_idx}: max_var={var_per_position.max():.2e}, all_zero={var_per_position.max() == 0}")

    results["token_variance"] = token_variance

    # Part 4: Corruption effect on different token positions
    print("\n=== Part 4: Per-Token Corruption Effect ===")
    token_corruption = {}

    for c in ['fog', 'night', 'blur', 'noise']:
        corrupted = apply_corruption(base_img, c, 0.5)
        inputs_c = processor(prompt, corrupted).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            fwd_c = model(**inputs_c, output_hidden_states=True)

        layer3_clean = fwd.hidden_states[3][0].float().cpu().numpy()  # [seq_len, dim]
        layer3_corr = fwd_c.hidden_states[3][0].float().cpu().numpy()

        # Cosine distance per token
        per_token_dist = []
        for t in range(min(layer3_clean.shape[0], layer3_corr.shape[0])):
            d = float(cosine(layer3_clean[t], layer3_corr[t]))
            per_token_dist.append(d)

        # Find BOS, first image, last image, first text, last text
        token_corruption[c] = {
            "bos_distance": per_token_dist[0],
            "token_10_distance": per_token_dist[10] if len(per_token_dist) > 10 else 0,
            "mid_distance": per_token_dist[len(per_token_dist)//2],
            "last_distance": per_token_dist[-1],
            "max_distance": float(max(per_token_dist)),
            "max_position": int(np.argmax(per_token_dist)),
            "mean_distance": float(np.mean(per_token_dist)),
            "n_nonzero": sum(1 for d in per_token_dist if d > 0),
            "n_total": len(per_token_dist),
        }
        print(f"  {c}: BOS={per_token_dist[0]:.6f}, max={max(per_token_dist):.6f} @ pos {np.argmax(per_token_dist)}, "
              f"nonzero={sum(1 for d in per_token_dist if d > 0)}/{len(per_token_dist)}")

    results["token_corruption"] = token_corruption

    # Part 5: Activation statistics per layer under corruption
    print("\n=== Part 5: Activation Statistics Under Corruption ===")
    activation_stats = {}

    for c in ['fog', 'blur']:
        corrupted = apply_corruption(base_img, c, 0.5)
        inputs_c = processor(prompt, corrupted).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            fwd_c = model(**inputs_c, output_hidden_states=True)

        stats = []
        for layer_idx in range(0, n_layers, 4):  # Sample every 4th layer
            clean_hs = fwd.hidden_states[layer_idx][0, -1, :].float().cpu().numpy()
            corr_hs = fwd_c.hidden_states[layer_idx][0, -1, :].float().cpu().numpy()

            diff = corr_hs - clean_hs
            stats.append({
                "layer": layer_idx,
                "clean_norm": float(np.linalg.norm(clean_hs)),
                "corr_norm": float(np.linalg.norm(corr_hs)),
                "norm_change_pct": float((np.linalg.norm(corr_hs) - np.linalg.norm(clean_hs)) / np.linalg.norm(clean_hs) * 100),
                "diff_norm": float(np.linalg.norm(diff)),
                "cosine_dist": float(cosine(clean_hs, corr_hs)),
                "n_sign_flips": int(np.sum(np.sign(clean_hs) != np.sign(corr_hs))),
                "pct_sign_flips": float(np.sum(np.sign(clean_hs) != np.sign(corr_hs)) / len(clean_hs) * 100),
            })

        activation_stats[c] = stats
        for s in stats:
            if s['layer'] in [0, 4, 16, 32]:
                print(f"  {c} L{s['layer']}: norm_change={s['norm_change_pct']:.2f}%, "
                      f"cos_dist={s['cosine_dist']:.6f}, sign_flips={s['pct_sign_flips']:.1f}%")

    results["activation_stats"] = activation_stats

    # Part 6: Distance amplification across layers
    print("\n=== Part 6: Distance Amplification ===")
    amplification = {}

    for c in ['fog', 'night', 'blur', 'noise']:
        dists = layer_distances[c]
        if dists[0] > 0:
            amp = [d / dists[0] for d in dists]
        else:
            first_nonzero = next((i for i, d in enumerate(dists) if d > 0), len(dists))
            amp = [0] * first_nonzero + [d / dists[first_nonzero] if first_nonzero < len(dists) else 0 for d in dists[first_nonzero:]]

        amplification[c] = {
            "first_nonzero_layer": next((i for i, d in enumerate(dists) if d > 0), -1),
            "max_layer": int(np.argmax(dists)),
            "max_distance": float(max(dists)),
            "min_nonzero_distance": float(min(d for d in dists if d > 0)) if any(d > 0 for d in dists) else 0,
            "amplification_ratio": float(max(dists) / min(d for d in dists if d > 0)) if any(d > 0 for d in dists) else 0,
        }
        print(f"  {c}: first@L{amplification[c]['first_nonzero_layer']}, "
              f"max@L{amplification[c]['max_layer']} ({amplification[c]['max_distance']:.6f}), "
              f"amplification={amplification[c]['amplification_ratio']:.1f}x")

    results["amplification"] = amplification

    # Save
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(v) for v in obj]
        return obj

    ts = results["timestamp"]
    out_path = f"experiments/arch_probe_{ts}.json"
    with open(out_path, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
