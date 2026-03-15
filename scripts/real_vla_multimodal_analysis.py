#!/usr/bin/env python3
"""Experiment 329: Multi-Modal Signal Analysis (Real OpenVLA-7B)

Analyzes how corruption signal distributes across modalities:
1. Image token vs text token embedding distances
2. Prompt length effect on detection
3. Token position importance ranking
4. Aggregation strategies: mean, max, min, median of token distances
5. Selective token masking: which tokens carry most signal?
6. Cross-modal signal transfer: image→text and text→image
"""

import json, time, os, sys
import numpy as np
import torch
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor

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

def cosine_dist(a, b):
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return 1.0 - dot / (na * nb)

def get_all_hidden(model, processor, image, prompt, layer=3):
    """Get hidden states for ALL token positions."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    # Shape: [1, seq_len, hidden_dim]
    hidden = out.hidden_states[layer][0].float().cpu().numpy()
    seq_len = hidden.shape[0]
    return hidden, seq_len

def main():
    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    np.random.seed(42)
    pixels = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    base_img = Image.fromarray(pixels)
    prompt = "In: What action should the robot take to pick up the object?\nOut:"

    results = {}

    # ========== 1. Per-token distance map ==========
    print("\n=== Per-Token Distance Map ===")
    clean_hidden, seq_len = get_all_hidden(model, processor, base_img, prompt)
    print(f"  Sequence length: {seq_len}")

    token_map_results = {}
    for ct in ['fog', 'night', 'blur', 'noise']:
        img = apply_corruption(base_img, ct, 1.0)
        corrupt_hidden, _ = get_all_hidden(model, processor, img, prompt)

        # Distance at each token position
        per_token_dist = []
        for i in range(seq_len):
            d = cosine_dist(clean_hidden[i], corrupt_hidden[i])
            per_token_dist.append(float(d))

        # Identify image vs text tokens (image tokens are typically 256 tokens starting after BOS)
        # In OpenVLA: BOS (1) + image tokens (256) + text tokens (rest)
        bos_dist = per_token_dist[0]
        img_dists = per_token_dist[1:257] if seq_len > 257 else per_token_dist[1:seq_len//2]
        text_dists = per_token_dist[257:] if seq_len > 257 else per_token_dist[seq_len//2:]

        token_map_results[ct] = {
            'seq_len': seq_len,
            'bos_dist': bos_dist,
            'img_mean': float(np.mean(img_dists)),
            'img_max': float(np.max(img_dists)),
            'img_min': float(np.min(img_dists)),
            'text_mean': float(np.mean(text_dists)),
            'text_max': float(np.max(text_dists)),
            'text_min': float(np.min(text_dists)),
            'img_text_ratio': float(np.mean(img_dists) / max(np.mean(text_dists), 1e-10)),
            'last_token_dist': per_token_dist[-1],
            # Sample distances at key positions
            'sample_positions': {
                'pos_0_bos': per_token_dist[0],
                'pos_1_img_first': per_token_dist[1] if seq_len > 1 else 0,
                'pos_128_img_mid': per_token_dist[128] if seq_len > 128 else 0,
                'pos_256_img_last': per_token_dist[256] if seq_len > 256 else 0,
                'pos_257_text_first': per_token_dist[257] if seq_len > 257 else 0,
                'pos_last': per_token_dist[-1],
            },
        }
        print(f"  {ct}: img_mean={np.mean(img_dists):.6f}, text_mean={np.mean(text_dists):.6f}, ratio={np.mean(img_dists)/max(np.mean(text_dists), 1e-10):.1f}x")

    results['token_map'] = token_map_results

    # ========== 2. Aggregation strategies ==========
    print("\n=== Aggregation Strategies ===")
    agg_results = {}

    for ct in ['fog', 'night', 'blur', 'noise']:
        img = apply_corruption(base_img, ct, 1.0)
        corrupt_hidden, _ = get_all_hidden(model, processor, img, prompt)

        per_token_dist = [cosine_dist(clean_hidden[i], corrupt_hidden[i]) for i in range(seq_len)]
        img_dists = per_token_dist[1:257] if seq_len > 257 else per_token_dist[1:seq_len//2]
        text_dists = per_token_dist[257:] if seq_len > 257 else per_token_dist[seq_len//2:]
        all_dists = per_token_dist

        # Various aggregation strategies
        strategies = {
            'last_token': per_token_dist[-1],
            'mean_all': float(np.mean(all_dists)),
            'max_all': float(np.max(all_dists)),
            'median_all': float(np.median(all_dists)),
            'mean_img': float(np.mean(img_dists)),
            'max_img': float(np.max(img_dists)),
            'mean_text': float(np.mean(text_dists)),
            'max_text': float(np.max(text_dists)),
            'sum_all': float(np.sum(all_dists)),
        }

        agg_results[ct] = strategies
        print(f"  {ct}: last={strategies['last_token']:.6f}, mean={strategies['mean_all']:.6f}, max={strategies['max_all']:.6f}")

    results['aggregation'] = agg_results

    # ========== 3. Prompt length effect ==========
    print("\n=== Prompt Length Effect ===")
    prompts = {
        'short': "In: Pick up?\nOut:",
        'medium': "In: What action should the robot take to pick up the object?\nOut:",
        'long': "In: Given the current scene, what is the optimal action sequence the robot should execute in order to successfully pick up the red object from the table?\nOut:",
    }

    prompt_results = {}
    for pname, p in prompts.items():
        clean_h, sl = get_all_hidden(model, processor, base_img, p)
        fog_h, _ = get_all_hidden(model, processor, apply_corruption(base_img, 'fog', 1.0), p)

        last_d = cosine_dist(clean_h[-1], fog_h[-1])
        prompt_results[pname] = {
            'seq_len': sl,
            'last_token_dist': float(last_d),
            'detected': bool(last_d > 0),
        }
        print(f"  {pname} (len={sl}): d={last_d:.6f}")

    results['prompt_length'] = prompt_results

    # ========== 4. Layer comparison for different tokens ==========
    print("\n=== Layer × Token Analysis ===")
    layer_token_results = {}
    for layer in [0, 1, 3, 15, 31]:
        clean_h, sl = get_all_hidden(model, processor, base_img, prompt, layer=layer)
        fog_h, _ = get_all_hidden(model, processor, apply_corruption(base_img, 'fog', 1.0), prompt, layer=layer)

        # Distances at key positions
        pos_results = {}
        positions = {'bos': 0, 'img_first': 1, 'img_mid': 128, 'last': -1}
        for pname, pidx in positions.items():
            if pidx == -1:
                pidx = sl - 1
            if pidx < sl:
                d = cosine_dist(clean_h[pidx], fog_h[pidx])
                pos_results[pname] = float(d)

        layer_token_results[f"L{layer}"] = pos_results
        print(f"  L{layer}: " + ", ".join(f"{k}={v:.6f}" for k, v in pos_results.items()))

    results['layer_token'] = layer_token_results

    # ========== 5. Signal concentration ==========
    print("\n=== Signal Concentration ===")
    conc_results = {}

    for ct in ['fog', 'blur']:
        img = apply_corruption(base_img, ct, 1.0)
        corrupt_hidden, _ = get_all_hidden(model, processor, img, prompt)

        per_token_dist = np.array([cosine_dist(clean_hidden[i], corrupt_hidden[i]) for i in range(seq_len)])
        total_signal = np.sum(per_token_dist)

        # What fraction of positions contribute 90% of total signal?
        sorted_dists = np.sort(per_token_dist)[::-1]
        cumsum = np.cumsum(sorted_dists)
        n_for_90 = int(np.searchsorted(cumsum, 0.9 * total_signal) + 1)
        n_for_50 = int(np.searchsorted(cumsum, 0.5 * total_signal) + 1)

        # Gini coefficient
        n = len(per_token_dist)
        sorted_d = np.sort(per_token_dist)
        index = np.arange(1, n + 1)
        gini = float((2 * np.sum(index * sorted_d) - (n + 1) * np.sum(sorted_d)) / (n * np.sum(sorted_d))) if np.sum(sorted_d) > 0 else 0

        conc_results[ct] = {
            'total_signal': float(total_signal),
            'n_for_50pct': n_for_50,
            'n_for_90pct': n_for_90,
            'pct_for_50_signal': float(n_for_50 / seq_len * 100),
            'pct_for_90_signal': float(n_for_90 / seq_len * 100),
            'gini': gini,
            'max_token_signal': float(np.max(per_token_dist)),
            'max_pct_of_total': float(np.max(per_token_dist) / total_signal * 100) if total_signal > 0 else 0,
        }
        print(f"  {ct}: 50% signal in {n_for_50}/{seq_len} tokens ({n_for_50/seq_len*100:.1f}%), Gini={gini:.3f}")

    results['signal_concentration'] = conc_results

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/multimodal_{ts}.json"

    def convert(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    def recursive_convert(d):
        if isinstance(d, dict):
            return {k: recursive_convert(v) for k, v in d.items()}
        if isinstance(d, list):
            return [recursive_convert(x) for x in d]
        return convert(d)

    results = recursive_convert(results)

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
