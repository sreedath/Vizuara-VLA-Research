#!/usr/bin/env python3
"""Experiment 363: Attention Pattern Analysis Under Corruption

How do corruptions affect the model's internal attention patterns?
1. Attention entropy change per corruption type
2. Per-head sensitivity ranking (Jensen-Shannon divergence)
3. Attention concentration shift (top-k mass)
4. Severity-dependent attention disruption
5. Early vs late layer sensitivity
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

def get_attention_maps(model, processor, image, prompt):
    """Get attention maps from all layers."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_attentions=True)
    attentions = []
    for layer_attn in fwd.attentions:
        attentions.append(layer_attn[0].float().cpu().numpy())
    return attentions, inputs['input_ids'].shape[1]

def attention_entropy(attn_row):
    """Compute entropy of an attention distribution."""
    p = attn_row + 1e-12
    p = p / p.sum()
    return -float(np.sum(p * np.log2(p)))

def top_k_mass(attn_row, k=5):
    """Fraction of attention in top-k positions."""
    sorted_vals = np.sort(attn_row)[::-1]
    return float(sorted_vals[:k].sum())

def js_divergence(p, q):
    """Jensen-Shannon divergence between two distributions."""
    p = np.asarray(p, dtype=np.float64) + 1e-12
    q = np.asarray(q, dtype=np.float64) + 1e-12
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * np.log2(p / m)))
    kl_qm = float(np.sum(q * np.log2(q / m)))
    return 0.5 * (kl_pm + kl_qm)

def main():
    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    results = {}
    ctypes = ['fog', 'night', 'noise', 'blur']

    # Generate test images
    print("Generating images...")
    seeds = list(range(0, 1000, 100))[:10]
    images = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        px = rng.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        images[seed] = Image.fromarray(px)

    # Get clean attention maps for first image to determine structure
    print("Probing attention structure...")
    test_attn, seq_len = get_attention_maps(model, processor, images[seeds[0]], prompt)
    n_layers = len(test_attn)
    n_heads = test_attn[0].shape[0]
    print(f"  {n_layers} layers, {n_heads} heads, seq_len={seq_len}")

    # ========== 1. Per-Layer Attention Entropy Change ==========
    print("\n=== Layer-wise Entropy Change ===")

    analysis_seeds = seeds[:5]

    layer_entropy = {}
    for ct in ctypes:
        per_layer_delta = {l: [] for l in range(n_layers)}

        for seed in analysis_seeds:
            clean_attn, _ = get_attention_maps(model, processor, images[seed], prompt)
            corrupt_img = apply_corruption(images[seed], ct, 0.5)
            corrupt_attn, _ = get_attention_maps(model, processor, corrupt_img, prompt)

            for l in range(n_layers):
                clean_ent = np.mean([attention_entropy(clean_attn[l][h, -1, :]) for h in range(n_heads)])
                corrupt_ent = np.mean([attention_entropy(corrupt_attn[l][h, -1, :]) for h in range(n_heads)])
                per_layer_delta[l].append(corrupt_ent - clean_ent)

        layer_entropy[ct] = {}
        for l in range(n_layers):
            deltas = per_layer_delta[l]
            layer_entropy[ct][str(l)] = {
                'mean_delta': float(np.mean(deltas)),
                'std_delta': float(np.std(deltas)),
                'max_abs_delta': float(max(abs(d) for d in deltas)),
            }

        most_affected = max(range(n_layers), key=lambda l: abs(float(np.mean(per_layer_delta[l]))))
        print(f"  {ct}: most affected layer={most_affected}, "
              f"delta={np.mean(per_layer_delta[most_affected]):.4f}")

    results['layer_entropy'] = layer_entropy

    # ========== 2. Per-Head Sensitivity Ranking ==========
    print("\n=== Per-Head Sensitivity ===")

    head_sensitivity = {}
    for ct in ctypes:
        head_jsd = {}

        for seed in analysis_seeds:
            clean_attn, _ = get_attention_maps(model, processor, images[seed], prompt)
            corrupt_img = apply_corruption(images[seed], ct, 0.5)
            corrupt_attn, _ = get_attention_maps(model, processor, corrupt_img, prompt)

            for l in range(n_layers):
                for h in range(n_heads):
                    key = f"{l}_{h}"
                    if key not in head_jsd:
                        head_jsd[key] = []
                    jsd = js_divergence(clean_attn[l][h, -1, :], corrupt_attn[l][h, -1, :])
                    head_jsd[key].append(jsd)

        head_means = {k: float(np.mean(v)) for k, v in head_jsd.items()}
        sorted_heads = sorted(head_means.items(), key=lambda x: -x[1])

        top5 = sorted_heads[:5]
        bottom5 = sorted_heads[-5:]

        head_sensitivity[ct] = {
            'top5_sensitive': [{'head': h, 'mean_jsd': v} for h, v in top5],
            'bottom5_sensitive': [{'head': h, 'mean_jsd': v} for h, v in bottom5],
            'mean_jsd_all': float(np.mean(list(head_means.values()))),
            'std_jsd_all': float(np.std(list(head_means.values()))),
            'max_jsd': float(max(head_means.values())),
        }
        print(f"  {ct}: mean_JSD={np.mean(list(head_means.values())):.6f}, "
              f"top_head={top5[0][0]} (JSD={top5[0][1]:.6f})")

    results['head_sensitivity'] = head_sensitivity

    # ========== 3. Attention Concentration (Top-K Mass) ==========
    print("\n=== Attention Concentration ===")

    concentration = {}
    for ct in ctypes:
        per_layer_conc = {l: {'clean': [], 'corrupt': []} for l in range(n_layers)}

        for seed in analysis_seeds:
            clean_attn, _ = get_attention_maps(model, processor, images[seed], prompt)
            corrupt_img = apply_corruption(images[seed], ct, 0.5)
            corrupt_attn, _ = get_attention_maps(model, processor, corrupt_img, prompt)

            for l in range(n_layers):
                clean_mass = np.mean([top_k_mass(clean_attn[l][h, -1, :], k=5) for h in range(n_heads)])
                corrupt_mass = np.mean([top_k_mass(corrupt_attn[l][h, -1, :], k=5) for h in range(n_heads)])
                per_layer_conc[l]['clean'].append(clean_mass)
                per_layer_conc[l]['corrupt'].append(corrupt_mass)

        concentration[ct] = {}
        for l in range(n_layers):
            concentration[ct][str(l)] = {
                'clean_top5_mass': float(np.mean(per_layer_conc[l]['clean'])),
                'corrupt_top5_mass': float(np.mean(per_layer_conc[l]['corrupt'])),
                'delta': float(np.mean(per_layer_conc[l]['corrupt']) - np.mean(per_layer_conc[l]['clean'])),
            }

        all_clean = [np.mean(per_layer_conc[l]['clean']) for l in range(n_layers)]
        all_corrupt = [np.mean(per_layer_conc[l]['corrupt']) for l in range(n_layers)]
        print(f"  {ct}: mean clean top5={np.mean(all_clean):.4f}, "
              f"corrupt top5={np.mean(all_corrupt):.4f}, "
              f"delta={np.mean(all_corrupt) - np.mean(all_clean):.4f}")

    results['concentration'] = concentration

    # ========== 4. Severity-Dependent Attention Disruption ==========
    print("\n=== Severity vs Attention Disruption ===")

    sev_disruption = {}
    test_seed = seeds[0]
    clean_attn, _ = get_attention_maps(model, processor, images[test_seed], prompt)

    for ct in ctypes:
        per_sev = {}
        for sev in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
            corrupt_img = apply_corruption(images[test_seed], ct, sev)
            corrupt_attn, _ = get_attention_maps(model, processor, corrupt_img, prompt)

            all_jsd = []
            for l in range(n_layers):
                for h in range(n_heads):
                    jsd = js_divergence(clean_attn[l][h, -1, :], corrupt_attn[l][h, -1, :])
                    all_jsd.append(jsd)

            per_sev[str(sev)] = {
                'mean_jsd': float(np.mean(all_jsd)),
                'max_jsd': float(max(all_jsd)),
                'fraction_affected': float(np.mean([1 if j > 1e-6 else 0 for j in all_jsd])),
            }

        sev_disruption[ct] = per_sev
        jsds = [per_sev[str(s)]['mean_jsd'] for s in [0.05, 0.1, 0.5, 1.0]]
        print(f"  {ct}: JSD@[0.05,0.1,0.5,1.0] = [{', '.join(f'{j:.6f}' for j in jsds)}]")

    results['severity_disruption'] = sev_disruption

    # ========== 5. Early vs Late Layer Sensitivity ==========
    print("\n=== Early vs Late Layer Sensitivity ===")

    early_late = {}
    mid = n_layers // 2

    for ct in ctypes:
        early_jsd = []
        late_jsd = []

        for seed in analysis_seeds:
            clean_attn, _ = get_attention_maps(model, processor, images[seed], prompt)
            corrupt_img = apply_corruption(images[seed], ct, 0.5)
            corrupt_attn, _ = get_attention_maps(model, processor, corrupt_img, prompt)

            for l in range(n_layers):
                for h in range(n_heads):
                    jsd = js_divergence(clean_attn[l][h, -1, :], corrupt_attn[l][h, -1, :])
                    if l < mid:
                        early_jsd.append(jsd)
                    else:
                        late_jsd.append(jsd)

        early_late[ct] = {
            'early_mean_jsd': float(np.mean(early_jsd)),
            'late_mean_jsd': float(np.mean(late_jsd)),
            'ratio_late_to_early': float(np.mean(late_jsd) / (np.mean(early_jsd) + 1e-12)),
            'early_std': float(np.std(early_jsd)),
            'late_std': float(np.std(late_jsd)),
        }
        print(f"  {ct}: early={np.mean(early_jsd):.6f}, late={np.mean(late_jsd):.6f}, "
              f"ratio={np.mean(late_jsd)/(np.mean(early_jsd)+1e-12):.2f}x")

    results['early_vs_late'] = early_late

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/attention_analysis_{ts}.json"
    def convert(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)): return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return obj
    def recursive_convert(d):
        if isinstance(d, dict): return {k: recursive_convert(v) for k, v in d.items()}
        if isinstance(d, list): return [recursive_convert(x) for x in d]
        return convert(d)
    results = recursive_convert(results)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
