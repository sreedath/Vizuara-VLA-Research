#!/usr/bin/env python3
"""Experiment 321: Attention Mechanism Deep Dive
Analyzes how attention patterns change under corruption:
1. Attention weight distribution shift per layer
2. Self-attention vs cross-attention patterns
3. Image token attention proportion under corruption
4. Attention entropy per head per layer
5. Which heads are most sensitive to corruption
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
        "experiment": "attention_deep",
        "experiment_number": 321,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    corruptions = ['fog', 'night', 'blur', 'noise']

    # Get clean attention patterns
    print("Computing clean attention...")
    inputs = processor(prompt, base_img).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd_clean = model(**inputs, output_attentions=True, output_hidden_states=True)

    n_layers = len(fwd_clean.attentions)
    seq_len = fwd_clean.attentions[0].shape[-1]
    n_heads = fwd_clean.attentions[0].shape[1]

    print(f"  {n_layers} layers, {n_heads} heads, seq_len={seq_len}")

    # Part 1: Attention entropy per layer (clean)
    print("\n=== Part 1: Clean Attention Entropy ===")
    clean_entropy = {}

    for layer_idx in range(0, n_layers, 4):  # Every 4th layer
        attn = fwd_clean.attentions[layer_idx][0]  # [n_heads, seq_len, seq_len]
        attn_np = attn.float().cpu().numpy()

        # Entropy of last token's attention distribution per head
        head_entropies = []
        for h in range(n_heads):
            attn_dist = attn_np[h, -1, :]  # Last token attending to all others
            attn_dist = attn_dist + 1e-30  # avoid log(0)
            entropy = -np.sum(attn_dist * np.log2(attn_dist + 1e-30))
            head_entropies.append(float(entropy))

        clean_entropy[layer_idx] = {
            "mean_entropy": float(np.mean(head_entropies)),
            "std_entropy": float(np.std(head_entropies)),
            "min_entropy": float(min(head_entropies)),
            "max_entropy": float(max(head_entropies)),
        }
        print(f"  L{layer_idx}: mean_H={np.mean(head_entropies):.3f}, "
              f"std={np.std(head_entropies):.3f}")

    results["clean_entropy"] = clean_entropy

    # Part 2: Attention change under corruption
    print("\n=== Part 2: Attention Change Under Corruption ===")
    attn_changes = {}

    for c in corruptions:
        print(f"  {c}...")
        corrupted = apply_corruption(base_img, c, 0.5)
        inp = processor(prompt, corrupted).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            fwd = model(**inp, output_attentions=True, output_hidden_states=True)

        layer_changes = {}
        for layer_idx in range(0, n_layers, 4):
            clean_attn = fwd_clean.attentions[layer_idx][0].float().cpu().numpy()
            corr_attn = fwd.attentions[layer_idx][0].float().cpu().numpy()

            # Attention KL divergence per head (last token)
            head_kls = []
            head_entropy_changes = []
            for h in range(n_heads):
                p = clean_attn[h, -1, :] + 1e-30
                q = corr_attn[h, -1, :] + 1e-30
                kl = float(np.sum(p * np.log(p / q)))
                head_kls.append(kl)

                h_clean = -np.sum(p * np.log2(p))
                h_corr = -np.sum(q * np.log2(q))
                head_entropy_changes.append(float(h_corr - h_clean))

            # Overall attention pattern change (Frobenius norm)
            diff_norm = float(np.linalg.norm(corr_attn[:, -1, :] - clean_attn[:, -1, :]))

            layer_changes[layer_idx] = {
                "mean_kl": float(np.mean(head_kls)),
                "max_kl": float(max(head_kls)),
                "mean_entropy_change": float(np.mean(head_entropy_changes)),
                "diff_norm": diff_norm,
                "most_affected_head": int(np.argmax(head_kls)),
            }

        attn_changes[c] = layer_changes

    results["attention_changes"] = attn_changes

    # Part 3: Image token attention proportion
    print("\n=== Part 3: Image Token Attention ===")
    # Estimate image token range (typically positions ~18 to ~274-18 in OpenVLA)
    # For simplicity, use middle 70% of tokens as image tokens
    img_start = max(1, int(seq_len * 0.05))
    img_end = int(seq_len * 0.95)

    img_attention = {}
    for c in ['clean'] + corruptions:
        if c == 'clean':
            fwd = fwd_clean
        else:
            corrupted = apply_corruption(base_img, c, 0.5)
            inp = processor(prompt, corrupted).to(model.device, dtype=torch.bfloat16)
            with torch.no_grad():
                fwd = model(**inp, output_attentions=True)

        layer_img_attn = {}
        for layer_idx in [0, 3, 15, 31]:
            if layer_idx >= n_layers:
                continue
            attn = fwd.attentions[layer_idx][0].float().cpu().numpy()
            # Last token's attention to image tokens vs all tokens
            img_attn = np.mean([attn[h, -1, img_start:img_end].sum() for h in range(n_heads)])
            total_attn = np.mean([attn[h, -1, :].sum() for h in range(n_heads)])
            layer_img_attn[layer_idx] = {
                "img_attention_pct": float(img_attn / total_attn * 100),
                "img_attention_abs": float(img_attn),
            }

        img_attention[c] = layer_img_attn

    results["image_attention"] = img_attention

    for c in ['clean'] + corruptions:
        l3_pct = img_attention[c].get(3, {}).get('img_attention_pct', 0)
        l31_pct = img_attention[c].get(31, {}).get('img_attention_pct', 0)
        print(f"  {c}: L3={l3_pct:.1f}%, L31={l31_pct:.1f}%")

    # Part 4: Per-severity attention entropy profile
    print("\n=== Part 4: Severity → Attention Entropy ===")
    severity_entropy = {}

    for c in ['fog', 'blur']:
        curve = []
        for sev in [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]:
            if sev == 0:
                fwd = fwd_clean
            else:
                corrupted = apply_corruption(base_img, c, sev)
                inp = processor(prompt, corrupted).to(model.device, dtype=torch.bfloat16)
                with torch.no_grad():
                    fwd = model(**inp, output_attentions=True)

            # L3 and L31 entropy
            for l_idx in [3, 31]:
                if l_idx < n_layers:
                    attn = fwd.attentions[l_idx][0].float().cpu().numpy()
                    entropies = []
                    for h in range(n_heads):
                        p = attn[h, -1, :] + 1e-30
                        entropies.append(float(-np.sum(p * np.log2(p))))

                    curve.append({
                        "severity": float(sev),
                        "layer": l_idx,
                        "mean_entropy": float(np.mean(entropies)),
                    })

        severity_entropy[c] = curve

    results["severity_entropy"] = severity_entropy

    # Save
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(v) for v in obj]
        return obj

    ts = results["timestamp"]
    out_path = f"experiments/attn_deep_{ts}.json"
    with open(out_path, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
