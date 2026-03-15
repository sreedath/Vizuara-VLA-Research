#!/usr/bin/env python3
"""Experiment 397: Attention Pattern Shifts Under Corruption

Examines how attention patterns change under corruption and whether attention
provides complementary detection signal to hidden states.

Tests:
1. Attention entropy under clean vs corrupt
2. Per-layer attention pattern divergence (cosine distance)
3. Per-head vulnerability analysis
4. Attention vs hidden state detection comparison
5. Attention to image vs text token ratio changes
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
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    return 1.0 - np.dot(a, b) / (na * nb)

def main():
    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    img = Image.fromarray(np.random.RandomState(42).randint(0, 255, (224, 224, 3), dtype=np.uint8))
    corruptions = ['fog', 'night', 'noise', 'blur']

    def get_attention_and_hidden(image):
        inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            out = model(**inputs, output_attentions=True, output_hidden_states=True)
        attentions = []
        for layer_attn in out.attentions:
            attentions.append(layer_attn[0].float().cpu().numpy())
        hidden = out.hidden_states[3][0, -1, :].float().cpu().numpy()
        return attentions, hidden

    print("Clean baseline...")
    clean_attns, clean_hidden = get_attention_and_hidden(img)
    n_layers = len(clean_attns)
    n_heads = clean_attns[0].shape[0]
    print(f"  {n_layers} layers, {n_heads} heads")

    results = {"n_layers": n_layers, "n_heads": n_heads, "corrupted": {}}

    # Per-corruption analysis
    for c in corruptions:
        print(f"\n=== {c} ===")
        corrupted = apply_corruption(img, c)
        corrupt_attns, corrupt_hidden = get_attention_and_hidden(corrupted)

        hidden_dist = float(cosine_dist(corrupt_hidden, clean_hidden))

        layer_data = []
        for li in range(n_layers):
            # Last-token attention pattern
            clean_last = clean_attns[li][:, -1, :]  # (heads, seq)
            corrupt_last = corrupt_attns[li][:, -1, :]

            # Attention entropy
            eps = 1e-10
            clean_ent = -np.sum(clean_last * np.log(clean_last + eps), axis=1)
            corrupt_ent = -np.sum(corrupt_last * np.log(corrupt_last + eps), axis=1)

            # Cosine distance of flattened attention
            attn_cos = cosine_dist(clean_last.flatten(), corrupt_last.flatten())

            # Per-head L1
            per_head_l1 = np.mean(np.abs(clean_last - corrupt_last), axis=1)

            layer_data.append({
                "layer": li,
                "attn_cosine_dist": float(attn_cos),
                "mean_entropy_change": float(np.mean(corrupt_ent - clean_ent)),
                "max_head_l1": float(np.max(per_head_l1)),
                "min_head_l1": float(np.min(per_head_l1)),
                "mean_head_l1": float(np.mean(per_head_l1))
            })

        # Find best detection layer by attention
        best_attn_layer = max(layer_data, key=lambda x: x["attn_cosine_dist"])
        worst_attn_layer = min(layer_data, key=lambda x: x["attn_cosine_dist"])

        results["corrupted"][c] = {
            "hidden_dist": hidden_dist,
            "best_attn_layer": best_attn_layer["layer"],
            "best_attn_cosine": best_attn_layer["attn_cosine_dist"],
            "worst_attn_layer": worst_attn_layer["layer"],
            "worst_attn_cosine": worst_attn_layer["attn_cosine_dist"],
            "layer_data": layer_data,
            "hidden_vs_best_attn": hidden_dist / best_attn_layer["attn_cosine_dist"]
                                    if best_attn_layer["attn_cosine_dist"] > 0 else 0
        }
        print(f"  hidden_dist={hidden_dist:.6f}")
        print(f"  best_attn: L{best_attn_layer['layer']} cos={best_attn_layer['attn_cosine_dist']:.6f}")
        print(f"  worst_attn: L{worst_attn_layer['layer']} cos={worst_attn_layer['attn_cosine_dist']:.6f}")
        print(f"  hidden/attn ratio: {hidden_dist / best_attn_layer['attn_cosine_dist']:.2f}")

    # Save
    out_path = "/workspace/Vizuara-VLA-Research/experiments/attention_corruption_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
