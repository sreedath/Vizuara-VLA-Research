#!/usr/bin/env python3
"""Experiment 421: Attention Pattern Analysis Under Corruption

Studies how the model's attention patterns change when processing corrupted
vs clean images. Which attention heads are most affected? Do corrupted inputs
cause attention to shift between image and text regions?

Tests:
1. Per-head attention entropy (clean vs corrupted)
2. Image-to-text vs text-to-image attention flow
3. Attention head sensitivity ranking
4. Cross-token attention concentration under corruption
5. Attention-based OOD detection (using attention divergence)
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

def extract_attentions(model, processor, image, prompt, layers=[0, 3, 15, 31]):
    """Extract attention weights for specified layers."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_attentions=True)
    # fwd.attentions is tuple of (batch, num_heads, seq_len, seq_len)
    result = {}
    for l in layers:
        if l < len(fwd.attentions):
            attn = fwd.attentions[l][0].float().cpu().numpy()  # (num_heads, seq_len, seq_len)
            result[l] = attn
    return result

def attention_entropy(attn_matrix):
    """Compute entropy of attention distribution for each head and position."""
    eps = 1e-10
    log_attn = np.log(attn_matrix + eps)
    entropy = -np.sum(attn_matrix * log_attn, axis=-1)  # (num_heads, seq_len)
    return entropy

def cosine_dist(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    return 1.0 - np.dot(a, b) / (na * nb)

def compute_auroc(id_scores, ood_scores):
    id_s = np.asarray(id_scores, dtype=np.float64)
    ood_s = np.asarray(ood_scores, dtype=np.float64)
    n_id, n_ood = len(id_s), len(ood_s)
    if n_id == 0 or n_ood == 0:
        return 0.5
    count = sum(float(np.sum(o > id_s) + 0.5 * np.sum(o == id_s)) for o in ood_s)
    return count / (n_id * n_ood)

def main():
    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    corruptions = ['fog', 'night', 'noise', 'blur']
    analysis_layers = [0, 3, 8, 15, 24, 31]

    seeds = [42, 123, 456]
    scenes = [Image.fromarray(np.random.RandomState(s).randint(0, 255, (224, 224, 3), dtype=np.uint8)) for s in seeds]

    print("Extracting clean attention patterns...")
    clean_attns = [extract_attentions(model, processor, s, prompt, analysis_layers) for s in scenes]
    num_heads = clean_attns[0][analysis_layers[0]].shape[0]
    seq_len = clean_attns[0][analysis_layers[0]].shape[1]
    print(f"  {num_heads} heads, {seq_len} positions, {len(analysis_layers)} layers")

    corrupt_attns = {}
    for c in corruptions:
        corrupt_attns[c] = [extract_attentions(model, processor, apply_corruption(s, c), prompt, analysis_layers) for s in scenes]
        print(f"  {c} extracted")

    results = {"num_heads": num_heads, "seq_len": seq_len, "layers_analyzed": analysis_layers}

    # === Test 1: Per-head attention entropy ===
    print("\n=== Per-Head Attention Entropy ===")
    entropy_results = {}
    for l in analysis_layers:
        clean_entropies = []
        for s in range(len(scenes)):
            ent = attention_entropy(clean_attns[s][l])
            clean_entropies.append(np.mean(ent, axis=1))
        clean_mean_ent = np.mean(clean_entropies, axis=0)

        layer_entry = {"clean_mean": float(np.mean(clean_mean_ent))}
        for c in corruptions:
            corr_entropies = []
            for s in range(len(scenes)):
                ent = attention_entropy(corrupt_attns[c][s][l])
                corr_entropies.append(np.mean(ent, axis=1))
            corr_mean_ent = np.mean(corr_entropies, axis=0)
            layer_entry[f"{c}_mean"] = float(np.mean(corr_mean_ent))
            layer_entry[f"{c}_shift"] = float(np.mean(corr_mean_ent - clean_mean_ent))
        entropy_results[str(l)] = layer_entry
        print(f"  Layer {l}: clean={layer_entry['clean_mean']:.4f}, fog_shift={layer_entry['fog_shift']:.4f}")
    results["attention_entropy"] = entropy_results

    # === Test 2: Image vs text attention flow ===
    print("\n=== Image-Text Attention Flow ===")
    img_range = slice(1, 257)
    text_range = slice(257, seq_len)
    flow_results = {}
    for l in analysis_layers:
        clean_img2text = []
        clean_text2img = []
        for s in range(len(scenes)):
            attn = clean_attns[s][l]
            img2text = np.mean(attn[:, img_range, text_range])
            text2img = np.mean(attn[:, text_range, img_range])
            clean_img2text.append(float(img2text))
            clean_text2img.append(float(text2img))

        layer_flow = {
            "clean_img2text": float(np.mean(clean_img2text)),
            "clean_text2img": float(np.mean(clean_text2img)),
        }
        for c in corruptions:
            corr_img2text = []
            corr_text2img = []
            for s in range(len(scenes)):
                attn = corrupt_attns[c][s][l]
                corr_img2text.append(float(np.mean(attn[:, img_range, text_range])))
                corr_text2img.append(float(np.mean(attn[:, text_range, img_range])))
            layer_flow[f"{c}_img2text"] = float(np.mean(corr_img2text))
            layer_flow[f"{c}_text2img"] = float(np.mean(corr_text2img))
        flow_results[str(l)] = layer_flow
        print(f"  Layer {l}: clean i2t={layer_flow['clean_img2text']:.6f}, t2i={layer_flow['clean_text2img']:.6f}")
    results["attention_flow"] = flow_results

    # === Test 3: Per-head sensitivity ranking ===
    print("\n=== Head Sensitivity Ranking ===")
    head_sensitivity = {}
    for l in analysis_layers:
        head_shifts = np.zeros(num_heads)
        for c in corruptions:
            for s in range(len(scenes)):
                clean_ent = attention_entropy(clean_attns[s][l])
                corr_ent = attention_entropy(corrupt_attns[c][s][l])
                shift = np.mean(np.abs(corr_ent - clean_ent), axis=1)
                head_shifts += shift
        head_shifts /= (len(corruptions) * len(scenes))
        ranked = np.argsort(head_shifts)[::-1]
        top5 = [(int(ranked[i]), float(head_shifts[ranked[i]])) for i in range(min(5, num_heads))]
        bot5 = [(int(ranked[-(i+1)]), float(head_shifts[ranked[-(i+1)]])) for i in range(min(5, num_heads))]
        head_sensitivity[str(l)] = {
            "top_5": top5,
            "bottom_5": bot5,
            "mean": float(np.mean(head_shifts)),
            "std": float(np.std(head_shifts)),
        }
        print(f"  Layer {l}: most sensitive head={top5[0][0]} (shift={top5[0][1]:.4f})")
    results["head_sensitivity"] = head_sensitivity

    # === Test 4: Attention concentration ===
    print("\n=== Attention Concentration ===")
    concentration = {}
    for l in analysis_layers:
        clean_conc = []
        for s in range(len(scenes)):
            max_attn = np.max(clean_attns[s][l], axis=-1)
            clean_conc.append(float(np.mean(max_attn)))
        layer_conc = {"clean_mean_max": float(np.mean(clean_conc))}
        for c in corruptions:
            corr_conc = []
            for s in range(len(scenes)):
                max_attn = np.max(corrupt_attns[c][s][l], axis=-1)
                corr_conc.append(float(np.mean(max_attn)))
            layer_conc[f"{c}_mean_max"] = float(np.mean(corr_conc))
        concentration[str(l)] = layer_conc
        print(f"  Layer {l}: clean={layer_conc['clean_mean_max']:.4f}, fog={layer_conc.get('fog_mean_max', 0):.4f}")
    results["concentration"] = concentration

    # === Test 5: Attention-based OOD detection ===
    print("\n=== Attention-Based OOD Detection ===")
    attn_detection = {}
    for l in analysis_layers:
        ref_attn = np.mean([clean_attns[s][l] for s in range(len(scenes))], axis=0)

        id_scores = []
        for s in range(len(scenes)):
            eps = 1e-10
            kl = np.sum(ref_attn * np.log((ref_attn + eps) / (clean_attns[s][l] + eps)))
            id_scores.append(float(kl / (num_heads * seq_len)))

        per_corr = {}
        all_ood = []
        for c in corruptions:
            ood_scores = []
            for s in range(len(scenes)):
                kl = np.sum(ref_attn * np.log((ref_attn + eps) / (corrupt_attns[c][s][l] + eps)))
                score = float(kl / (num_heads * seq_len))
                ood_scores.append(score)
                all_ood.append(score)
            per_corr[c] = {
                "auroc": float(compute_auroc(id_scores, ood_scores)),
                "mean_kl": float(np.mean(ood_scores)),
            }

        overall_auroc = float(compute_auroc(id_scores, all_ood))
        attn_detection[str(l)] = {
            "overall_auroc": overall_auroc,
            "id_mean_kl": float(np.mean(id_scores)),
            "per_corruption": per_corr,
        }
        print(f"  Layer {l}: AUROC={overall_auroc:.4f}")
    results["attention_detection"] = attn_detection

    out_path = "/workspace/Vizuara-VLA-Research/experiments/attention_analysis_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
