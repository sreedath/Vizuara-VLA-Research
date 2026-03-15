#!/usr/bin/env python3
"""Experiment 297: Attention Head Specialization Analysis
Examines which attention heads in the transformer respond most to corruptions:
1. Per-head attention entropy change under corruption
2. Head-level OOD detection (which heads alone achieve AUROC=1.0)
3. Head agreement/disagreement on corruption detection
4. Attention pattern similarity across corruption types per head
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

def compute_auroc(id_scores, ood_scores):
    id_s = np.asarray(id_scores)
    ood_s = np.asarray(ood_scores)
    n_id, n_ood = len(id_s), len(ood_s)
    if n_id == 0 or n_ood == 0: return 0.5
    count = sum(float(np.sum(o > id_s) + 0.5 * np.sum(o == id_s)) for o in ood_s)
    return count / (n_id * n_ood)

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
        "experiment": "attention_specialization",
        "experiment_number": 297,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    corruptions = ['fog', 'night', 'blur', 'noise']

    # Get attention outputs for clean image
    print("Getting clean attention outputs...")
    inputs = processor(prompt, base_img).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True, output_attentions=True)

    # Check attention shape
    n_layers = len(fwd.attentions)
    n_heads = fwd.attentions[0].shape[1]
    seq_len = fwd.attentions[0].shape[2]
    print(f"  Model has {n_layers} layers, {n_heads} heads, seq_len={seq_len}")

    results["model_info"] = {
        "n_layers": n_layers,
        "n_heads": n_heads,
        "seq_len": seq_len
    }

    # Part 1: Per-head attention entropy for clean vs corrupted
    print("\n=== Part 1: Per-Head Attention Entropy Change ===")
    target_layers = [0, 3, 7, 15, 31]  # Sample across depth
    entropy_changes = {}

    # Clean attention entropy
    clean_entropies = {}
    for layer_idx in target_layers:
        attn = fwd.attentions[layer_idx][0].float().cpu().numpy()  # (heads, seq, seq)
        layer_entropies = []
        for h in range(n_heads):
            # Entropy of last token's attention distribution
            attn_dist = attn[h, -1, :]
            attn_dist = attn_dist / (attn_dist.sum() + 1e-30)
            ent = -np.sum(attn_dist * np.log(attn_dist + 1e-30))
            layer_entropies.append(float(ent))
        clean_entropies[layer_idx] = layer_entropies

    for c in corruptions:
        print(f"  Processing {c}...")
        corrupted = apply_corruption(base_img, c, 1.0)
        inputs = processor(prompt, corrupted).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            fwd_c = model(**inputs, output_hidden_states=True, output_attentions=True)

        entropy_changes[c] = {}
        for layer_idx in target_layers:
            attn = fwd_c.attentions[layer_idx][0].float().cpu().numpy()
            layer_changes = []
            for h in range(n_heads):
                attn_dist = attn[h, -1, :]
                attn_dist = attn_dist / (attn_dist.sum() + 1e-30)
                ent = -np.sum(attn_dist * np.log(attn_dist + 1e-30))
                change = ent - clean_entropies[layer_idx][h]
                layer_changes.append(float(change))
            entropy_changes[c][f"L{layer_idx}"] = layer_changes

        # Find most affected heads
        all_changes = []
        for layer_idx in target_layers:
            for h, ch in enumerate(entropy_changes[c][f"L{layer_idx}"]):
                all_changes.append((abs(ch), layer_idx, h, ch))
        all_changes.sort(reverse=True)
        top5 = all_changes[:5]
        print(f"    Top 5 most affected heads:")
        for abs_ch, li, hi, ch in top5:
            print(f"      L{li}H{hi}: delta_entropy={ch:+.4f}")

    results["entropy_changes"] = entropy_changes
    results["clean_entropies"] = {f"L{k}": v for k, v in clean_entropies.items()}

    # Part 2: Per-head hidden state OOD detection
    print("\n=== Part 2: Per-Head Hidden State Detection ===")
    # Extract per-head outputs from attention layers
    # We'll use hidden states split by head dimension
    # For LLaMA-style: hidden_size=4096, n_heads=32 -> head_dim=128

    head_dim = 4096 // n_heads
    head_aurocs = {}

    # Get clean hidden state at target layers
    clean_hiddens = {}
    for layer_idx in target_layers:
        h = fwd.hidden_states[layer_idx][0, -1, :].float().cpu().numpy()
        clean_hiddens[layer_idx] = h.reshape(n_heads, head_dim)

    for c in corruptions:
        print(f"  {c}...")
        head_aurocs[c] = {}

        # Get corrupted at multiple severities for AUROC
        id_dists_per_head = {li: {h: [] for h in range(n_heads)} for li in target_layers}
        ood_dists_per_head = {li: {h: [] for h in range(n_heads)} for li in target_layers}

        for sev in [0.0, 0.0, 0.0]:  # 3 clean samples (identical)
            inputs = processor(prompt, base_img).to(model.device, dtype=torch.bfloat16)
            with torch.no_grad():
                fwd_clean = model(**inputs, output_hidden_states=True)
            for layer_idx in target_layers:
                h = fwd_clean.hidden_states[layer_idx][0, -1, :].float().cpu().numpy()
                h_heads = h.reshape(n_heads, head_dim)
                for hi in range(n_heads):
                    d = float(cosine(clean_hiddens[layer_idx][hi], h_heads[hi]))
                    id_dists_per_head[layer_idx][hi].append(d)

        for sev in [0.3, 0.5, 1.0]:
            corrupted = apply_corruption(base_img, c, sev)
            inputs = processor(prompt, corrupted).to(model.device, dtype=torch.bfloat16)
            with torch.no_grad():
                fwd_c = model(**inputs, output_hidden_states=True)
            for layer_idx in target_layers:
                h = fwd_c.hidden_states[layer_idx][0, -1, :].float().cpu().numpy()
                h_heads = h.reshape(n_heads, head_dim)
                for hi in range(n_heads):
                    d = float(cosine(clean_hiddens[layer_idx][hi], h_heads[hi]))
                    ood_dists_per_head[layer_idx][hi].append(d)

        for layer_idx in target_layers:
            layer_aurocs = []
            for hi in range(n_heads):
                auroc = compute_auroc(id_dists_per_head[layer_idx][hi],
                                     ood_dists_per_head[layer_idx][hi])
                layer_aurocs.append(auroc)
            head_aurocs[c][f"L{layer_idx}"] = layer_aurocs
            perfect_count = sum(1 for a in layer_aurocs if a >= 1.0)
            print(f"    L{layer_idx}: {perfect_count}/{n_heads} heads AUROC=1.0, "
                  f"min={min(layer_aurocs):.3f}, mean={np.mean(layer_aurocs):.3f}")

    results["head_aurocs"] = head_aurocs

    # Part 3: Head agreement analysis
    print("\n=== Part 3: Head Agreement ===")
    agreement = {}
    for c in corruptions:
        for layer_key in [f"L{li}" for li in target_layers]:
            aurocs = head_aurocs[c][layer_key]
            n_perfect = sum(1 for a in aurocs if a >= 1.0)
            n_high = sum(1 for a in aurocs if a >= 0.9)
            n_low = sum(1 for a in aurocs if a < 0.7)
            agreement[f"{c}_{layer_key}"] = {
                "n_perfect": n_perfect,
                "n_high_90": n_high,
                "n_low_70": n_low,
                "unanimity": n_perfect == n_heads
            }
    results["head_agreement"] = agreement

    # Part 4: Cross-corruption head similarity
    print("\n=== Part 4: Cross-Corruption Head Patterns ===")
    # For each head, compare entropy change vectors across corruptions
    cross_sim = {}
    for layer_idx in target_layers:
        layer_key = f"L{layer_idx}"
        vecs = {}
        for c in corruptions:
            vecs[c] = np.array(entropy_changes[c][layer_key])

        for i, c1 in enumerate(corruptions):
            for c2 in corruptions[i+1:]:
                n1, n2 = np.linalg.norm(vecs[c1]), np.linalg.norm(vecs[c2])
                if n1 > 0 and n2 > 0:
                    sim = float(np.dot(vecs[c1], vecs[c2]) / (n1 * n2))
                else:
                    sim = 0.0
                cross_sim[f"{layer_key}_{c1}_vs_{c2}"] = sim

    results["cross_corruption_head_similarity"] = cross_sim

    # Summary statistics
    print("\n=== Summary ===")
    for c in corruptions:
        for layer_idx in target_layers:
            layer_key = f"L{layer_idx}"
            aurocs = head_aurocs[c][layer_key]
            print(f"  {c} {layer_key}: AUROC mean={np.mean(aurocs):.3f}, "
                  f"perfect={sum(1 for a in aurocs if a>=1.0)}/{n_heads}")

    # Save
    ts = results["timestamp"]
    out_path = f"experiments/attention_spec_{ts}.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
