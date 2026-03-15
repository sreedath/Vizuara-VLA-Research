#!/usr/bin/env python3
"""Experiment 447: Visual Token Analysis

Analyzes how visual tokens (the 260 patch embeddings from the vision encoder)
respond to corruptions. While we extract hidden states from the last token,
the information flows through visual tokens first — understanding their
structure reveals WHY the detector works.

Tests:
1. Visual token statistics (mean, variance per token position)
2. Token-wise corruption sensitivity
3. CLS token vs mean-pooled visual tokens for detection
4. Visual token attention pattern changes
5. Token diversity/redundancy under corruption
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

    seeds = [42, 123, 456, 789, 999, 1111, 2222, 3333]
    scenes = [Image.fromarray(np.random.RandomState(s).randint(0, 255, (224, 224, 3), dtype=np.uint8)) for s in seeds]

    results = {"n_scenes": len(scenes)}

    # === Test 1: Full hidden state per token position (layer 3) ===
    print("\n=== Per-Position Hidden State Analysis ===")

    def get_all_hidden(model, processor, image, prompt, layer=3):
        inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            fwd = model(**inputs, output_hidden_states=True)
        return fwd.hidden_states[layer][0].float().cpu().numpy()  # (seq_len, hidden_dim)

    # Get hidden states for a clean scene
    s0 = scenes[0]
    h_clean = get_all_hidden(model, processor, s0, prompt)
    seq_len = h_clean.shape[0]
    hidden_dim = h_clean.shape[1]
    print(f"  Sequence length: {seq_len}, Hidden dim: {hidden_dim}")
    results["seq_len"] = seq_len
    results["hidden_dim"] = hidden_dim

    # Token norms across positions
    clean_norms = [float(np.linalg.norm(h_clean[i])) for i in range(seq_len)]

    # Compare with corrupted
    position_sensitivity = {}
    for c in ['fog', 'night']:
        h_corr = get_all_hidden(model, processor, apply_corruption(s0, c), prompt)
        per_pos_dist = [float(cosine_dist(h_clean[i], h_corr[i])) for i in range(seq_len)]
        per_pos_norm_change = [float(np.linalg.norm(h_corr[i]) - np.linalg.norm(h_clean[i])) for i in range(seq_len)]

        # Identify visual (0-259) vs text (260+) sensitivity
        vis_dists = per_pos_dist[:260] if seq_len > 260 else per_pos_dist
        txt_dists = per_pos_dist[260:] if seq_len > 260 else []

        position_sensitivity[c] = {
            "mean_visual_dist": float(np.mean(vis_dists)),
            "mean_text_dist": float(np.mean(txt_dists)) if txt_dists else 0,
            "max_visual_dist": float(np.max(vis_dists)),
            "last_token_dist": float(per_pos_dist[-1]),
            "visual_to_text_ratio": float(np.mean(vis_dists) / max(np.mean(txt_dists), 1e-12)) if txt_dists else 0,
            # Sample positions
            "sample_dists": {str(i): float(per_pos_dist[i]) for i in [0, 50, 100, 150, 200, 259, seq_len//2, seq_len-1]},
        }
        print(f"  {c}: visual_mean={np.mean(vis_dists):.6f}, text_mean={np.mean(txt_dists):.6f}" if txt_dists else f"  {c}: all_mean={np.mean(vis_dists):.6f}")
    results["position_sensitivity"] = position_sensitivity

    # === Test 2: Mean-pooled visual tokens vs last token for detection ===
    print("\n=== Pooling Strategy Comparison ===")
    # Extract embeddings using different strategies
    pooling_results = {}
    strategies = {
        "last_token": lambda h: h[-1],
        "mean_all": lambda h: np.mean(h, axis=0),
        "mean_visual": lambda h: np.mean(h[:min(260, len(h))], axis=0),
        "mean_text": lambda h: np.mean(h[260:], axis=0) if len(h) > 260 else h[-1],
        "first_token": lambda h: h[0],
        "max_pool": lambda h: np.max(h, axis=0),
    }

    for sname, sfunc in strategies.items():
        clean_embs = []
        for s in scenes:
            h = get_all_hidden(model, processor, s, prompt)
            clean_embs.append(sfunc(h))
        centroid = np.mean(clean_embs, axis=0)
        clean_dists = [cosine_dist(e, centroid) for e in clean_embs]

        per_corr = {}
        for c in corruptions:
            ood_dists = []
            for s in scenes:
                h = get_all_hidden(model, processor, apply_corruption(s, c), prompt)
                ood_dists.append(cosine_dist(sfunc(h), centroid))
            auroc = float(compute_auroc(clean_dists, ood_dists))
            per_corr[c] = auroc

        pooling_results[sname] = {
            "auroc_per_corruption": per_corr,
            "mean_auroc": float(np.mean(list(per_corr.values()))),
            "mean_clean_dist": float(np.mean(clean_dists)),
        }
        print(f"  {sname}: mean_auroc={np.mean(list(per_corr.values())):.4f}, per_corr={per_corr}")
    results["pooling_comparison"] = pooling_results

    # === Test 3: Token diversity (how unique are visual tokens?) ===
    print("\n=== Token Diversity ===")
    diversity_results = {}
    for condition in ['clean', 'fog', 'night']:
        img = s0 if condition == 'clean' else apply_corruption(s0, condition)
        h = get_all_hidden(model, processor, img, prompt)
        n_vis = min(260, seq_len)
        vis_tokens = h[:n_vis]

        # Pairwise cosine similarity among visual tokens
        sims = []
        for i in range(0, n_vis, 10):
            for j in range(i+10, n_vis, 10):
                sims.append(1.0 - cosine_dist(vis_tokens[i], vis_tokens[j]))

        diversity_results[condition] = {
            "mean_pairwise_sim": float(np.mean(sims)),
            "std_pairwise_sim": float(np.std(sims)),
            "min_pairwise_sim": float(np.min(sims)),
            "max_pairwise_sim": float(np.max(sims)),
            "token_norm_mean": float(np.mean([np.linalg.norm(vis_tokens[i]) for i in range(n_vis)])),
            "token_norm_std": float(np.std([np.linalg.norm(vis_tokens[i]) for i in range(n_vis)])),
        }
        print(f"  {condition}: pairwise_sim={np.mean(sims):.4f} ± {np.std(sims):.4f}")
    results["token_diversity"] = diversity_results

    # === Test 4: Position norm profile ===
    print("\n=== Position Norm Profile ===")
    norm_profile = {}
    for condition in ['clean', 'fog', 'night']:
        img = s0 if condition == 'clean' else apply_corruption(s0, condition)
        h = get_all_hidden(model, processor, img, prompt)
        norms = [float(np.linalg.norm(h[i])) for i in range(seq_len)]
        # Sample at key positions
        samples = {}
        for pos in list(range(0, min(260, seq_len), 50)) + [259, seq_len-1]:
            if pos < seq_len:
                samples[str(pos)] = norms[pos]
        norm_profile[condition] = {
            "visual_mean_norm": float(np.mean(norms[:min(260, seq_len)])),
            "text_mean_norm": float(np.mean(norms[260:])) if seq_len > 260 else 0,
            "last_norm": float(norms[-1]),
            "sample_norms": samples,
        }
        print(f"  {condition}: visual_norm={norm_profile[condition]['visual_mean_norm']:.4f}, last={norms[-1]:.4f}")
    results["norm_profile"] = norm_profile

    out_path = "/workspace/Vizuara-VLA-Research/experiments/visual_tokens_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
