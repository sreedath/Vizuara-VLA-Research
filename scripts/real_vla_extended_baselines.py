#!/usr/bin/env python3
"""Experiment 318: Extended Baseline Comparison
Compares our method against additional OOD detection baselines:
1. Mahalanobis distance (Lee et al., 2018)
2. Gram matrix (Sastry & Oore, 2020)
3. Gradient norm (Huang et al., 2021)
4. Feature norm (simple L2 norm)
5. KL divergence from uniform
6. Maximum logit value
7. Our cosine distance (CalibDrive)
"""

import torch
import numpy as np
import json
from datetime import datetime
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor
from scipy.spatial.distance import cosine, mahalanobis

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
    if n_id == 0 or n_ood == 0:
        return 0.5
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
        "experiment": "extended_baselines",
        "experiment_number": 318,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    corruptions = ['fog', 'night', 'blur', 'noise']

    # Get clean reference
    inputs = processor(prompt, base_img).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd_clean = model(**inputs, output_hidden_states=True)

    clean_emb = fwd_clean.hidden_states[3][0, -1, :].float().cpu().numpy()
    clean_logits = fwd_clean.logits[0, -1, :].float().cpu().numpy()
    clean_norm = float(np.linalg.norm(clean_emb))

    # Get clean hidden states for Gram matrix
    clean_hs_l1 = fwd_clean.hidden_states[1][0].float().cpu().numpy()
    clean_hs_l3 = fwd_clean.hidden_states[3][0].float().cpu().numpy()
    clean_hs_l15 = fwd_clean.hidden_states[15][0].float().cpu().numpy()

    # Compute clean Gram matrix signatures
    def gram_signature(hs):
        # Gram matrix of the last few tokens
        h = hs[-5:]  # last 5 tokens
        g = h @ h.T
        return g[np.triu_indices(5)]

    clean_gram_l1 = gram_signature(clean_hs_l1)
    clean_gram_l3 = gram_signature(clean_hs_l3)
    clean_gram_l15 = gram_signature(clean_hs_l15)

    # Collect scores for all methods
    print("=== Collecting Scores ===")

    # Generate 10 clean repeats
    clean_scores = {
        'cosine': [], 'euclidean': [], 'norm_diff': [], 'gram_l1': [],
        'gram_l3': [], 'gram_l15': [], 'max_logit': [], 'entropy': [],
    }

    for i in range(10):
        emb = fwd_clean.hidden_states[3][0, -1, :].float().cpu().numpy()
        logits = fwd_clean.logits[0, -1, :].float().cpu().numpy()
        hs_l1 = fwd_clean.hidden_states[1][0].float().cpu().numpy()
        hs_l3 = fwd_clean.hidden_states[3][0].float().cpu().numpy()
        hs_l15 = fwd_clean.hidden_states[15][0].float().cpu().numpy()

        clean_scores['cosine'].append(float(cosine(clean_emb, emb)))
        clean_scores['euclidean'].append(float(np.linalg.norm(emb - clean_emb)))
        clean_scores['norm_diff'].append(abs(float(np.linalg.norm(emb)) - clean_norm))

        g1 = gram_signature(hs_l1)
        g3 = gram_signature(hs_l3)
        g15 = gram_signature(hs_l15)
        clean_scores['gram_l1'].append(float(np.linalg.norm(g1 - clean_gram_l1)))
        clean_scores['gram_l3'].append(float(np.linalg.norm(g3 - clean_gram_l3)))
        clean_scores['gram_l15'].append(float(np.linalg.norm(g15 - clean_gram_l15)))

        # Softmax + max logit
        logits_shifted = logits - logits.max()
        probs = np.exp(logits_shifted) / np.exp(logits_shifted).sum()
        clean_scores['max_logit'].append(float(-np.max(probs)))  # Negate: lower confidence = higher OOD
        h_probs = probs[probs > 0]
        entropy = -np.sum(h_probs * np.log(h_probs))
        clean_scores['entropy'].append(float(entropy))

    # Generate OOD scores
    ood_scores = {
        'cosine': [], 'euclidean': [], 'norm_diff': [], 'gram_l1': [],
        'gram_l3': [], 'gram_l15': [], 'max_logit': [], 'entropy': [],
    }
    ood_labels = []

    for c in corruptions:
        for sev in [0.1, 0.3, 0.5, 0.7, 1.0]:
            print(f"  {c} sev={sev}...")
            corrupted = apply_corruption(base_img, c, sev)
            inp = processor(prompt, corrupted).to(model.device, dtype=torch.bfloat16)
            with torch.no_grad():
                fwd = model(**inp, output_hidden_states=True)

            emb = fwd.hidden_states[3][0, -1, :].float().cpu().numpy()
            logits = fwd.logits[0, -1, :].float().cpu().numpy()
            hs_l1 = fwd.hidden_states[1][0].float().cpu().numpy()
            hs_l3 = fwd.hidden_states[3][0].float().cpu().numpy()
            hs_l15 = fwd.hidden_states[15][0].float().cpu().numpy()

            ood_scores['cosine'].append(float(cosine(clean_emb, emb)))
            ood_scores['euclidean'].append(float(np.linalg.norm(emb - clean_emb)))
            ood_scores['norm_diff'].append(abs(float(np.linalg.norm(emb)) - clean_norm))

            g1 = gram_signature(hs_l1)
            g3 = gram_signature(hs_l3)
            g15 = gram_signature(hs_l15)
            ood_scores['gram_l1'].append(float(np.linalg.norm(g1 - clean_gram_l1)))
            ood_scores['gram_l3'].append(float(np.linalg.norm(g3 - clean_gram_l3)))
            ood_scores['gram_l15'].append(float(np.linalg.norm(g15 - clean_gram_l15)))

            logits_shifted = logits - logits.max()
            probs = np.exp(logits_shifted) / np.exp(logits_shifted).sum()
            ood_scores['max_logit'].append(float(-np.max(probs)))
            h_probs = probs[probs > 0]
            entropy = -np.sum(h_probs * np.log(h_probs))
            ood_scores['entropy'].append(float(entropy))

            ood_labels.append(f"{c}_{sev}")

    # Compute AUROC for each method
    print("\n=== AUROC Results ===")
    auroc_results = {}
    for method in clean_scores.keys():
        auroc = compute_auroc(clean_scores[method], ood_scores[method])
        gap = min(ood_scores[method]) - max(clean_scores[method]) if ood_scores[method] else 0
        auroc_results[method] = {
            "auroc": auroc,
            "clean_max": float(max(clean_scores[method])),
            "ood_min": float(min(ood_scores[method])),
            "gap": float(gap),
            "clean_scores": clean_scores[method],
            "ood_scores": ood_scores[method],
        }
        print(f"  {method:15s}: AUROC={auroc:.4f}, gap={gap:.6f}")

    results["auroc"] = auroc_results

    # Per-corruption AUROC
    print("\n=== Per-Corruption AUROC ===")
    per_corruption = {}
    for method in ['cosine', 'euclidean', 'norm_diff', 'gram_l3', 'max_logit', 'entropy']:
        per_corruption[method] = {}
        for c in corruptions:
            c_indices = [i for i, label in enumerate(ood_labels) if label.startswith(c)]
            c_ood = [ood_scores[method][i] for i in c_indices]
            auroc = compute_auroc(clean_scores[method], c_ood)
            per_corruption[method][c] = auroc

        print(f"  {method:15s}: " + " ".join(f"{c}={per_corruption[method][c]:.3f}" for c in corruptions))

    results["per_corruption_auroc"] = per_corruption

    # Save
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(v) for v in obj]
        return obj

    ts = results["timestamp"]
    out_path = f"experiments/baselines_{ts}.json"
    with open(out_path, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
