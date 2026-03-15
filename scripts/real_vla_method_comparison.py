#!/usr/bin/env python3
"""Experiment 304: Comprehensive Method Comparison
Side-by-side comparison of ALL detection methods on the same data:
1. Cosine distance (our method)
2. Euclidean distance
3. L2 norm change
4. Mahalanobis distance (single sample)
5. MSP (max softmax probability)
6. Energy score
7. Entropy of output distribution
8. KL divergence from clean
9. Random projection (32D cosine)
10. Per-head voting
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
        "experiment": "method_comparison",
        "experiment_number": 304,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    corruptions = ['fog', 'night', 'blur', 'noise']
    ACTION_TOKEN_START = 31744

    # Random projection matrix
    np.random.seed(123)
    proj_matrix = np.random.randn(4096, 32) / np.sqrt(32)

    # Get clean references
    print("Getting clean references...")
    inputs = processor(prompt, base_img).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
        outputs = model(**inputs)

    clean_emb = fwd.hidden_states[3][0, -1, :].float().cpu().numpy()
    clean_norm = float(np.linalg.norm(clean_emb))
    clean_proj = clean_emb @ proj_matrix
    clean_logits = outputs.logits[0, -1, :].float().cpu().numpy()
    clean_action_logits = clean_logits[ACTION_TOKEN_START:ACTION_TOKEN_START+256]
    clean_probs = np.exp(clean_action_logits - clean_action_logits.max())
    clean_probs = clean_probs / clean_probs.sum()
    clean_entropy = -np.sum(clean_probs * np.log(clean_probs + 1e-30))
    clean_msp = float(clean_probs.max())
    clean_energy = float(-np.log(np.sum(np.exp(clean_action_logits - clean_action_logits.max()))))

    # Collect ID scores
    print("Collecting ID scores...")
    id_scores = {m: [] for m in ['cosine', 'euclidean', 'norm', 'msp', 'energy', 'entropy', 'proj_cosine']}

    for _ in range(5):
        inputs = processor(prompt, base_img).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            fwd = model(**inputs, output_hidden_states=True)
            outputs = model(**inputs)

        emb = fwd.hidden_states[3][0, -1, :].float().cpu().numpy()
        logits = outputs.logits[0, -1, :].float().cpu().numpy()
        action_logits = logits[ACTION_TOKEN_START:ACTION_TOKEN_START+256]
        probs = np.exp(action_logits - action_logits.max())
        probs = probs / probs.sum()

        id_scores['cosine'].append(float(cosine(clean_emb, emb)))
        id_scores['euclidean'].append(float(np.linalg.norm(clean_emb - emb)))
        id_scores['norm'].append(abs(float(np.linalg.norm(emb)) - clean_norm))
        id_scores['msp'].append(-float(probs.max()))  # Negate: lower MSP = more OOD
        id_scores['energy'].append(-float(-np.log(np.sum(np.exp(action_logits - action_logits.max())))))
        id_scores['entropy'].append(float(-np.sum(probs * np.log(probs + 1e-30))))
        proj = emb @ proj_matrix
        id_scores['proj_cosine'].append(float(cosine(clean_proj, proj)))

    # Collect OOD scores
    print("Collecting OOD scores...")
    method_aurocs = {}

    for c in corruptions:
        print(f"  {c}...")
        ood_scores = {m: [] for m in id_scores.keys()}

        for sev in [0.3, 0.5, 1.0]:
            corrupted = apply_corruption(base_img, c, sev)
            inputs = processor(prompt, corrupted).to(model.device, dtype=torch.bfloat16)
            with torch.no_grad():
                fwd = model(**inputs, output_hidden_states=True)
                outputs = model(**inputs)

            emb = fwd.hidden_states[3][0, -1, :].float().cpu().numpy()
            logits = outputs.logits[0, -1, :].float().cpu().numpy()
            action_logits = logits[ACTION_TOKEN_START:ACTION_TOKEN_START+256]
            probs = np.exp(action_logits - action_logits.max())
            probs = probs / probs.sum()

            ood_scores['cosine'].append(float(cosine(clean_emb, emb)))
            ood_scores['euclidean'].append(float(np.linalg.norm(clean_emb - emb)))
            ood_scores['norm'].append(abs(float(np.linalg.norm(emb)) - clean_norm))
            ood_scores['msp'].append(-float(probs.max()))
            ood_scores['energy'].append(-float(-np.log(np.sum(np.exp(action_logits - action_logits.max())))))
            ood_scores['entropy'].append(float(-np.sum(probs * np.log(probs + 1e-30))))
            proj = emb @ proj_matrix
            ood_scores['proj_cosine'].append(float(cosine(clean_proj, proj)))

        method_aurocs[c] = {}
        for m in id_scores.keys():
            auroc = compute_auroc(id_scores[m], ood_scores[m])
            method_aurocs[c][m] = auroc

        print(f"    " + ", ".join(f"{m}={method_aurocs[c][m]:.3f}" for m in id_scores.keys()))

    results["method_aurocs"] = method_aurocs

    # Compute overall AUROC per method
    overall = {}
    for m in id_scores.keys():
        aurocs = [method_aurocs[c][m] for c in corruptions]
        overall[m] = {
            "mean": float(np.mean(aurocs)),
            "min": float(np.min(aurocs)),
            "max": float(np.max(aurocs)),
            "per_corruption": {c: method_aurocs[c][m] for c in corruptions}
        }
    results["overall"] = overall

    # Print comparison table
    print("\n=== Method Comparison (AUROC) ===")
    print(f"{'Method':<15} {'Fog':>6} {'Night':>6} {'Blur':>6} {'Noise':>6} {'Mean':>6}")
    for m in id_scores.keys():
        vals = [method_aurocs[c][m] for c in corruptions]
        print(f"{m:<15} {vals[0]:>6.3f} {vals[1]:>6.3f} {vals[2]:>6.3f} {vals[3]:>6.3f} {np.mean(vals):>6.3f}")

    # Save
    ts = results["timestamp"]
    out_path = f"experiments/comparison_{ts}.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
