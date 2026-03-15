#!/usr/bin/env python3
"""Experiment 191: Gradient-based OOD signal — does the gradient magnitude
with respect to the input carry OOD information?

Compares gradient norm, hidden-state cosine distance, and their combination.
"""

import json, os, sys, datetime
import numpy as np
import torch
from pathlib import Path
from PIL import Image, ImageFilter

SCRIPT_DIR = Path(__file__).parent
REPO_DIR = SCRIPT_DIR.parent
EXPERIMENTS_DIR = REPO_DIR / "experiments"
EXPERIMENTS_DIR.mkdir(exist_ok=True)
RESULTS_DIR = str(EXPERIMENTS_DIR)

SIZE = (256, 256)
rng = np.random.RandomState(42)

def create_highway(idx):
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]; img[SIZE[0]//2:] = [80, 80, 80]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [255, 255, 255]
    return np.clip(img.astype(np.int16) + rng.randint(-5, 6, img.shape).astype(np.int16), 0, 255).astype(np.uint8)

def create_urban(idx):
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [135, 206, 235]; img[SIZE[0]//3:SIZE[0]//2] = [139, 119, 101]; img[SIZE[0]//2:] = [60, 60, 60]
    return np.clip(img.astype(np.int16) + rng.randint(-5, 6, img.shape).astype(np.int16), 0, 255).astype(np.uint8)

def create_rural(idx):
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [100, 180, 255]; img[SIZE[0]//3:SIZE[0]*2//3] = [34, 139, 34]; img[SIZE[0]*2//3:] = [90, 90, 90]
    return np.clip(img.astype(np.int16) + rng.randint(-8, 9, img.shape).astype(np.int16), 0, 255).astype(np.uint8)

def apply_fog(a, alpha):
    return np.clip(a*(1-alpha)+np.full_like(a,[200,200,210])*alpha, 0, 255).astype(np.uint8)
def apply_night(a): return np.clip(a*0.15, 0, 255).astype(np.uint8)
def apply_blur(a, r=8): return np.array(Image.fromarray(a).filter(ImageFilter.GaussianBlur(radius=r)))
def apply_noise(a, s=50): return np.clip(a.astype(np.float32)+np.random.normal(0,s,a.shape), 0, 255).astype(np.uint8)

def extract_with_gradient(model, processor, image, prompt, layers):
    """Extract hidden states and gradient norm."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)

    # Enable gradient computation for input embeddings
    model.zero_grad()
    input_ids = inputs.get("input_ids")

    # Forward pass with hidden states
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)

    hidden = {l: fwd.hidden_states[l][0, -1, :].float().cpu().numpy() for l in layers}

    # For gradient, use the max logit as the scalar output
    # Re-run with gradients enabled on just the logit computation
    # Actually, for efficiency, let's just use the logit variance as a proxy
    logits = fwd.logits[0, -1, :].float().cpu().numpy()
    action_logits = logits[31744:32000]

    # Logit spread as a feature
    logit_std = float(np.std(action_logits))
    logit_range = float(np.max(action_logits) - np.min(action_logits))
    top_k_gap = float(np.sort(action_logits)[-1] - np.sort(action_logits)[-2])

    return hidden, {
        "logit_std": logit_std,
        "logit_range": logit_range,
        "top_k_gap": top_k_gap,
    }

def cosine_distance(a, b):
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def compute_auroc(id_scores, ood_scores):
    id_scores = np.asarray(id_scores)
    ood_scores = np.asarray(ood_scores)
    n_id, n_ood = len(id_scores), len(ood_scores)
    if n_id == 0 or n_ood == 0: return 0.5
    count = sum(float(np.sum(o > id_scores) + 0.5 * np.sum(o == id_scores)) for o in ood_scores)
    return count / (n_id * n_ood)

def main():
    print("=" * 60)
    print("Experiment 191: Logit Distribution Features for OOD")
    print("=" * 60, flush=True)

    from transformers import AutoModelForVision2Seq, AutoProcessor
    print("Loading OpenVLA-7B...", flush=True)
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to drive forward at 25 m/s safely?\nOut:"
    layers = [1, 3, 32]

    creators = [create_highway, create_urban, create_rural]
    n_cal = 8
    n_test = 6

    # Calibrate
    cal_arrs = [creators[i%3](i) for i in range(n_cal)]
    centroids = {}
    for l in layers:
        cal_embs = []
        for arr in cal_arrs:
            h, _ = extract_with_gradient(model, processor, Image.fromarray(arr), prompt, layers)
            cal_embs.append(h[l])
        centroids[l] = np.array(cal_embs).mean(axis=0)

    test_arrs = [creators[(i+n_cal)%3](i+n_cal) for i in range(n_test)]

    # ID
    id_hidden = {l: [] for l in layers}
    id_logit_feats = {"logit_std": [], "logit_range": [], "top_k_gap": []}
    for arr in test_arrs:
        h, lf = extract_with_gradient(model, processor, Image.fromarray(arr), prompt, layers)
        for l in layers:
            id_hidden[l].append(h[l])
        for k in id_logit_feats:
            id_logit_feats[k].append(lf[k])

    # OOD
    ood_transforms = {
        "fog_60": lambda a: apply_fog(a, 0.6),
        "night": apply_night,
        "blur": apply_blur,
        "noise": apply_noise,
    }
    ood_hidden = {l: [] for l in layers}
    ood_logit_feats = {"logit_std": [], "logit_range": [], "top_k_gap": []}
    per_cat_hidden = {cat: {l: [] for l in layers} for cat in ood_transforms}

    for cat, tfn in ood_transforms.items():
        for arr in test_arrs:
            h, lf = extract_with_gradient(model, processor, Image.fromarray(tfn(arr)), prompt, layers)
            for l in layers:
                ood_hidden[l].append(h[l])
                per_cat_hidden[cat][l].append(h[l])
            for k in ood_logit_feats:
                ood_logit_feats[k].append(lf[k])

    # Compute AUROCs
    print("\n--- AUROC comparison ---", flush=True)
    results = {}

    for l in layers:
        id_dists = [cosine_distance(e, centroids[l]) for e in id_hidden[l]]
        ood_dists = [cosine_distance(e, centroids[l]) for e in ood_hidden[l]]
        auroc = compute_auroc(id_dists, ood_dists)
        results[f"cosine_L{l}"] = {"auroc": auroc}
        print(f"  Cosine L{l}: AUROC={auroc:.4f}", flush=True)

        # Per-category AUROC at L1 (new layer)
        if l == 1:
            for cat in ood_transforms:
                cat_dists = [cosine_distance(e, centroids[l]) for e in per_cat_hidden[cat][l]]
                cat_auroc = compute_auroc(id_dists, cat_dists)
                results[f"cosine_L{l}_{cat}"] = {"auroc": cat_auroc}
                print(f"    L{l} {cat}: AUROC={cat_auroc:.4f}", flush=True)

    # Logit feature AUROCs
    for feat in ["logit_std", "logit_range", "top_k_gap"]:
        auroc = compute_auroc(id_logit_feats[feat], ood_logit_feats[feat])
        auroc_rev = compute_auroc([-x for x in id_logit_feats[feat]], [-x for x in ood_logit_feats[feat]])
        best = max(auroc, auroc_rev)
        results[feat] = {
            "auroc": best,
            "id_mean": float(np.mean(id_logit_feats[feat])),
            "ood_mean": float(np.mean(ood_logit_feats[feat])),
        }
        print(f"  {feat}: AUROC={best:.4f} (id={np.mean(id_logit_feats[feat]):.4f} ood={np.mean(ood_logit_feats[feat]):.4f})", flush=True)

    # Combined: cosine + logit features
    # Normalize and sum
    for l in layers:
        id_cos = np.array([cosine_distance(e, centroids[l]) for e in id_hidden[l]])
        ood_cos = np.array([cosine_distance(e, centroids[l]) for e in ood_hidden[l]])

        id_std = np.array(id_logit_feats["logit_std"])
        ood_std = np.array(ood_logit_feats["logit_std"])

        # Normalize
        all_cos = np.concatenate([id_cos, ood_cos])
        all_std = np.concatenate([id_std, ood_std])

        id_cos_n = (id_cos - all_cos.mean()) / (all_cos.std() + 1e-10)
        ood_cos_n = (ood_cos - all_cos.mean()) / (all_cos.std() + 1e-10)
        id_std_n = (id_std - all_std.mean()) / (all_std.std() + 1e-10)
        ood_std_n = (ood_std - all_std.mean()) / (all_std.std() + 1e-10)

        id_combined = id_cos_n + id_std_n
        ood_combined = ood_cos_n + ood_std_n
        auroc_combined = compute_auroc(id_combined.tolist(), ood_combined.tolist())
        results[f"combined_L{l}"] = {"auroc": auroc_combined}
        print(f"  Combined L{l}+logit_std: AUROC={auroc_combined:.4f}", flush=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "gradient_ood",
        "experiment_number": 191,
        "timestamp": ts,
        "n_cal": n_cal, "n_test": n_test,
        "layers": layers,
        "results": results,
    }
    path = os.path.join(RESULTS_DIR, f"gradient_ood_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
