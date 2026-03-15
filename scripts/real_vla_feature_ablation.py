#!/usr/bin/env python3
"""Experiment 161: Feature ablation — which embedding dimensions carry OOD signal.

Systematically ablates (zeros out) groups of embedding dimensions at L3 and L32
to identify which feature subspaces carry the most OOD-discriminative information.
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

def extract_hidden(model, processor, image, prompt, layers):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return {l: fwd.hidden_states[l][0, -1, :].float().cpu().numpy() for l in layers}

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
    print("Experiment 161: Feature Ablation — OOD Signal Localization")
    print("=" * 60, flush=True)

    from transformers import AutoModelForVision2Seq, AutoProcessor
    print("Loading OpenVLA-7B...", flush=True)
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to drive forward at 25 m/s safely?\nOut:"
    layers = [3, 32]

    creators = [create_highway, create_urban, create_rural]
    n_cal = 8
    n_test = 6

    # Extract all embeddings
    print("\n--- Extracting embeddings ---", flush=True)
    cal_arrs = [creators[i%3](i) for i in range(n_cal)]
    test_arrs = [creators[(i+n_cal)%3](i+n_cal) for i in range(n_test)]

    cal_embs = {l: [] for l in layers}
    for arr in cal_arrs:
        h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
        for l in layers:
            cal_embs[l].append(h[l])

    id_embs = {l: [] for l in layers}
    for arr in test_arrs:
        h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
        for l in layers:
            id_embs[l].append(h[l])

    ood_transforms = {
        "fog_60": lambda a: apply_fog(a, 0.6),
        "night": apply_night,
        "blur": apply_blur,
        "noise": apply_noise,
    }
    ood_embs = {l: [] for l in layers}
    for cat, tfn in ood_transforms.items():
        for j in range(n_test):
            arr = tfn(test_arrs[j % n_test])
            h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
            for l in layers:
                ood_embs[l].append(h[l])
    print("  All embeddings extracted.", flush=True)

    # Feature ablation: divide 4096 dims into groups
    dim = 4096
    group_sizes = [64, 128, 256, 512]  # different granularities

    results = {}
    for l in layers:
        layer_results = {}
        centroid = np.array(cal_embs[l]).mean(axis=0)
        
        # Baseline AUROC (no ablation)
        id_dists = [cosine_distance(e, centroid) for e in id_embs[l]]
        ood_dists = [cosine_distance(e, centroid) for e in ood_embs[l]]
        baseline_auroc = compute_auroc(id_dists, ood_dists)
        layer_results["baseline_auroc"] = baseline_auroc
        print(f"\n  L{l} baseline AUROC: {baseline_auroc:.4f}", flush=True)

        # PCA to find most important directions
        centered = np.array(cal_embs[l]) - centroid
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        
        # Ablate PCA dimensions
        max_pca = len(S)  # max available PCA components
        pca_ablation = {}
        for k in [1, 2, 5, 10, 20, 50]:
            k_eff = min(k, max_pca)
            # Keep only top-k PCA dimensions
            proj = Vt[:k_eff]
            id_proj = [(e - centroid) @ proj.T for e in id_embs[l]]
            ood_proj = [(e - centroid) @ proj.T for e in ood_embs[l]]
            centroid_proj = np.zeros(k_eff)

            id_dists_k = [float(np.linalg.norm(p - centroid_proj)) for p in id_proj]
            ood_dists_k = [float(np.linalg.norm(p - centroid_proj)) for p in ood_proj]
            auroc_k = compute_auroc(id_dists_k, ood_dists_k)
            pca_ablation[k] = auroc_k
            print(f"    PCA top-{k} (eff={k_eff}): AUROC={auroc_k:.4f}", flush=True)
        layer_results["pca_ablation"] = pca_ablation

        # Ablate by zeroing out blocks of raw dimensions
        for gs in group_sizes:
            n_groups = dim // gs
            block_aurocs = []
            for g in range(n_groups):
                start, end = g * gs, (g + 1) * gs
                # Zero out this block
                centroid_abl = centroid.copy()
                centroid_abl[start:end] = 0
                id_dists_abl = [cosine_distance(np.concatenate([e[:start], np.zeros(gs), e[end:]]), centroid_abl) for e in id_embs[l]]
                ood_dists_abl = [cosine_distance(np.concatenate([e[:start], np.zeros(gs), e[end:]]), centroid_abl) for e in ood_embs[l]]
                auroc_abl = compute_auroc(id_dists_abl, ood_dists_abl)
                block_aurocs.append(auroc_abl)
            
            # Find most important blocks (AUROC drops most when removed)
            drops = [baseline_auroc - a for a in block_aurocs]
            top_blocks = sorted(range(len(drops)), key=lambda i: -drops[i])[:5]
            
            layer_results[f"block_{gs}"] = {
                "aurocs": block_aurocs,
                "max_drop": float(max(drops)),
                "min_drop": float(min(drops)),
                "mean_drop": float(np.mean(drops)),
                "top_blocks": [(b, float(drops[b]), b*gs, (b+1)*gs) for b in top_blocks],
            }
            print(f"    Block-{gs}: max_drop={max(drops):.4f}, mean_drop={np.mean(drops):.4f}", flush=True)

        # Singular value spectrum analysis
        layer_results["singular_values"] = S.tolist()
        cumvar = np.cumsum(S**2) / np.sum(S**2)
        dims_for_90 = int(np.searchsorted(cumvar, 0.90)) + 1
        dims_for_95 = int(np.searchsorted(cumvar, 0.95)) + 1
        dims_for_99 = int(np.searchsorted(cumvar, 0.99)) + 1
        layer_results["dims_for_90pct"] = dims_for_90
        layer_results["dims_for_95pct"] = dims_for_95
        layer_results["dims_for_99pct"] = dims_for_99
        print(f"    Dims for 90%: {dims_for_90}, 95%: {dims_for_95}, 99%: {dims_for_99}", flush=True)

        results[f"L{l}"] = layer_results

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "feature_ablation",
        "experiment_number": 161,
        "timestamp": ts,
        "n_cal": n_cal, "n_test_id": n_test,
        "n_ood_total": len(ood_embs[layers[0]]),
        "ood_categories": list(ood_transforms.keys()),
        "layers": layers,
        "dim": dim,
        "results": results,
    }
    path = os.path.join(RESULTS_DIR, f"feature_ablation_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
