#!/usr/bin/env python3
"""Experiment 150: PCA reconstruction-based OOD detection at L3.

Uses PCA reconstruction error as an OOD score, leveraging the finding that
the L3 ID manifold is 2-dimensional. OOD images should have high reconstruction
error when projected onto the 2D ID subspace.
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
def apply_occlusion(a):
    o=a.copy(); h,w=o.shape[:2]; o[h//4:3*h//4, w//4:3*w//4]=128; return o
def apply_snow(a):
    o=a.astype(np.float32)*0.7+76.5; o[np.random.random(a.shape[:2])>0.97]=255
    return np.clip(o,0,255).astype(np.uint8)
def apply_overexpose(a): return np.clip(a.astype(np.float32)*3.0, 0, 255).astype(np.uint8)

def extract_hidden(model, processor, image, prompt, layers):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return {l: fwd.hidden_states[l][0, -1, :].float().cpu().numpy() for l in layers}

def cosine_distance(a, b):
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def compute_auroc(id_scores, ood_scores):
    n_id, n_ood = len(id_scores), len(ood_scores)
    correct = sum(1 for o in ood_scores for i in id_scores if o > i)
    correct += 0.5 * sum(1 for o in ood_scores for i in id_scores if o == i)
    return correct / (n_id * n_ood) if n_id * n_ood > 0 else 0.5

def compute_dprime(id_scores, ood_scores):
    id_arr, ood_arr = np.array(id_scores), np.array(ood_scores)
    pooled = np.sqrt((np.var(id_arr) + np.var(ood_arr)) / 2 + 1e-10)
    return float((np.mean(ood_arr) - np.mean(id_arr)) / pooled)

def main():
    print("=" * 60)
    print("Experiment 150: PCA Reconstruction-Based OOD Detection")
    print("=" * 60, flush=True)

    from transformers import AutoModelForVision2Seq, AutoProcessor
    print("Loading OpenVLA-7B...", flush=True)
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to drive forward at 25 m/s safely?\nOut:"
    creators = [create_highway, create_urban, create_rural]
    layers = [3, 32]

    n_cal = 15
    n_test = 10
    cal_arrs = [creators[i%3](i) for i in range(n_cal)]
    test_arrs = [creators[(i+n_cal)%3](i+n_cal) for i in range(n_test)]

    ood_transforms = {
        "fog_30": lambda a: apply_fog(a, 0.3),
        "fog_60": lambda a: apply_fog(a, 0.6),
        "night": apply_night, "blur": apply_blur,
        "noise": apply_noise, "occlusion": apply_occlusion,
        "snow": apply_snow, "overexpose": apply_overexpose,
    }
    n_ood_per = 6

    # Calibration
    print(f"\n--- Calibration (n={n_cal}) ---", flush=True)
    cal_embs = {l: [] for l in layers}
    for i, arr in enumerate(cal_arrs):
        h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
        for l in layers:
            cal_embs[l].append(h[l])
        if (i+1) % 5 == 0:
            print(f"  Cal {i+1}/{n_cal}", flush=True)

    # Build PCA model and cosine stats
    pca_models = {}
    cosine_stats = {}
    k_values = [1, 2, 3, 5, 10]
    for l in layers:
        embs = np.array(cal_embs[l])
        centroid = embs.mean(axis=0)
        centered = embs - centroid
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        pca_models[l] = {"centroid": centroid, "Vt": Vt, "S": S}
        # Cosine stats
        dists = [cosine_distance(e, centroid) for e in embs]
        cosine_stats[l] = {"mean": float(np.mean(dists)), "std": float(np.std(dists))}

    def recon_error(emb, layer, k):
        m = pca_models[layer]
        diff = emb - m["centroid"]
        Vk = m["Vt"][:k]
        proj = diff @ Vk.T @ Vk
        return float(np.sum((diff - proj)**2))

    # Test
    print(f"\n--- Test ID (n={n_test}) ---", flush=True)
    id_scores = {l: {"cosine": [], **{f"recon_k{k}": [] for k in k_values}} for l in layers}
    for i, arr in enumerate(test_arrs):
        h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
        for l in layers:
            id_scores[l]["cosine"].append(cosine_distance(h[l], pca_models[l]["centroid"]))
            for k in k_values:
                id_scores[l][f"recon_k{k}"].append(recon_error(h[l], l, k))
        print(f"  ID {i+1}/{n_test}", flush=True)

    print(f"\n--- OOD ({len(ood_transforms)} cats) ---", flush=True)
    cats = list(ood_transforms.keys())
    ood_scores = {l: {c: {"cosine": [], **{f"recon_k{k}": [] for k in k_values}} for c in cats} for l in layers}
    for cat, tfn in ood_transforms.items():
        for j in range(n_ood_per):
            arr = tfn(test_arrs[j % n_test])
            h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
            for l in layers:
                ood_scores[l][cat]["cosine"].append(cosine_distance(h[l], pca_models[l]["centroid"]))
                for k in k_values:
                    ood_scores[l][cat][f"recon_k{k}"].append(recon_error(h[l], l, k))
        print(f"  {cat}: done", flush=True)

    # Compute metrics
    print("\n--- Metrics ---", flush=True)
    results = {}
    metrics = ["cosine"] + [f"recon_k{k}" for k in k_values]
    for l in layers:
        ln = f"L{l}"
        results[ln] = {}
        for m in metrics:
            all_ood = []
            per_cat = {}
            for c in cats:
                auroc = compute_auroc(id_scores[l][m], ood_scores[l][c][m])
                dp = compute_dprime(id_scores[l][m], ood_scores[l][c][m])
                per_cat[c] = {"auroc": auroc, "d_prime": dp}
                all_ood.extend(ood_scores[l][c][m])
            overall_auroc = compute_auroc(id_scores[l][m], all_ood)
            overall_dp = compute_dprime(id_scores[l][m], all_ood)
            results[ln][m] = {"overall_auroc": overall_auroc, "overall_d_prime": overall_dp, "per_category": per_cat}

    # Print
    print("\n" + "=" * 80)
    for ln in ["L3", "L32"]:
        print(f"\n--- {ln} ---")
        for m in metrics:
            r = results[ln][m]
            print(f"  {m:15s}: AUROC={r['overall_auroc']:.4f}  d'={r['overall_d_prime']:.1f}")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {"experiment": "recon_ood", "experiment_number": 150, "timestamp": ts,
              "n_cal": n_cal, "n_test_id": n_test, "n_ood_per_cat": n_ood_per,
              "ood_categories": cats, "layers": layers, "k_values": k_values,
              "results": results}
    path = os.path.join(RESULTS_DIR, f"recon_ood_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
