#!/usr/bin/env python3
"""Experiment 158: Ensemble OOD scoring.

Combines cosine distance, Mahalanobis distance, and PCA reconstruction error
into a single ensemble score. Tests whether the combination outperforms
individual metrics.
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

def extract_hidden(model, processor, image, prompt, layers):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return {l: fwd.hidden_states[l][0, -1, :].float().cpu().numpy() for l in layers}

def cosine_distance(a, b):
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def compute_auroc(id_scores, ood_scores):
    """Compute AUROC where higher score = more likely OOD."""
    id_scores = np.asarray(id_scores)
    ood_scores = np.asarray(ood_scores)
    n_id = len(id_scores)
    n_ood = len(ood_scores)
    if n_id == 0 or n_ood == 0:
        return 0.5
    # Wilcoxon-Mann-Whitney statistic
    count = 0
    for o in ood_scores:
        count += np.sum(o > id_scores) + 0.5 * np.sum(o == id_scores)
    return float(count / (n_id * n_ood))

def main():
    print("=" * 60)
    print("Experiment 158: Ensemble OOD Scoring")
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
    n_cal = 10
    n_test = 8

    # Calibration
    print("\n--- Calibrating ---", flush=True)
    cal_arrs = [creators[i%3](i) for i in range(n_cal)]
    cal_embs = {l: [] for l in layers}
    for arr in cal_arrs:
        h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
        for l in layers:
            cal_embs[l].append(h[l])

    centroids = {}
    cal_stats = {}
    pca_components = {}
    mahal_params = {}

    for l in layers:
        embs = np.array(cal_embs[l])
        centroid = embs.mean(axis=0)
        centroids[l] = centroid

        # Cosine stats
        cos_dists = [cosine_distance(e, centroid) for e in embs]
        cal_stats[l] = {"cos_mean": float(np.mean(cos_dists)), "cos_std": float(np.std(cos_dists))}

        # PCA components (for reconstruction)
        centered = embs - centroid
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        pca_components[l] = Vt[:2]  # top-2 PCA directions

        # PCA recon stats
        recon = (centered @ Vt[:2].T) @ Vt[:2]
        recon_errors = np.linalg.norm(centered - recon, axis=1)
        cal_stats[l]["recon_mean"] = float(np.mean(recon_errors))
        cal_stats[l]["recon_std"] = float(np.std(recon_errors))

        # Mahalanobis (PCA-based for numerical stability)
        k_maha = min(len(embs) - 1, 50)
        proj = centered @ Vt[:k_maha].T
        cov_diag = np.var(proj, axis=0) + 1e-6
        mahal_params[l] = {"Vt_k": Vt[:k_maha], "cov_diag": cov_diag}
        maha_dists = [float(np.sqrt(np.sum(((e - centroid) @ Vt[:k_maha].T)**2 / cov_diag))) for e in embs]
        cal_stats[l]["maha_mean"] = float(np.mean(maha_dists))
        cal_stats[l]["maha_std"] = float(np.std(maha_dists))

    print("  Calibration done.", flush=True)

    # Test
    test_arrs = [creators[(i+n_cal)%3](i+n_cal) for i in range(n_test)]
    ood_transforms = {
        "fog_30": lambda a: apply_fog(a, 0.3),
        "fog_60": lambda a: apply_fog(a, 0.6),
        "night": apply_night,
        "blur": apply_blur,
        "noise": apply_noise,
        "occlusion": apply_occlusion,
    }
    n_ood_per = 5

    def compute_scores(emb_dict):
        scores = {}
        for l in layers:
            e = emb_dict[l]
            scores[f"cos_L{l}"] = cosine_distance(e, centroids[l])
            centered = e - centroids[l]
            recon = (centered @ pca_components[l].T) @ pca_components[l]
            scores[f"recon_L{l}"] = float(np.linalg.norm(centered - recon))
            proj = centered @ mahal_params[l]["Vt_k"].T
            scores[f"maha_L{l}"] = float(np.sqrt(np.sum(proj**2 / mahal_params[l]["cov_diag"])))
        return scores

    def normalize_score(val, cal_mean, cal_std):
        return (val - cal_mean) / (cal_std + 1e-10)

    # Collect all scores
    id_raw = []
    for arr in test_arrs:
        h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
        id_raw.append(compute_scores(h))

    ood_raw = {cat: [] for cat in ood_transforms}
    for cat, tfn in ood_transforms.items():
        print(f"  OOD: {cat}", flush=True)
        for j in range(n_ood_per):
            arr = tfn(test_arrs[j % n_test])
            h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
            ood_raw[cat].append(compute_scores(h))

    score_keys = list(id_raw[0].keys())
    print(f"\n  Score dimensions: {score_keys}", flush=True)

    # Print raw score statistics for debugging
    print("\n  Raw score stats:")
    for key in score_keys:
        id_vals = [s[key] for s in id_raw]
        ood_vals = [s[key] for s in sum(ood_raw.values(), [])]
        print(f"    {key}: ID mean={np.mean(id_vals):.6f} OOD mean={np.mean(ood_vals):.6f}", flush=True)

    # Individual AUROC
    individual_aurocs = {}
    for key in score_keys:
        id_vals = np.array([s[key] for s in id_raw])
        ood_vals = np.array([s[key] for s in sum(ood_raw.values(), [])])
        individual_aurocs[key] = compute_auroc(id_vals, ood_vals)

    # Ensemble combinations
    ensembles = {
        "cos_L3+L32": ["cos_L3", "cos_L32"],
        "maha_L3+L32": ["maha_L3", "maha_L32"],
        "recon_L3+L32": ["recon_L3", "recon_L32"],
        "cos+maha_L3": ["cos_L3", "maha_L3"],
        "cos+maha_L32": ["cos_L32", "maha_L32"],
        "cos+recon_L3": ["cos_L3", "recon_L3"],
        "all_L3": ["cos_L3", "maha_L3", "recon_L3"],
        "all_L32": ["cos_L32", "maha_L32", "recon_L32"],
        "all_6": score_keys,
        "cos_L3+maha_L3+cos_L32": ["cos_L3", "maha_L3", "cos_L32"],
    }

    cal_norms = {}
    for key in score_keys:
        l = int(key.split("L")[1])
        metric = key.split("_L")[0]
        cal_norms[key] = {
            "mean": cal_stats[l].get(f"{metric}_mean", 0),
            "std": cal_stats[l].get(f"{metric}_std", 1),
        }

    ensemble_aurocs = {}
    for ename, ekeys in ensembles.items():
        id_ensemble = np.array([
            sum(normalize_score(s[k], cal_norms[k]["mean"], cal_norms[k]["std"]) for k in ekeys)
            for s in id_raw
        ])
        ood_ensemble = np.array([
            sum(normalize_score(s[k], cal_norms[k]["mean"], cal_norms[k]["std"]) for k in ekeys)
            for s in sum(ood_raw.values(), [])
        ])
        ensemble_aurocs[ename] = compute_auroc(id_ensemble, ood_ensemble)

    # Per-category AUROC
    per_cat_aurocs = {}
    for ename in ["cos_L3+L32", "all_6", "cos_L3+maha_L3+cos_L32"]:
        ekeys = ensembles[ename]
        per_cat = {}
        for cat in ood_transforms:
            id_e = np.array([sum(normalize_score(s[k], cal_norms[k]["mean"], cal_norms[k]["std"]) for k in ekeys) for s in id_raw])
            ood_e = np.array([sum(normalize_score(s[k], cal_norms[k]["mean"], cal_norms[k]["std"]) for k in ekeys) for s in ood_raw[cat]])
            per_cat[cat] = compute_auroc(id_e, ood_e)
        per_cat_aurocs[ename] = per_cat

    # Print results
    print("\n" + "=" * 80)
    print("INDIVIDUAL METRIC AUROC")
    for k, v in sorted(individual_aurocs.items(), key=lambda x: -x[1]):
        print(f"  {k:<15s}: AUROC = {v:.4f}")

    print("\nENSEMBLE AUROC")
    for k, v in sorted(ensemble_aurocs.items(), key=lambda x: -x[1]):
        print(f"  {k:<30s}: AUROC = {v:.4f}")

    print("\nPER-CATEGORY AUROC (selected ensembles)")
    for ename, pc in per_cat_aurocs.items():
        print(f"  {ename}:")
        for cat, auroc in pc.items():
            print(f"    {cat}: {auroc:.4f}")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "ensemble_scoring",
        "experiment_number": 158,
        "timestamp": ts,
        "n_cal": n_cal, "n_test_id": n_test, "n_ood_per_cat": n_ood_per,
        "layers": layers,
        "individual_aurocs": individual_aurocs,
        "ensemble_aurocs": ensemble_aurocs,
        "per_category_aurocs": per_cat_aurocs,
        "cal_stats": {str(l): {k: float(v) for k, v in cal_stats[l].items()} for l in layers},
    }
    path = os.path.join(RESULTS_DIR, f"ensemble_scoring_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
