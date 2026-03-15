#!/usr/bin/env python3
"""Experiment 201: FPR@95TPR and threshold calibration — compute optimal thresholds,
FPR at various TPR levels, and the detection operating curve.
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

def cosine_distance(a, b):
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def compute_auroc(id_scores, ood_scores):
    id_scores = np.asarray(id_scores)
    ood_scores = np.asarray(ood_scores)
    n_id, n_ood = len(id_scores), len(ood_scores)
    if n_id == 0 or n_ood == 0: return 0.5
    count = sum(float(np.sum(o > id_scores) + 0.5 * np.sum(o == id_scores)) for o in ood_scores)
    return count / (n_id * n_ood)

def compute_roc_curve(id_scores, ood_scores, n_thresholds=200):
    """Compute ROC curve points."""
    all_scores = np.concatenate([id_scores, ood_scores])
    thresholds = np.linspace(min(all_scores), max(all_scores), n_thresholds)
    
    roc = []
    for t in thresholds:
        # OOD = positive, ID = negative
        tp = np.sum(ood_scores >= t)
        fp = np.sum(id_scores >= t)
        fn = np.sum(ood_scores < t)
        tn = np.sum(id_scores < t)
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        roc.append({"threshold": float(t), "tpr": float(tpr), "fpr": float(fpr)})
    
    return roc

def compute_fpr_at_tpr(roc, target_tpr):
    """Find FPR at a given TPR level."""
    # Find first threshold where TPR >= target_tpr
    best_fpr = 1.0
    for point in roc:
        if point["tpr"] >= target_tpr:
            best_fpr = min(best_fpr, point["fpr"])
    return best_fpr

def main():
    print("=" * 60)
    print("Experiment 201: FPR@TPR and Threshold Calibration")
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
    n_cal = 10
    n_test = 12  # more test samples for smoother ROC

    def extract_all(image):
        inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            fwd = model(**inputs, output_hidden_states=True)
        return {l: fwd.hidden_states[l][0, -1, :].float().cpu().numpy() for l in layers}

    # Calibrate
    print("\n--- Calibration ---", flush=True)
    cal_arrs = [creators[i%3](i) for i in range(n_cal)]
    cal_embs = {l: [] for l in layers}
    for arr in cal_arrs:
        h = extract_all(Image.fromarray(arr))
        for l in layers:
            cal_embs[l].append(h[l])
    centroids = {l: np.array(cal_embs[l]).mean(axis=0) for l in layers}

    # ID test
    print("--- ID test ---", flush=True)
    test_arrs = [creators[(i+n_cal)%3](i+n_cal) for i in range(n_test)]
    id_dists = {l: [] for l in layers}
    for arr in test_arrs:
        h = extract_all(Image.fromarray(arr))
        for l in layers:
            id_dists[l].append(cosine_distance(h[l], centroids[l]))

    # OOD test
    print("--- OOD test ---", flush=True)
    ood_transforms = {
        "fog_60": lambda a: apply_fog(a, 0.6),
        "night": apply_night,
        "blur": apply_blur,
        "noise": apply_noise,
    }
    ood_dists_all = {l: [] for l in layers}
    ood_dists_per = {cat: {l: [] for l in layers} for cat in ood_transforms}
    for cat, tfn in ood_transforms.items():
        for arr in test_arrs:
            h = extract_all(Image.fromarray(tfn(arr)))
            for l in layers:
                d = cosine_distance(h[l], centroids[l])
                ood_dists_all[l].append(d)
                ood_dists_per[cat][l].append(d)

    # Compute ROC and FPR@TPR
    print("\n--- Results ---", flush=True)
    results = {}
    for l in layers:
        id_scores = np.array(id_dists[l])
        ood_scores = np.array(ood_dists_all[l])
        
        auroc = compute_auroc(id_scores.tolist(), ood_scores.tolist())
        roc = compute_roc_curve(id_scores, ood_scores)
        
        fpr95 = compute_fpr_at_tpr(roc, 0.95)
        fpr99 = compute_fpr_at_tpr(roc, 0.99)
        
        # Optimal threshold (maximize TPR - FPR)
        best_j = max(roc, key=lambda x: x["tpr"] - x["fpr"])
        
        layer_results = {
            "auroc": auroc,
            "fpr_at_95tpr": fpr95,
            "fpr_at_99tpr": fpr99,
            "optimal_threshold": best_j["threshold"],
            "optimal_tpr": best_j["tpr"],
            "optimal_fpr": best_j["fpr"],
            "id_stats": {"mean": float(np.mean(id_scores)), "std": float(np.std(id_scores)),
                        "min": float(np.min(id_scores)), "max": float(np.max(id_scores))},
            "ood_stats": {"mean": float(np.mean(ood_scores)), "std": float(np.std(ood_scores)),
                        "min": float(np.min(ood_scores)), "max": float(np.max(ood_scores))},
            "roc_curve": roc[::5],  # subsample for storage
        }
        
        # Per-category
        per_cat = {}
        for cat in ood_transforms:
            cat_scores = np.array(ood_dists_per[cat][l])
            cat_auroc = compute_auroc(id_scores.tolist(), cat_scores.tolist())
            cat_roc = compute_roc_curve(id_scores, cat_scores)
            per_cat[cat] = {
                "auroc": cat_auroc,
                "fpr_at_95tpr": compute_fpr_at_tpr(cat_roc, 0.95),
            }
        layer_results["per_category"] = per_cat
        
        results[f"L{l}"] = layer_results
        print(f"  L{l}: AUROC={auroc:.4f} FPR@95={fpr95:.4f} FPR@99={fpr99:.4f} "
              f"optimal_t={best_j['threshold']:.6f}", flush=True)
        print(f"    ID: {np.mean(id_scores):.6f}±{np.std(id_scores):.6f} "
              f"range=[{np.min(id_scores):.6f}, {np.max(id_scores):.6f}]", flush=True)
        print(f"    OOD: {np.mean(ood_scores):.6f}±{np.std(ood_scores):.6f} "
              f"range=[{np.min(ood_scores):.6f}, {np.max(ood_scores):.6f}]", flush=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "threshold_fpr",
        "experiment_number": 201,
        "timestamp": ts,
        "n_cal": n_cal, "n_test": n_test,
        "layers": layers,
        "results": results,
    }
    path = os.path.join(RESULTS_DIR, f"threshold_fpr_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
