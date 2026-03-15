#!/usr/bin/env python3
"""Experiment 149: Multi-centroid prompt-aware detector.

Calibrates with N prompts simultaneously. At inference, computes distance to
the nearest centroid (matching the inference prompt). Tests if a prompt-aware
multi-centroid approach enables cross-prompt generalization.
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

def main():
    print("=" * 60)
    print("Experiment 149: Multi-Centroid Prompt-Aware Detector")
    print("=" * 60, flush=True)

    from transformers import AutoModelForVision2Seq, AutoProcessor
    print("Loading OpenVLA-7B...", flush=True)
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompts = {
        "drive_forward": "In: What action should the robot take to drive forward at 25 m/s safely?\nOut:",
        "navigate": "In: What action should the robot take to navigate this road?\nOut:",
        "follow_lane": "In: What action should the robot take to follow the lane markings?\nOut:",
        "stop": "In: What action should the robot take to stop the vehicle?\nOut:",
        "turn_left": "In: What action should the robot take to turn left at the intersection?\nOut:",
    }

    creators = [create_highway, create_urban, create_rural]
    layers = [3, 32]
    n_cal = 6
    n_test = 6

    cal_arrs = [creators[i%3](i) for i in range(n_cal)]
    test_arrs = [creators[(i+n_cal)%3](i+n_cal) for i in range(n_test)]

    ood_transforms = {
        "fog_30": lambda a: apply_fog(a, 0.3),
        "fog_60": lambda a: apply_fog(a, 0.6),
        "night": apply_night,
        "blur": apply_blur,
        "noise": apply_noise,
        "occlusion": apply_occlusion,
    }
    n_ood_per = 4

    # Step 1: Calibrate ALL prompts
    print("\n--- Calibrating all prompts ---", flush=True)
    centroids = {}  # {prompt_name: {layer: centroid}}
    cal_stats = {}  # {prompt_name: {layer: {mean, std}}}
    for pname, prompt in prompts.items():
        centroids[pname] = {}
        cal_stats[pname] = {}
        for l in layers:
            embs = []
            for arr in cal_arrs:
                h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
                embs.append(h[l])
            embs = np.array(embs)
            centroid = embs.mean(axis=0)
            dists = [cosine_distance(e, centroid) for e in embs]
            centroids[pname][l] = centroid
            cal_stats[pname][l] = {"mean": float(np.mean(dists)), "std": float(np.std(dists))}
        print(f"  {pname}: calibrated", flush=True)

    # Step 2: Define detection strategies
    sigma = 3.0

    def single_centroid_detect(d_l3, d_l32, cal_pname):
        """Standard: check against the calibration prompt's centroid only."""
        t3 = cal_stats[cal_pname][3]["mean"] + sigma * cal_stats[cal_pname][3]["std"]
        t32 = cal_stats[cal_pname][32]["mean"] + sigma * cal_stats[cal_pname][32]["std"]
        return d_l3 > t3 or d_l32 > t32

    def nearest_centroid_detect(emb_l3, emb_l32):
        """Multi-centroid: find nearest prompt centroid, then check distance."""
        best_dist = float('inf')
        best_pname = None
        for pname in prompts:
            d = cosine_distance(emb_l32, centroids[pname][32])
            if d < best_dist:
                best_dist = d
                best_pname = pname
        # Check against nearest centroid's thresholds
        d3 = cosine_distance(emb_l3, centroids[best_pname][3])
        d32 = best_dist
        t3 = cal_stats[best_pname][3]["mean"] + sigma * cal_stats[best_pname][3]["std"]
        t32 = cal_stats[best_pname][32]["mean"] + sigma * cal_stats[best_pname][32]["std"]
        return d3 > t3 or d32 > t32

    def min_distance_detect(emb_l3, emb_l32):
        """Min-distance: take minimum distance across all centroids, check against pooled threshold."""
        min_d3 = min(cosine_distance(emb_l3, centroids[p][3]) for p in prompts)
        min_d32 = min(cosine_distance(emb_l32, centroids[p][32]) for p in prompts)
        # Pooled threshold: mean of all per-prompt thresholds
        pool_t3 = np.mean([cal_stats[p][3]["mean"] + sigma * cal_stats[p][3]["std"] for p in prompts])
        pool_t32 = np.mean([cal_stats[p][32]["mean"] + sigma * cal_stats[p][32]["std"] for p in prompts])
        return min_d3 > pool_t3 or min_d32 > pool_t32

    # Step 3: Evaluate
    print("\n--- Evaluating ---", flush=True)
    results = {}

    for inf_pname, inf_prompt in prompts.items():
        print(f"\n  Inference prompt: {inf_pname}", flush=True)

        # Collect embeddings for test ID and OOD
        id_embs = {l: [] for l in layers}
        for arr in test_arrs:
            h = extract_hidden(model, processor, Image.fromarray(arr), inf_prompt, layers)
            for l in layers:
                id_embs[l].append(h[l])

        ood_embs = {l: {c: [] for c in ood_transforms} for l in layers}
        for cat, tfn in ood_transforms.items():
            for j in range(n_ood_per):
                arr = tfn(test_arrs[j % n_test])
                h = extract_hidden(model, processor, Image.fromarray(arr), inf_prompt, layers)
                for l in layers:
                    ood_embs[l][cat].append(h[l])

        # Evaluate each strategy
        strat_results = {}

        # 1. Same-prompt single centroid (oracle)
        id_flags = sum(1 for i in range(n_test)
                      if single_centroid_detect(
                          cosine_distance(id_embs[3][i], centroids[inf_pname][3]),
                          cosine_distance(id_embs[32][i], centroids[inf_pname][32]),
                          inf_pname))
        ood_flags = 0; ood_total = 0
        for cat in ood_transforms:
            for j in range(n_ood_per):
                if single_centroid_detect(
                    cosine_distance(ood_embs[3][cat][j], centroids[inf_pname][3]),
                    cosine_distance(ood_embs[32][cat][j], centroids[inf_pname][32]),
                    inf_pname):
                    ood_flags += 1
                ood_total += 1
        fpr = id_flags / n_test
        rec = ood_flags / ood_total
        prec = ood_flags / (ood_flags + id_flags) if (ood_flags + id_flags) > 0 else 1.0
        f1 = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0
        strat_results["same_prompt"] = {"fpr": float(fpr), "recall": float(rec), "precision": float(prec), "f1": float(f1)}

        # 2. Wrong-prompt single centroid (worst case: use first OTHER prompt)
        wrong_pname = [p for p in prompts if p != inf_pname][0]
        id_flags = sum(1 for i in range(n_test)
                      if single_centroid_detect(
                          cosine_distance(id_embs[3][i], centroids[wrong_pname][3]),
                          cosine_distance(id_embs[32][i], centroids[wrong_pname][32]),
                          wrong_pname))
        ood_flags = 0; ood_total = 0
        for cat in ood_transforms:
            for j in range(n_ood_per):
                if single_centroid_detect(
                    cosine_distance(ood_embs[3][cat][j], centroids[wrong_pname][3]),
                    cosine_distance(ood_embs[32][cat][j], centroids[wrong_pname][32]),
                    wrong_pname):
                    ood_flags += 1
                ood_total += 1
        fpr = id_flags / n_test
        rec = ood_flags / ood_total
        prec = ood_flags / (ood_flags + id_flags) if (ood_flags + id_flags) > 0 else 1.0
        f1 = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0
        strat_results["wrong_prompt"] = {"fpr": float(fpr), "recall": float(rec), "precision": float(prec), "f1": float(f1)}

        # 3. Nearest centroid
        id_flags = sum(1 for i in range(n_test)
                      if nearest_centroid_detect(id_embs[3][i], id_embs[32][i]))
        ood_flags = 0; ood_total = 0
        for cat in ood_transforms:
            for j in range(n_ood_per):
                if nearest_centroid_detect(ood_embs[3][cat][j], ood_embs[32][cat][j]):
                    ood_flags += 1
                ood_total += 1
        fpr = id_flags / n_test
        rec = ood_flags / ood_total
        prec = ood_flags / (ood_flags + id_flags) if (ood_flags + id_flags) > 0 else 1.0
        f1 = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0
        strat_results["nearest_centroid"] = {"fpr": float(fpr), "recall": float(rec), "precision": float(prec), "f1": float(f1)}

        # 4. Min distance
        id_flags = sum(1 for i in range(n_test)
                      if min_distance_detect(id_embs[3][i], id_embs[32][i]))
        ood_flags = 0; ood_total = 0
        for cat in ood_transforms:
            for j in range(n_ood_per):
                if min_distance_detect(ood_embs[3][cat][j], ood_embs[32][cat][j]):
                    ood_flags += 1
                ood_total += 1
        fpr = id_flags / n_test
        rec = ood_flags / ood_total
        prec = ood_flags / (ood_flags + id_flags) if (ood_flags + id_flags) > 0 else 1.0
        f1 = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0
        strat_results["min_distance"] = {"fpr": float(fpr), "recall": float(rec), "precision": float(prec), "f1": float(f1)}

        results[inf_pname] = strat_results
        for sn, sr in strat_results.items():
            print(f"    {sn:20s}: F1={sr['f1']:.3f} Recall={sr['recall']:.3f} FPR={sr['fpr']:.3f}", flush=True)

    # Summary
    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON (mean across prompts)")
    for sn in ["same_prompt", "wrong_prompt", "nearest_centroid", "min_distance"]:
        f1s = [results[p][sn]["f1"] for p in prompts]
        fprs = [results[p][sn]["fpr"] for p in prompts]
        recs = [results[p][sn]["recall"] for p in prompts]
        print(f"  {sn:20s}: F1={np.mean(f1s):.3f}±{np.std(f1s):.3f}  Recall={np.mean(recs):.3f}  FPR={np.mean(fprs):.3f}")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {"experiment": "multi_centroid", "experiment_number": 149, "timestamp": ts,
              "n_cal": n_cal, "n_test_id": n_test, "n_ood_per_cat": n_ood_per,
              "ood_categories": list(ood_transforms.keys()),
              "prompts": list(prompts.keys()), "sigma": sigma,
              "layers": layers, "results": results}
    path = os.path.join(RESULTS_DIR, f"multi_centroid_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
