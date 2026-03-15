#!/usr/bin/env python3
"""Experiment 148: Cross-prompt OR-gate validation.

Tests whether the OR-gate detector (calibrated with one prompt) generalizes
when a DIFFERENT prompt is used at inference time.
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
    print("Experiment 148: Cross-Prompt OR-Gate Validation")
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
    n_cal = 8
    n_test = 8

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

    results = {}

    # For each calibration prompt
    for cal_prompt_name, cal_prompt in prompts.items():
        print(f"\n=== Calibrating with: {cal_prompt_name} ===", flush=True)

        # Calibrate
        centroids = {}
        cal_stats = {}
        for l in layers:
            cal_embs = []
            for i, arr in enumerate(cal_arrs):
                h = extract_hidden(model, processor, Image.fromarray(arr), cal_prompt, layers)
                cal_embs.append(h[l])
            embs = np.array(cal_embs)
            centroid = embs.mean(axis=0)
            dists = [cosine_distance(e, centroid) for e in embs]
            centroids[l] = centroid
            cal_stats[l] = {"mean": float(np.mean(dists)), "std": float(np.std(dists))}
        print(f"  Calibrated: L3 mean={cal_stats[3]['mean']:.6f}, L32 mean={cal_stats[32]['mean']:.6f}", flush=True)

        # Test with each inference prompt
        prompt_results = {}
        for inf_prompt_name, inf_prompt in prompts.items():
            # ID test
            id_dists = {l: [] for l in layers}
            for i, arr in enumerate(test_arrs):
                h = extract_hidden(model, processor, Image.fromarray(arr), inf_prompt, layers)
                for l in layers:
                    id_dists[l].append(cosine_distance(h[l], centroids[l]))

            # OOD test
            ood_dists = {l: {c: [] for c in ood_transforms} for l in layers}
            for cat, tfn in ood_transforms.items():
                for j in range(n_ood_per):
                    arr = tfn(test_arrs[j % n_test])
                    h = extract_hidden(model, processor, Image.fromarray(arr), inf_prompt, layers)
                    for l in layers:
                        ood_dists[l][cat].append(cosine_distance(h[l], centroids[l]))

            # Evaluate OR-gate at 3σ
            sigma = 3.0
            thresholds = {l: cal_stats[l]["mean"] + sigma * cal_stats[l]["std"] for l in layers}

            # FPR
            id_flagged = sum(1 for i in range(n_test)
                           if id_dists[3][i] > thresholds[3] or id_dists[32][i] > thresholds[32])
            fpr = id_flagged / n_test

            # Per-category recall
            per_cat = {}
            total_flag = 0; total_ood = 0
            for cat in ood_transforms:
                fl = sum(1 for j in range(n_ood_per)
                        if ood_dists[3][cat][j] > thresholds[3] or ood_dists[32][cat][j] > thresholds[32])
                per_cat[cat] = fl / n_ood_per
                total_flag += fl; total_ood += n_ood_per

            recall = total_flag / total_ood
            prec = total_flag / (total_flag + id_flagged) if (total_flag + id_flagged) > 0 else 1.0
            f1 = 2*prec*recall / (prec+recall) if (prec+recall) > 0 else 0.0

            same_prompt = "SAME" if cal_prompt_name == inf_prompt_name else "CROSS"
            prompt_results[inf_prompt_name] = {
                "type": same_prompt,
                "fpr": float(fpr),
                "recall": float(recall),
                "precision": float(prec),
                "f1": float(f1),
                "per_category_recall": {c: float(v) for c, v in per_cat.items()},
            }
            print(f"  → {inf_prompt_name} [{same_prompt}]: recall={recall:.3f} FPR={fpr:.3f} F1={f1:.3f}", flush=True)

        results[cal_prompt_name] = prompt_results

    # Summary
    print("\n" + "=" * 80)
    print("CROSS-PROMPT TRANSFER MATRIX (F1 scores)")
    header = "Cal\\Inf"
    print(f"{header:>15s}", end="")
    for p in prompts:
        print(f" {p[:8]:>10s}", end="")
    print()
    for cal_p in prompts:
        print(f"{cal_p[:15]:>15s}", end="")
        for inf_p in prompts:
            f1 = results[cal_p][inf_p]["f1"]
            marker = "*" if cal_p == inf_p else " "
            print(f" {f1:9.3f}{marker}", end="")
        print()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {"experiment": "cross_prompt_orgate", "experiment_number": 148, "timestamp": ts,
              "n_cal": n_cal, "n_test_id": n_test, "n_ood_per_cat": n_ood_per,
              "ood_categories": list(ood_transforms.keys()),
              "prompts": list(prompts.keys()),
              "sigma": 3.0, "layers": layers, "results": results}
    path = os.path.join(RESULTS_DIR, f"cross_prompt_orgate_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
