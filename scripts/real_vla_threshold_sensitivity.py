#!/usr/bin/env python3
"""Experiment 146: Threshold sensitivity analysis for OR-gate detector.

Sweeps threshold from 1σ to 6σ for L3-only, L32-only, L3∨L32 OR-gate, L3∧L32 AND-gate.
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
def apply_occlusion(a, f=0.3):
    o=a.copy(); h,w=o.shape[:2]; bh,bw=int(h*f),int(w*f)
    o[h//2-bh//2:h//2+bh//2, w//2-bw//2:w//2+bw//2]=128; return o
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

def main():
    print("=" * 60)
    print("Experiment 146: Threshold Sensitivity Analysis")
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
    n_cal, n_test_id = 12, 15

    cal_arrs = [creators[i%3](i) for i in range(n_cal)]
    test_arrs = [creators[(i+n_cal)%3](i+n_cal) for i in range(n_test_id)]

    ood_transforms = {
        "fog_30": lambda a: apply_fog(a, 0.3), "fog_60": lambda a: apply_fog(a, 0.6),
        "night": apply_night, "blur": apply_blur,
        "noise": apply_noise, "occlusion": apply_occlusion,
        "snow": apply_snow, "overexpose": apply_overexpose,
    }
    sigma_levels = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]

    # Calibration
    print(f"\n--- Calibration (n={n_cal}) ---", flush=True)
    cal_embs = {l: [] for l in layers}
    for i, arr in enumerate(cal_arrs):
        h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
        for l in layers:
            cal_embs[l].append(h[l])
        print(f"  Cal {i+1}/{n_cal}", flush=True)

    centroids, cal_d = {}, {}
    for l in layers:
        embs = np.array(cal_embs[l])
        centroids[l] = embs.mean(axis=0)
        dists = [cosine_distance(e, centroids[l]) for e in embs]
        cal_d[l] = {"mean": float(np.mean(dists)), "std": float(np.std(dists)),
                    "max": float(np.max(dists)), "values": [float(d) for d in dists]}

    # Test ID
    print(f"\n--- Test ID (n={n_test_id}) ---", flush=True)
    id_d = {l: [] for l in layers}
    for i, arr in enumerate(test_arrs):
        h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
        for l in layers:
            id_d[l].append(cosine_distance(h[l], centroids[l]))
        print(f"  ID {i+1}/{n_test_id}", flush=True)

    # OOD
    n_ood = 8
    cats = list(ood_transforms.keys())
    print(f"\n--- OOD ({len(cats)} cats × {n_ood}) ---", flush=True)
    ood_d = {l: {c: [] for c in cats} for l in layers}
    for cat, tfn in ood_transforms.items():
        for j in range(n_ood):
            arr = tfn(test_arrs[j % n_test_id])
            h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
            for l in layers:
                ood_d[l][cat].append(cosine_distance(h[l], centroids[l]))
        print(f"  {cat}: done", flush=True)

    # Sweep
    print(f"\n--- Threshold sweep ---", flush=True)
    results = {}
    for sigma in sigma_levels:
        thresholds = {l: cal_d[l]["mean"] + sigma * cal_d[l]["std"] for l in layers}
        strats = {
            "L3_only": lambda d3, d32, t=thresholds: d3 > t[3],
            "L32_only": lambda d3, d32, t=thresholds: d32 > t[32],
            "OR_gate": lambda d3, d32, t=thresholds: d3 > t[3] or d32 > t[32],
            "AND_gate": lambda d3, d32, t=thresholds: d3 > t[3] and d32 > t[32],
        }

        sr = {"thresholds": {f"L{l}": float(thresholds[l]) for l in layers}}
        for sn, sf in strats.items():
            id_f = sum(1 for i in range(n_test_id) if sf(id_d[3][i], id_d[32][i]))
            fpr = id_f / n_test_id
            pc = {}; tot_ood = 0; tot_flag = 0
            for c in cats:
                fl = sum(1 for j in range(n_ood) if sf(ood_d[3][c][j], ood_d[32][c][j]))
                pc[c] = {"recall": fl/n_ood, "n_flagged": fl, "n_total": n_ood}
                tot_ood += n_ood; tot_flag += fl
            rec = tot_flag / tot_ood
            prec = tot_flag / (tot_flag + id_f) if (tot_flag + id_f) > 0 else 1.0
            f1 = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0
            sr[sn] = {"fpr": float(fpr), "recall": float(rec), "precision": float(prec),
                       "f1": float(f1), "per_category": pc}
        results[f"sigma_{sigma}"] = sr
        print(f"  σ={sigma}: OR recall={sr['OR_gate']['recall']:.3f} FPR={sr['OR_gate']['fpr']:.3f}", flush=True)

    # Summary
    print("\n" + "=" * 80)
    print("OR-Gate Performance Across Thresholds")
    print(f"{'σ':>5s} {'Recall':>8s} {'Prec':>8s} {'F1':>8s} {'FPR':>8s}")
    for s in sigma_levels:
        r = results[f"sigma_{s}"]["OR_gate"]
        print(f"{s:5.1f} {r['recall']:8.3f} {r['precision']:8.3f} {r['f1']:8.3f} {r['fpr']:8.3f}")

    print("\nStrategies at σ=3.0:")
    for sn in ["L3_only","L32_only","OR_gate","AND_gate"]:
        r = results["sigma_3.0"][sn]
        print(f"  {sn:12s}: Recall={r['recall']:.3f} Prec={r['precision']:.3f} F1={r['f1']:.3f} FPR={r['fpr']:.3f}")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {"experiment": "threshold_sensitivity", "experiment_number": 146, "timestamp": ts,
           "n_cal": n_cal, "n_test_id": n_test_id, "n_ood_per_cat": n_ood,
           "ood_categories": cats, "sigma_levels": sigma_levels,
           "cal_distances": {f"L{l}": {"mean": cal_d[l]["mean"], "std": cal_d[l]["std"]} for l in layers},
           "results": results}
    path = os.path.join(RESULTS_DIR, f"threshold_sensitivity_{ts}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
