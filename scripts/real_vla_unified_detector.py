#!/usr/bin/env python3
"""Experiment 151: Unified PCA-Recon OR-Gate with Multi-Centroid Routing.

Combines the best findings: PCA reconstruction (k=2) at both L3 and L32,
multi-centroid routing across prompts, in a single unified detector.
Tests this "best-of-all" architecture against all baselines.
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

def main():
    print("=" * 60)
    print("Experiment 151: Unified PCA-Recon OR-Gate Detector")
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
    }

    creators = [create_highway, create_urban, create_rural]
    layers = [3, 32]
    n_cal = 8
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
    n_ood_per = 5

    # Step 1: Build calibration models per prompt
    print("\n--- Calibrating all prompts ---", flush=True)
    prompt_models = {}
    for pname, prompt in prompts.items():
        embs = {l: [] for l in layers}
        for arr in cal_arrs:
            h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
            for l in layers:
                embs[l].append(h[l])

        pm = {}
        for l in layers:
            mat = np.array(embs[l])
            centroid = mat.mean(axis=0)
            centered = mat - centroid
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            # Cosine stats
            cos_dists = [cosine_distance(e, centroid) for e in mat]
            # Recon stats (k=2)
            Vk = Vt[:2]
            recon_errs = [float(np.sum((c - c @ Vk.T @ Vk)**2)) for c in centered]
            pm[l] = {
                "centroid": centroid, "Vt2": Vt[:2],
                "cos_mean": float(np.mean(cos_dists)), "cos_std": float(np.std(cos_dists)),
                "recon_mean": float(np.mean(recon_errs)), "recon_std": float(np.std(recon_errs)),
            }
        prompt_models[pname] = pm
        print(f"  {pname}: calibrated", flush=True)

    sigma = 3.0

    # Define detectors
    def cosine_orgate(emb_l3, emb_l32, pname):
        pm = prompt_models[pname]
        d3 = cosine_distance(emb_l3, pm[3]["centroid"])
        d32 = cosine_distance(emb_l32, pm[32]["centroid"])
        t3 = pm[3]["cos_mean"] + sigma * pm[3]["cos_std"]
        t32 = pm[32]["cos_mean"] + sigma * pm[32]["cos_std"]
        return d3 > t3 or d32 > t32

    def recon_orgate(emb_l3, emb_l32, pname):
        pm = prompt_models[pname]
        for l, emb in [(3, emb_l3), (32, emb_l32)]:
            diff = emb - pm[l]["centroid"]
            Vk = pm[l]["Vt2"]
            err = float(np.sum((diff - diff @ Vk.T @ Vk)**2))
            t = pm[l]["recon_mean"] + sigma * pm[l]["recon_std"]
            if err > t:
                return True
        return False

    def nearest_cosine(emb_l3, emb_l32):
        # Route via L32 nearest centroid
        best = min(prompts.keys(), key=lambda p: cosine_distance(emb_l32, prompt_models[p][32]["centroid"]))
        return cosine_orgate(emb_l3, emb_l32, best)

    def nearest_recon(emb_l3, emb_l32):
        # Route via L32 nearest centroid
        best = min(prompts.keys(), key=lambda p: cosine_distance(emb_l32, prompt_models[p][32]["centroid"]))
        return recon_orgate(emb_l3, emb_l32, best)

    # Step 2: Evaluate
    print("\n--- Evaluating ---", flush=True)
    results = {}

    for inf_pname, inf_prompt in prompts.items():
        print(f"\n  Prompt: {inf_pname}", flush=True)

        # Collect embeddings
        id_embs = [(None, None)] * n_test
        for i, arr in enumerate(test_arrs):
            h = extract_hidden(model, processor, Image.fromarray(arr), inf_prompt, layers)
            id_embs[i] = (h[3], h[32])

        ood_embs = {c: [] for c in ood_transforms}
        for cat, tfn in ood_transforms.items():
            for j in range(n_ood_per):
                arr = tfn(test_arrs[j % n_test])
                h = extract_hidden(model, processor, Image.fromarray(arr), inf_prompt, layers)
                ood_embs[cat].append((h[3], h[32]))

        # Evaluate all detectors
        detectors = {
            "cosine_same": lambda e3, e32: cosine_orgate(e3, e32, inf_pname),
            "recon_same": lambda e3, e32: recon_orgate(e3, e32, inf_pname),
            "cosine_nearest": lambda e3, e32: nearest_cosine(e3, e32),
            "recon_nearest": lambda e3, e32: nearest_recon(e3, e32),
        }

        prompt_res = {}
        for det_name, det_fn in detectors.items():
            id_flagged = sum(1 for e3, e32 in id_embs if det_fn(e3, e32))
            ood_flagged = 0; ood_total = 0
            per_cat = {}
            for cat in ood_transforms:
                fl = sum(1 for e3, e32 in ood_embs[cat] if det_fn(e3, e32))
                per_cat[cat] = fl / n_ood_per
                ood_flagged += fl; ood_total += n_ood_per
            fpr = id_flagged / n_test
            rec = ood_flagged / ood_total
            prec = ood_flagged / (ood_flagged + id_flagged) if (ood_flagged + id_flagged) > 0 else 1.0
            f1 = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0
            prompt_res[det_name] = {
                "fpr": float(fpr), "recall": float(rec), "precision": float(prec),
                "f1": float(f1), "per_category_recall": per_cat,
            }
            print(f"    {det_name:20s}: F1={f1:.3f} Rec={rec:.3f} FPR={fpr:.3f}", flush=True)

        results[inf_pname] = prompt_res

    # Summary
    print("\n" + "=" * 80)
    print("DETECTOR COMPARISON (mean across prompts)")
    for dn in ["cosine_same", "recon_same", "cosine_nearest", "recon_nearest"]:
        f1s = [results[p][dn]["f1"] for p in prompts]
        fprs = [results[p][dn]["fpr"] for p in prompts]
        recs = [results[p][dn]["recall"] for p in prompts]
        print(f"  {dn:22s}: F1={np.mean(f1s):.3f}  Recall={np.mean(recs):.3f}  FPR={np.mean(fprs):.3f}")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {"experiment": "unified_detector", "experiment_number": 151, "timestamp": ts,
              "n_cal": n_cal, "n_test_id": n_test, "n_ood_per_cat": n_ood_per,
              "ood_categories": list(ood_transforms.keys()),
              "prompts": list(prompts.keys()), "sigma": sigma,
              "layers": layers, "k_pca": 2, "results": results}
    path = os.path.join(RESULTS_DIR, f"unified_detector_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
