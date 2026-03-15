#!/usr/bin/env python3
"""Experiment 206: Image resolution sensitivity — does OOD detection work
when images are captured at different resolutions? Tests 64x64 to 512x512.
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

rng = np.random.RandomState(42)

def create_scene(scene_type, size, idx):
    """Create a scene at arbitrary resolution."""
    h, w = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    if scene_type == "highway":
        img[:h//2] = [135, 206, 235]; img[h//2:] = [80, 80, 80]
        img[h//2:, w//2-max(1,w//85):w//2+max(1,w//85)] = [255, 255, 255]
    elif scene_type == "urban":
        img[:h//3] = [135, 206, 235]; img[h//3:h//2] = [139, 119, 101]; img[h//2:] = [60, 60, 60]
    else:
        img[:h//3] = [100, 180, 255]; img[h//3:h*2//3] = [34, 139, 34]; img[h*2//3:] = [90, 90, 90]
    return np.clip(img.astype(np.int16) + rng.randint(-5, 6, img.shape).astype(np.int16), 0, 255).astype(np.uint8)

def apply_fog(a, alpha):
    return np.clip(a*(1-alpha)+np.full_like(a,[200,200,210])*alpha, 0, 255).astype(np.uint8)
def apply_night(a): return np.clip(a*0.15, 0, 255).astype(np.uint8)
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

def main():
    print("=" * 60)
    print("Experiment 206: Resolution Sensitivity")
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

    resolutions = [(64, 64), (128, 128), (224, 224), (256, 256), (384, 384), (512, 512)]
    scenes = ["highway", "urban", "rural"]
    n_cal = 6
    n_test = 6

    def extract_all(image):
        inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            fwd = model(**inputs, output_hidden_states=True)
        return {l: fwd.hidden_states[l][0, -1, :].float().cpu().numpy() for l in layers}

    results = {}
    for res in resolutions:
        print(f"\n--- Resolution {res[0]}x{res[1]} ---", flush=True)
        
        # Calibrate at this resolution
        cal_embs = {l: [] for l in layers}
        for i in range(n_cal):
            arr = create_scene(scenes[i%3], res, i)
            h = extract_all(Image.fromarray(arr))
            for l in layers:
                cal_embs[l].append(h[l])
        centroids = {l: np.array(cal_embs[l]).mean(axis=0) for l in layers}

        # ID test
        id_dists = {l: [] for l in layers}
        for i in range(n_test):
            arr = create_scene(scenes[(i+n_cal)%3], res, i+n_cal)
            h = extract_all(Image.fromarray(arr))
            for l in layers:
                id_dists[l].append(cosine_distance(h[l], centroids[l]))

        # OOD test
        ood_transforms = {
            "fog": lambda a: apply_fog(a, 0.6),
            "night": apply_night,
            "noise": lambda a: apply_noise(a, 50),
        }
        res_results = {}
        for cat, tfn in ood_transforms.items():
            ood_dists = {l: [] for l in layers}
            for i in range(n_test):
                arr = create_scene(scenes[(i+n_cal)%3], res, i+n_cal)
                h = extract_all(Image.fromarray(tfn(arr)))
                for l in layers:
                    ood_dists[l].append(cosine_distance(h[l], centroids[l]))

            cat_results = {}
            for l in layers:
                auroc = compute_auroc(id_dists[l], ood_dists[l])
                sep = float(np.mean(ood_dists[l]) / (np.mean(id_dists[l]) + 1e-10))
                cat_results[f"L{l}"] = {"auroc": auroc, "separation": sep}
            res_results[cat] = cat_results

        # Overall AUROC (all corruptions combined)
        overall = {}
        for l in layers:
            all_ood = []
            for cat in ood_transforms:
                for i in range(n_test):
                    arr = create_scene(scenes[(i+n_cal)%3], res, i+n_cal)
                    h = extract_all(Image.fromarray(ood_transforms[cat](arr)))
                    all_ood.append(cosine_distance(h[l], centroids[l]))
            overall[f"L{l}"] = compute_auroc(id_dists[l], all_ood)

        res_results["overall"] = overall
        results[f"{res[0]}x{res[1]}"] = res_results
        
        print(f"  Overall: L1={overall['L1']:.3f} L3={overall['L3']:.3f} L32={overall['L32']:.3f}", flush=True)
        for cat in ood_transforms:
            print(f"  {cat}: L1={res_results[cat]['L1']['auroc']:.3f} L3={res_results[cat]['L3']['auroc']:.3f}", flush=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "resolution_sensitivity",
        "experiment_number": 206,
        "timestamp": ts,
        "resolutions": [list(r) for r in resolutions],
        "results": results,
    }
    path = os.path.join(RESULTS_DIR, f"resolution_sensitivity_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
