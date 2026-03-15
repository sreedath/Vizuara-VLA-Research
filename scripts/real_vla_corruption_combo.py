#!/usr/bin/env python3
"""Experiment 164: Corruption combination — interaction effects.

Tests how simultaneous corruptions (e.g., fog+blur, night+noise) interact.
Are combined corruptions harder to detect than individual ones?
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
    print("Experiment 164: Corruption Combination Interactions")
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

    # Calibrate
    cal_arrs = [creators[i%3](i) for i in range(n_cal)]
    centroids = {}
    for l in layers:
        cal_embs = []
        for arr in cal_arrs:
            h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
            cal_embs.append(h[l])
        centroids[l] = np.array(cal_embs).mean(axis=0)

    test_arrs = [creators[(i+n_cal)%3](i+n_cal) for i in range(n_test)]

    # Define single and combination corruptions
    conditions = {
        # Singles
        "fog_30": lambda a: apply_fog(a, 0.3),
        "fog_60": lambda a: apply_fog(a, 0.6),
        "night": apply_night,
        "blur": apply_blur,
        "noise": lambda a: apply_noise(a, 50),
        "occlusion": apply_occlusion,
        # Combinations
        "fog_30+blur": lambda a: apply_blur(apply_fog(a, 0.3)),
        "fog_30+noise": lambda a: apply_noise(apply_fog(a, 0.3), 50),
        "fog_60+blur": lambda a: apply_blur(apply_fog(a, 0.6)),
        "fog_60+noise": lambda a: apply_noise(apply_fog(a, 0.6), 50),
        "night+noise": lambda a: apply_noise(apply_night(a), 50),
        "night+blur": lambda a: apply_blur(apply_night(a)),
        "blur+noise": lambda a: apply_noise(apply_blur(a), 50),
        "fog_30+occlusion": lambda a: apply_occlusion(apply_fog(a, 0.3)),
        "night+occlusion": lambda a: apply_occlusion(apply_night(a)),
        # Triple
        "fog_30+blur+noise": lambda a: apply_noise(apply_blur(apply_fog(a, 0.3)), 50),
        "night+blur+noise": lambda a: apply_noise(apply_blur(apply_night(a)), 50),
    }

    results = {}
    for cname, cfn in conditions.items():
        print(f"\n--- {cname} ---", flush=True)
        dists = {l: [] for l in layers}
        for arr in test_arrs:
            corr = cfn(arr)
            h = extract_hidden(model, processor, Image.fromarray(corr), prompt, layers)
            for l in layers:
                dists[l].append(cosine_distance(h[l], centroids[l]))

        entry = {}
        for l in layers:
            entry[f"L{l}_mean"] = float(np.mean(dists[l]))
            entry[f"L{l}_std"] = float(np.std(dists[l]))
        
        n_components = cname.count("+") + 1
        entry["n_components"] = n_components
        entry["components"] = cname.split("+")
        results[cname] = entry
        print(f"  L3={entry['L3_mean']:.6f} L32={entry['L32_mean']:.4f} (n_comp={n_components})", flush=True)

    # Analyze interaction effects
    print("\n" + "=" * 80)
    print("INTERACTION ANALYSIS")
    combos = [(c, r) for c, r in results.items() if r["n_components"] >= 2]
    for cname, entry in combos:
        components = entry["components"]
        # Sum of individual distances
        sum_l3 = sum(results[c]["L3_mean"] for c in components if c in results)
        sum_l32 = sum(results[c]["L32_mean"] for c in components if c in results)
        combo_l3 = entry["L3_mean"]
        combo_l32 = entry["L32_mean"]
        # Interaction = combo - sum (superadditive if > 0, subadditive if < 0)
        int_l3 = combo_l3 - sum_l3
        int_l32 = combo_l32 - sum_l32
        ratio_l3 = combo_l3 / (sum_l3 + 1e-10)
        ratio_l32 = combo_l32 / (sum_l32 + 1e-10)
        entry["interaction_L3"] = int_l3
        entry["interaction_L32"] = int_l32
        entry["ratio_L3"] = ratio_l3
        entry["ratio_L32"] = ratio_l32
        print(f"  {cname}: L3 ratio={ratio_l3:.3f} L32 ratio={ratio_l32:.3f} "
              f"({'super' if ratio_l3 > 1 else 'sub'}additive)", flush=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "corruption_combination",
        "experiment_number": 164,
        "timestamp": ts,
        "n_cal": n_cal, "n_test": n_test,
        "layers": layers,
        "conditions": list(conditions.keys()),
        "results": results,
    }
    path = os.path.join(RESULTS_DIR, f"corruption_combo_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
