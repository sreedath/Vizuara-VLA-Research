#!/usr/bin/env python3
"""Experiment 184: Prompt sensitivity — does the text prompt affect OOD detection?

Tests whether different action prompts produce different OOD detection
performance, or if the visual OOD signal dominates prompt choice.
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
    print("Experiment 184: Prompt Sensitivity Analysis")
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
        "stop": "In: What action should the robot take to stop the vehicle?\nOut:",
        "turn_left": "In: What action should the robot take to turn left?\nOut:",
        "generic_robot": "In: What action should the robot take to complete the task?\nOut:",
        "minimal": "In: What action should the robot take?\nOut:",
    }

    layers = [3, 32]
    creators = [create_highway, create_urban, create_rural]
    n_cal = 6
    n_test = 4

    cal_arrs = [creators[i%3](i) for i in range(n_cal)]
    test_arrs = [creators[(i+n_cal)%3](i+n_cal) for i in range(n_test)]

    ood_transforms = {
        "fog_60": lambda a: apply_fog(a, 0.6),
        "night": apply_night,
        "blur": apply_blur,
        "noise": apply_noise,
    }

    results = {}
    for prompt_name, prompt in prompts.items():
        print(f"\n--- Prompt: {prompt_name} ---", flush=True)

        # Calibrate
        centroids = {}
        for l in layers:
            cal_embs = []
            for arr in cal_arrs:
                h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
                cal_embs.append(h[l])
            centroids[l] = np.array(cal_embs).mean(axis=0)

        # ID test
        id_embs = {l: [] for l in layers}
        for arr in test_arrs:
            h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
            for l in layers:
                id_embs[l].append(h[l])

        # OOD test
        ood_embs = {l: [] for l in layers}
        for cat, tfn in ood_transforms.items():
            for arr in test_arrs:
                h = extract_hidden(model, processor, Image.fromarray(tfn(arr)), prompt, layers)
                for l in layers:
                    ood_embs[l].append(h[l])

        prompt_results = {}
        for l in layers:
            id_dists = [cosine_distance(e, centroids[l]) for e in id_embs[l]]
            ood_dists = [cosine_distance(e, centroids[l]) for e in ood_embs[l]]
            auroc = compute_auroc(id_dists, ood_dists)

            prompt_results[f"L{l}"] = {
                "auroc": auroc,
                "id_mean_dist": float(np.mean(id_dists)),
                "ood_mean_dist": float(np.mean(ood_dists)),
                "separation_ratio": float(np.mean(ood_dists) / (np.mean(id_dists) + 1e-10)),
            }
            print(f"  L{l}: AUROC={auroc:.4f} sep_ratio={prompt_results[f'L{l}']['separation_ratio']:.2f}", flush=True)

        results[prompt_name] = prompt_results

    # Cross-prompt centroid comparison: do different prompts produce different centroids?
    print("\n--- Cross-prompt centroid comparison ---", flush=True)
    prompt_centroids = {}
    ref_prompt = list(prompts.keys())[0]
    ref_text = prompts[ref_prompt]
    ref_centroids = {}
    for l in layers:
        ref_embs = []
        for arr in cal_arrs:
            h = extract_hidden(model, processor, Image.fromarray(arr), ref_text, layers)
            ref_embs.append(h[l])
        ref_centroids[l] = np.array(ref_embs).mean(axis=0)

    cross_prompt = {}
    for prompt_name, prompt in prompts.items():
        if prompt_name == ref_prompt:
            continue
        cp_centroids = {}
        for l in layers:
            cp_embs = []
            for arr in cal_arrs:
                h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
                cp_embs.append(h[l])
            cp_centroids[l] = np.array(cp_embs).mean(axis=0)

        cross_prompt[prompt_name] = {
            f"L{l}": cosine_distance(ref_centroids[l], cp_centroids[l]) for l in layers
        }
        print(f"  {ref_prompt} vs {prompt_name}: L3={cross_prompt[prompt_name]['L3']:.6f} L32={cross_prompt[prompt_name]['L32']:.6f}", flush=True)

    results["cross_prompt_centroid_distances"] = cross_prompt

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "prompt_sensitivity",
        "experiment_number": 184,
        "timestamp": ts,
        "prompts": prompts,
        "n_cal": n_cal, "n_test": n_test,
        "layers": layers,
        "results": results,
    }
    path = os.path.join(RESULTS_DIR, f"prompt_sensitivity_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
