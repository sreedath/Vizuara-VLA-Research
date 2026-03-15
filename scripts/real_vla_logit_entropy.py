#!/usr/bin/env python3
"""Experiment 190: Logit-based OOD detection — compare hidden-state cosine
distance with output logit entropy and max-logit methods.

Tests whether traditional OOD detection metrics (logit entropy, max softmax
probability) work for VLA models, and how they compare to our cosine approach.
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

def extract_with_logits(model, processor, image, prompt, layers):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)

    # Hidden states
    hidden = {l: fwd.hidden_states[l][0, -1, :].float().cpu().numpy() for l in layers}

    # Logits from last token
    logits = fwd.logits[0, -1, :].float().cpu().numpy()

    # Action token logits (31744-32000)
    action_logits = logits[31744:32000]

    # Softmax probabilities
    action_probs = np.exp(action_logits - np.max(action_logits))
    action_probs = action_probs / action_probs.sum()

    # Entropy of action distribution
    entropy = -np.sum(action_probs * np.log(action_probs + 1e-10))

    # Max probability
    max_prob = float(np.max(action_probs))

    # Max logit
    max_logit = float(np.max(action_logits))

    # Energy score: -log(sum(exp(logits)))
    energy = -float(np.log(np.sum(np.exp(action_logits - np.max(action_logits)))) + np.max(action_logits))

    return hidden, {
        "entropy": float(entropy),
        "max_prob": max_prob,
        "max_logit": max_logit,
        "energy": energy,
        "predicted_token": int(np.argmax(action_logits)),
    }

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
    print("Experiment 190: Logit-Based OOD Detection Comparison")
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
    cal_logit_stats = {"entropy": [], "max_prob": [], "max_logit": [], "energy": []}

    for l in layers:
        cal_embs_l = []
        for arr in cal_arrs:
            h, logit_info = extract_with_logits(model, processor, Image.fromarray(arr), prompt, layers)
            cal_embs_l.append(h[l])
            if l == layers[0]:
                for k in cal_logit_stats:
                    cal_logit_stats[k].append(logit_info[k])
        centroids[l] = np.array(cal_embs_l).mean(axis=0)

    # ID test
    test_arrs = [creators[(i+n_cal)%3](i+n_cal) for i in range(n_test)]
    id_hidden = {l: [] for l in layers}
    id_logits = {"entropy": [], "max_prob": [], "max_logit": [], "energy": []}

    for arr in test_arrs:
        h, logit_info = extract_with_logits(model, processor, Image.fromarray(arr), prompt, layers)
        for l in layers:
            id_hidden[l].append(h[l])
        for k in id_logits:
            id_logits[k].append(logit_info[k])

    # OOD test
    ood_transforms = {
        "fog_60": lambda a: apply_fog(a, 0.6),
        "night": apply_night,
        "blur": apply_blur,
        "noise": apply_noise,
    }
    ood_hidden = {l: [] for l in layers}
    ood_logits = {"entropy": [], "max_prob": [], "max_logit": [], "energy": []}

    for cat, tfn in ood_transforms.items():
        for arr in test_arrs:
            h, logit_info = extract_with_logits(model, processor, Image.fromarray(tfn(arr)), prompt, layers)
            for l in layers:
                ood_hidden[l].append(h[l])
            for k in ood_logits:
                ood_logits[k].append(logit_info[k])

    # Compute AUROCs for all methods
    print("\n--- Method comparison ---", flush=True)
    results = {}

    # Cosine distance AUROC
    for l in layers:
        id_dists = [cosine_distance(e, centroids[l]) for e in id_hidden[l]]
        ood_dists = [cosine_distance(e, centroids[l]) for e in ood_hidden[l]]
        auroc = compute_auroc(id_dists, ood_dists)
        results[f"cosine_L{l}"] = {"auroc": auroc, "id_mean": float(np.mean(id_dists)), "ood_mean": float(np.mean(ood_dists))}
        print(f"  Cosine L{l}: AUROC={auroc:.4f}", flush=True)

    # Logit-based AUROCs
    # Entropy: higher = more uncertain = OOD
    auroc_entropy = compute_auroc(id_logits["entropy"], ood_logits["entropy"])
    results["entropy"] = {"auroc": auroc_entropy, "id_mean": float(np.mean(id_logits["entropy"])), "ood_mean": float(np.mean(ood_logits["entropy"]))}
    print(f"  Entropy: AUROC={auroc_entropy:.4f} (id={np.mean(id_logits['entropy']):.4f} ood={np.mean(ood_logits['entropy']):.4f})", flush=True)

    # Max prob: lower = more uncertain = OOD (reverse scoring)
    auroc_maxprob = compute_auroc([-p for p in id_logits["max_prob"]], [-p for p in ood_logits["max_prob"]])
    results["max_prob"] = {"auroc": auroc_maxprob, "id_mean": float(np.mean(id_logits["max_prob"])), "ood_mean": float(np.mean(ood_logits["max_prob"]))}
    print(f"  MaxProb: AUROC={auroc_maxprob:.4f} (id={np.mean(id_logits['max_prob']):.4f} ood={np.mean(ood_logits['max_prob']):.4f})", flush=True)

    # Max logit: lower = OOD (reverse scoring)
    auroc_maxlogit = compute_auroc([-l for l in id_logits["max_logit"]], [-l for l in ood_logits["max_logit"]])
    results["max_logit"] = {"auroc": auroc_maxlogit, "id_mean": float(np.mean(id_logits["max_logit"])), "ood_mean": float(np.mean(ood_logits["max_logit"]))}
    print(f"  MaxLogit: AUROC={auroc_maxlogit:.4f} (id={np.mean(id_logits['max_logit']):.4f} ood={np.mean(ood_logits['max_logit']):.4f})", flush=True)

    # Energy: higher energy = OOD
    auroc_energy = compute_auroc(id_logits["energy"], ood_logits["energy"])
    results["energy"] = {"auroc": auroc_energy, "id_mean": float(np.mean(id_logits["energy"])), "ood_mean": float(np.mean(ood_logits["energy"]))}
    print(f"  Energy: AUROC={auroc_energy:.4f} (id={np.mean(id_logits['energy']):.4f} ood={np.mean(ood_logits['energy']):.4f})", flush=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "logit_ood_detection",
        "experiment_number": 190,
        "timestamp": ts,
        "n_cal": n_cal, "n_test": n_test,
        "layers": layers,
        "results": results,
    }
    path = os.path.join(RESULTS_DIR, f"logit_ood_detection_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
