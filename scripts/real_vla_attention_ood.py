#!/usr/bin/env python3
"""Experiment 192: Attention pattern analysis under OOD — do attention
distributions change when inputs are corrupted?

Extracts attention weights from key layers and compares ID vs OOD
attention patterns (entropy, concentration, etc.).
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

def extract_attention(model, processor, image, prompt):
    """Extract attention weights from the model."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True, output_attentions=True)

    # fwd.attentions is a tuple of (n_layers) tensors, each [batch, heads, seq, seq]
    attn_layers = [1, 3, 16, 31]
    result = {}
    for l in attn_layers:
        if l < len(fwd.attentions):
            attn = fwd.attentions[l][0].float().cpu().numpy()  # [heads, seq, seq]
            # Last token's attention distribution
            last_attn = attn[:, -1, :]  # [heads, seq]

            # Attention entropy per head
            entropies = []
            for h in range(last_attn.shape[0]):
                a = last_attn[h]
                a = a + 1e-10
                ent = -float(np.sum(a * np.log(a)))
                entropies.append(ent)

            # Max attention weight
            max_attn = float(np.max(last_attn))

            # Mean of max attention per head
            mean_max = float(np.mean(np.max(last_attn, axis=1)))

            # Attention concentration (top-5% tokens)
            n_top = max(1, last_attn.shape[1] // 20)
            top_mass = float(np.mean([np.sum(np.sort(last_attn[h])[-n_top:]) for h in range(last_attn.shape[0])]))

            result[l] = {
                "mean_entropy": float(np.mean(entropies)),
                "std_entropy": float(np.std(entropies)),
                "max_attn": max_attn,
                "mean_max_per_head": mean_max,
                "top5pct_mass": top_mass,
            }

    hidden = {l: fwd.hidden_states[l][0, -1, :].float().cpu().numpy() for l in attn_layers}
    return result, hidden

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
    print("Experiment 192: Attention Pattern OOD Analysis")
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

    # Calibrate
    print("\n--- Extracting attention patterns ---", flush=True)
    attn_layers = [1, 3, 16, 31]
    centroids = {}
    for arr in cal_arrs:
        _, hidden = extract_attention(model, processor, Image.fromarray(arr), prompt)
        for l in attn_layers:
            if l not in centroids:
                centroids[l] = []
            centroids[l].append(hidden[l])
    centroids = {l: np.array(v).mean(axis=0) for l, v in centroids.items()}

    # ID
    id_attn = {l: [] for l in attn_layers}
    id_hidden = {l: [] for l in attn_layers}
    for arr in test_arrs:
        attn, hidden = extract_attention(model, processor, Image.fromarray(arr), prompt)
        for l in attn_layers:
            id_attn[l].append(attn[l])
            id_hidden[l].append(hidden[l])

    # OOD
    ood_attn = {l: [] for l in attn_layers}
    ood_hidden = {l: [] for l in attn_layers}
    for cat, tfn in ood_transforms.items():
        for arr in test_arrs:
            attn, hidden = extract_attention(model, processor, Image.fromarray(tfn(arr)), prompt)
            for l in attn_layers:
                ood_attn[l].append(attn[l])
                ood_hidden[l].append(hidden[l])

    # Analyze attention-based OOD detection
    print("\n--- Attention-based OOD metrics ---", flush=True)
    results = {}
    for l in attn_layers:
        layer_results = {}

        # Cosine AUROC (baseline)
        id_dists = [cosine_distance(e, centroids[l]) for e in id_hidden[l]]
        ood_dists = [cosine_distance(e, centroids[l]) for e in ood_hidden[l]]
        layer_results["cosine_auroc"] = compute_auroc(id_dists, ood_dists)

        # Attention entropy AUROC
        id_ent = [a["mean_entropy"] for a in id_attn[l]]
        ood_ent = [a["mean_entropy"] for a in ood_attn[l]]
        layer_results["attention_entropy_auroc"] = compute_auroc(id_ent, ood_ent)
        layer_results["id_entropy"] = {"mean": float(np.mean(id_ent)), "std": float(np.std(id_ent))}
        layer_results["ood_entropy"] = {"mean": float(np.mean(ood_ent)), "std": float(np.std(ood_ent))}

        # Attention concentration AUROC
        id_conc = [a["top5pct_mass"] for a in id_attn[l]]
        ood_conc = [a["top5pct_mass"] for a in ood_attn[l]]
        auroc_conc = compute_auroc(id_conc, ood_conc)
        auroc_conc_rev = compute_auroc([-x for x in id_conc], [-x for x in ood_conc])
        layer_results["concentration_auroc"] = max(auroc_conc, auroc_conc_rev)

        print(f"  L{l}: cosine={layer_results['cosine_auroc']:.4f} attn_ent={layer_results['attention_entropy_auroc']:.4f} conc={layer_results['concentration_auroc']:.4f}", flush=True)
        print(f"        id_ent={np.mean(id_ent):.4f}±{np.std(id_ent):.4f} ood_ent={np.mean(ood_ent):.4f}±{np.std(ood_ent):.4f}", flush=True)

        results[f"L{l}"] = layer_results

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "attention_ood",
        "experiment_number": 192,
        "timestamp": ts,
        "n_cal": n_cal, "n_test": n_test,
        "attn_layers": attn_layers,
        "results": results,
    }
    path = os.path.join(RESULTS_DIR, f"attention_ood_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
