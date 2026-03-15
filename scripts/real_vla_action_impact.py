#!/usr/bin/env python3
"""Experiment 199: Action prediction impact analysis — how much do OOD inputs
actually change the model's action predictions? Measures the cosine distance between
ID and OOD action predictions to validate that OOD detection is meaningful.
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

ACTION_TOKEN_START = 31744
ACTION_TOKEN_END = 31999
N_BINS = 256

def decode_actions(token_ids):
    """Decode action token IDs to continuous values."""
    actions = []
    for tid in token_ids:
        if ACTION_TOKEN_START <= tid <= ACTION_TOKEN_END:
            bin_idx = tid - ACTION_TOKEN_START
            actions.append((bin_idx / (N_BINS - 1)) * 2 - 1)  # map to [-1, 1]
    return actions

def main():
    print("=" * 60)
    print("Experiment 199: Action Prediction Impact Analysis")
    print("=" * 60, flush=True)

    from transformers import AutoModelForVision2Seq, AutoProcessor
    print("Loading OpenVLA-7B...", flush=True)
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to drive forward at 25 m/s safely?\nOut:"
    layers = [1, 3]

    creators = [create_highway, create_urban, create_rural]
    n_test = 8

    def extract_and_predict(image):
        inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            fwd = model(**inputs, output_hidden_states=True)
        
        # Get predicted action tokens (greedy)
        logits = fwd.logits[0, -1, :]  # last token logits
        # Get top-7 tokens in action range
        action_logits = logits[ACTION_TOKEN_START:ACTION_TOKEN_END+1]
        top_token = int(torch.argmax(action_logits).item()) + ACTION_TOKEN_START
        
        # Get full probability distribution over action bins
        action_probs = torch.softmax(action_logits.float(), dim=0).cpu().numpy()
        
        hidden = {l: fwd.hidden_states[l][0, -1, :].float().cpu().numpy() for l in layers}
        
        return {
            "top_token": top_token,
            "top_action": float((top_token - ACTION_TOKEN_START) / (N_BINS - 1) * 2 - 1),
            "action_probs": action_probs,
            "expected_action": float(np.sum(action_probs * np.linspace(-1, 1, N_BINS))),
            "action_entropy": float(-np.sum(action_probs * np.log(action_probs + 1e-10))),
            "hidden": hidden,
        }

    # Calibration centroids
    print("\n--- Calibration ---", flush=True)
    cal_arrs = [creators[i%3](i) for i in range(10)]
    cal_embs = {l: [] for l in layers}
    for arr in cal_arrs:
        result = extract_and_predict(Image.fromarray(arr))
        for l in layers:
            cal_embs[l].append(result["hidden"][l])
    centroids = {l: np.array(cal_embs[l]).mean(axis=0) for l in layers}

    # ID predictions
    print("--- ID predictions ---", flush=True)
    test_arrs = [creators[(i+10)%3](i+10) for i in range(n_test)]
    id_results = []
    for arr in test_arrs:
        result = extract_and_predict(Image.fromarray(arr))
        id_results.append(result)

    # OOD predictions
    print("--- OOD predictions ---", flush=True)
    ood_transforms = {
        "fog_60": lambda a: apply_fog(a, 0.6),
        "night": apply_night,
        "blur": apply_blur,
        "noise": apply_noise,
    }
    ood_results = {cat: [] for cat in ood_transforms}
    for cat, tfn in ood_transforms.items():
        for arr in test_arrs:
            result = extract_and_predict(Image.fromarray(tfn(arr)))
            ood_results[cat].append(result)

    # Analysis
    print("\n--- Analysis ---", flush=True)
    results = {}

    # ID statistics
    id_tokens = [r["top_token"] for r in id_results]
    id_actions = [r["top_action"] for r in id_results]
    id_expected = [r["expected_action"] for r in id_results]
    id_entropy = [r["action_entropy"] for r in id_results]
    results["id"] = {
        "mean_action": float(np.mean(id_actions)),
        "std_action": float(np.std(id_actions)),
        "mean_expected": float(np.mean(id_expected)),
        "mean_entropy": float(np.mean(id_entropy)),
        "unique_tokens": len(set(id_tokens)),
    }
    print(f"  ID: action={np.mean(id_actions):.4f}±{np.std(id_actions):.4f} entropy={np.mean(id_entropy):.4f}", flush=True)

    # Per-category OOD
    for cat in ood_transforms:
        ood_tokens = [r["top_token"] for r in ood_results[cat]]
        ood_actions = [r["top_action"] for r in ood_results[cat]]
        ood_expected = [r["expected_action"] for r in ood_results[cat]]
        ood_entropy = [r["action_entropy"] for r in ood_results[cat]]

        # Action prediction change
        action_deltas = [abs(ood_actions[i] - id_actions[i]) for i in range(n_test)]
        token_changes = sum(1 for i in range(n_test) if ood_tokens[i] != id_tokens[i])
        
        # Prob distribution KL divergence
        kl_divs = []
        for i in range(n_test):
            p = id_results[i]["action_probs"]
            q = ood_results[cat][i]["action_probs"]
            kl = float(np.sum(p * np.log((p + 1e-10) / (q + 1e-10))))
            kl_divs.append(kl)

        # Hidden state distances
        hidden_dists = {f"L{l}": float(np.mean([
            cosine_distance(ood_results[cat][i]["hidden"][l], centroids[l])
            for i in range(n_test)])) for l in layers}

        cat_results = {
            "mean_action": float(np.mean(ood_actions)),
            "std_action": float(np.std(ood_actions)),
            "mean_expected": float(np.mean(ood_expected)),
            "mean_entropy": float(np.mean(ood_entropy)),
            "mean_action_delta": float(np.mean(action_deltas)),
            "max_action_delta": float(np.max(action_deltas)),
            "token_change_rate": float(token_changes / n_test),
            "mean_kl_divergence": float(np.mean(kl_divs)),
            "hidden_dists": hidden_dists,
        }
        results[cat] = cat_results
        print(f"  {cat}: action={np.mean(ood_actions):.4f}±{np.std(ood_actions):.4f} "
              f"delta={np.mean(action_deltas):.4f} token_change={token_changes}/{n_test} "
              f"KL={np.mean(kl_divs):.4f}", flush=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "action_impact",
        "experiment_number": 199,
        "timestamp": ts,
        "n_test": n_test,
        "results": results,
    }
    path = os.path.join(RESULTS_DIR, f"action_impact_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
