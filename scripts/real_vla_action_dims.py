#!/usr/bin/env python3
"""Experiment 207: Per-action-dimension analysis — which of the 7 action 
dimensions (dx, dy, dz, droll, dpitch, dyaw, gripper) is most affected 
by OOD inputs? Uses multi-step generation to extract all 7 action tokens.
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

ACTION_TOKEN_START = 31744
ACTION_TOKEN_END = 31999
N_BINS = 256
ACTION_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]

def decode_action(token_id):
    """Decode a single action token to continuous value."""
    if ACTION_TOKEN_START <= token_id <= ACTION_TOKEN_END:
        bin_idx = token_id - ACTION_TOKEN_START
        return (bin_idx / (N_BINS - 1)) * 2 - 1
    return None

def main():
    print("=" * 60)
    print("Experiment 207: Per-Action-Dimension Analysis")
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
    n_test = 6

    def get_actions(image):
        """Get all 7 action predictions via greedy generation."""
        inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            # Generate 7 tokens
            generated = model.generate(
                **inputs,
                max_new_tokens=7,
                do_sample=False,
            )
        # Extract generated token IDs
        input_len = inputs["input_ids"].shape[1]
        new_tokens = generated[0, input_len:].cpu().tolist()
        
        actions = []
        for t in new_tokens[:7]:
            val = decode_action(t)
            if val is not None:
                actions.append(val)
            else:
                actions.append(0.0)  # fallback for non-action tokens
        
        # Pad to 7 if needed
        while len(actions) < 7:
            actions.append(0.0)
        
        return actions[:7], new_tokens[:7]

    # ID predictions
    print("\n--- ID predictions ---", flush=True)
    test_arrs = [creators[(i+10)%3](i+10) for i in range(n_test)]
    id_actions = []
    id_tokens = []
    for arr in test_arrs:
        actions, tokens = get_actions(Image.fromarray(arr))
        id_actions.append(actions)
        id_tokens.append(tokens)
        print(f"  ID: {[f'{a:.3f}' for a in actions]}", flush=True)

    # OOD predictions
    print("\n--- OOD predictions ---", flush=True)
    ood_transforms = {
        "fog_60": lambda a: apply_fog(a, 0.6),
        "night": apply_night,
        "blur": apply_blur,
        "noise": apply_noise,
    }
    ood_actions = {cat: [] for cat in ood_transforms}
    ood_tokens = {cat: [] for cat in ood_transforms}
    for cat, tfn in ood_transforms.items():
        for arr in test_arrs:
            actions, tokens = get_actions(Image.fromarray(tfn(arr)))
            ood_actions[cat].append(actions)
            ood_tokens[cat].append(tokens)
        print(f"  {cat}: {[f'{np.mean([a[d] for a in ood_actions[cat]]):.3f}' for d in range(7)]}", flush=True)

    # Analysis
    print("\n--- Per-dimension analysis ---", flush=True)
    results = {}
    id_arr = np.array(id_actions)  # [n_test, 7]
    
    for cat in ood_transforms:
        ood_arr = np.array(ood_actions[cat])  # [n_test, 7]
        
        per_dim = {}
        for d in range(7):
            id_vals = id_arr[:, d]
            ood_vals = ood_arr[:, d]
            
            delta = np.abs(ood_vals - id_vals)
            pooled_std = np.sqrt((np.std(id_vals)**2 + np.std(ood_vals)**2) / 2) + 1e-10
            cohens_d = float(np.abs(np.mean(ood_vals) - np.mean(id_vals)) / pooled_std)
            
            token_changes = sum(1 for i in range(n_test) if ood_tokens[cat][i][d] != id_tokens[0][d])
            
            per_dim[ACTION_NAMES[d]] = {
                "id_mean": float(np.mean(id_vals)),
                "ood_mean": float(np.mean(ood_vals)),
                "mean_delta": float(np.mean(delta)),
                "max_delta": float(np.max(delta)),
                "cohens_d": cohens_d,
                "token_change_rate": float(token_changes / n_test),
            }
        
        results[cat] = per_dim
        print(f"  {cat}:", flush=True)
        for d in range(7):
            dim = ACTION_NAMES[d]
            print(f"    {dim}: delta={per_dim[dim]['mean_delta']:.4f} d={per_dim[dim]['cohens_d']:.2f}", flush=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "action_dimensions",
        "experiment_number": 207,
        "timestamp": ts,
        "n_test": n_test,
        "action_names": ACTION_NAMES,
        "id_actions": id_arr.tolist(),
        "results": results,
    }
    path = os.path.join(RESULTS_DIR, f"action_dimensions_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
