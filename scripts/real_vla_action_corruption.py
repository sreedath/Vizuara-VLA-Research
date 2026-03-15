#!/usr/bin/env python3
"""Experiment 155: Action corruption under OOD inputs.

Measures how OOD corruptions change the model's actual action token outputs
compared to clean-input actions. Quantifies steering/velocity errors per
corruption type and severity.
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

# Action token range for OpenVLA: 31744-31999 (256 bins)
ACTION_TOKEN_START = 31744
ACTION_TOKEN_END = 31999
N_BINS = 256

def tokens_to_actions(token_ids):
    """Convert action token IDs to normalized action values in [-1, 1]."""
    bins = np.array(token_ids) - ACTION_TOKEN_START
    return 2.0 * bins / (N_BINS - 1) - 1.0

def extract_actions(model, processor, image, prompt, n_actions=7):
    """Generate action tokens and return both token IDs and normalized values."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=n_actions,
            do_sample=False,
        )
    gen_ids = out[0, input_len:].cpu().tolist()
    # Only keep valid action tokens
    action_ids = [t for t in gen_ids if ACTION_TOKEN_START <= t <= ACTION_TOKEN_END]
    if len(action_ids) < n_actions:
        # Pad with center bin if model produced fewer tokens
        action_ids.extend([ACTION_TOKEN_START + N_BINS//2] * (n_actions - len(action_ids)))
    action_ids = action_ids[:n_actions]
    return action_ids, tokens_to_actions(action_ids)

def main():
    print("=" * 60)
    print("Experiment 155: Action Corruption Under OOD Inputs")
    print("=" * 60, flush=True)

    from transformers import AutoModelForVision2Seq, AutoProcessor
    print("Loading OpenVLA-7B...", flush=True)
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to drive forward at 25 m/s safely?\nOut:"
    ACTION_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]

    creators = [create_highway, create_urban, create_rural]
    n_images = 6  # 2 per scene type

    # Generate clean images
    clean_arrs = [creators[i%3](i) for i in range(n_images)]

    # Get clean baseline actions
    print("\n--- Clean baseline actions ---", flush=True)
    clean_actions = []
    clean_token_ids = []
    for i, arr in enumerate(clean_arrs):
        tids, acts = extract_actions(model, processor, Image.fromarray(arr), prompt)
        clean_token_ids.append(tids)
        clean_actions.append(acts)
        print(f"  Image {i}: tokens={tids}, actions=[{', '.join(f'{a:.3f}' for a in acts)}]", flush=True)
    clean_actions = np.array(clean_actions)  # (n_images, 7)
    clean_mean = clean_actions.mean(axis=0)
    print(f"  Mean clean action: [{', '.join(f'{a:.3f}' for a in clean_mean)}]", flush=True)

    # Define corruption types with severities
    corruptions = {
        "fog_20": lambda a: apply_fog(a, 0.2),
        "fog_40": lambda a: apply_fog(a, 0.4),
        "fog_60": lambda a: apply_fog(a, 0.6),
        "fog_80": lambda a: apply_fog(a, 0.8),
        "night": apply_night,
        "blur_4": lambda a: apply_blur(a, 4),
        "blur_8": lambda a: apply_blur(a, 8),
        "blur_16": lambda a: apply_blur(a, 16),
        "noise_25": lambda a: apply_noise(a, 25),
        "noise_50": lambda a: apply_noise(a, 50),
        "noise_100": lambda a: apply_noise(a, 100),
        "occlusion": apply_occlusion,
    }

    results = {
        "clean_mean_actions": clean_mean.tolist(),
        "clean_per_image": [ca.tolist() for ca in clean_actions],
        "clean_token_ids": clean_token_ids,
        "clean_std": clean_actions.std(axis=0).tolist(),
        "corruptions": {},
    }

    for cname, cfn in corruptions.items():
        print(f"\n--- {cname} ---", flush=True)
        corr_actions = []
        corr_token_ids = []
        for i, arr in enumerate(clean_arrs):
            corr_arr = cfn(arr)
            tids, acts = extract_actions(model, processor, Image.fromarray(corr_arr), prompt)
            corr_token_ids.append(tids)
            corr_actions.append(acts)
        corr_actions = np.array(corr_actions)
        corr_mean = corr_actions.mean(axis=0)

        # Compute errors
        abs_errors = np.abs(corr_actions - clean_actions)  # per image
        mean_abs_error = abs_errors.mean(axis=0)  # per dimension
        max_abs_error = abs_errors.max(axis=0)
        rmse = np.sqrt((( corr_actions - clean_actions)**2).mean(axis=0))

        # Overall metrics
        l2_shift = float(np.linalg.norm(corr_mean - clean_mean))
        max_dim_shift = float(np.max(np.abs(corr_mean - clean_mean)))
        most_affected_dim = int(np.argmax(np.abs(corr_mean - clean_mean)))

        # Token-level analysis
        token_changes = []
        for i in range(n_images):
            changes = sum(1 for a, b in zip(clean_token_ids[i], corr_token_ids[i]) if a != b)
            token_changes.append(changes)

        entry = {
            "corr_mean_actions": corr_mean.tolist(),
            "corr_std": corr_actions.std(axis=0).tolist(),
            "corr_token_ids": corr_token_ids,
            "mean_abs_error_per_dim": mean_abs_error.tolist(),
            "max_abs_error_per_dim": max_abs_error.tolist(),
            "rmse_per_dim": rmse.tolist(),
            "l2_shift": l2_shift,
            "max_dim_shift": max_dim_shift,
            "most_affected_dim": most_affected_dim,
            "most_affected_name": ACTION_NAMES[most_affected_dim],
            "mean_tokens_changed": float(np.mean(token_changes)),
            "total_mae": float(mean_abs_error.mean()),
        }
        results["corruptions"][cname] = entry
        print(f"  L2 shift={l2_shift:.4f}, max_dim_shift={max_dim_shift:.4f} ({ACTION_NAMES[most_affected_dim]})", flush=True)
        print(f"  MAE per dim: [{', '.join(f'{e:.4f}' for e in mean_abs_error)}]", flush=True)
        print(f"  Tokens changed per image: {token_changes} (mean={np.mean(token_changes):.1f}/7)", flush=True)

    # Summary table
    print("\n" + "=" * 80)
    print("ACTION CORRUPTION SUMMARY")
    print(f"{'Corruption':<15} {'L2 Shift':>10} {'Max Dim':>10} {'Affected':>10} {'Tok Chg':>10} {'Total MAE':>10}")
    for cname, entry in results["corruptions"].items():
        print(f"{cname:<15} {entry['l2_shift']:10.4f} {entry['max_dim_shift']:10.4f} "
              f"{entry['most_affected_name']:>10} {entry['mean_tokens_changed']:10.1f} {entry['total_mae']:10.4f}")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "action_corruption",
        "experiment_number": 155,
        "timestamp": ts,
        "n_images": n_images,
        "action_names": ACTION_NAMES,
        "n_bins": N_BINS,
        "results": results,
    }
    path = os.path.join(RESULTS_DIR, f"action_corruption_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
