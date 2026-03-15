#!/usr/bin/env python3
"""Experiment 169: Adaptive threshold via running statistics.

Tests an adaptive threshold that updates based on a sliding window of
recent embeddings, enabling the detector to handle gradual distribution
shift (e.g., time-of-day changes) without recalibration.
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
def apply_night(a, brightness=0.15):
    return np.clip(a*brightness, 0, 255).astype(np.uint8)

def extract_hidden(model, processor, image, prompt, layers):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return {l: fwd.hidden_states[l][0, -1, :].float().cpu().numpy() for l in layers}

def cosine_distance(a, b):
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def main():
    print("=" * 60)
    print("Experiment 169: Adaptive Threshold via Running Statistics")
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

    # Initial calibration
    n_cal = 6
    cal_arrs = [creators[i%3](i) for i in range(n_cal)]
    centroids = {}
    cal_stats = {}
    for l in layers:
        cal_embs = []
        for arr in cal_arrs:
            h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
            cal_embs.append(h[l])
        centroid = np.array(cal_embs).mean(axis=0)
        dists = [cosine_distance(e, centroid) for e in cal_embs]
        centroids[l] = centroid
        cal_stats[l] = {"mean": float(np.mean(dists)), "std": float(np.std(dists))}

    # Scenario: Gradual fog increase then sudden night
    # Frame sequence: 30 frames
    base_img = creators[0](100)
    frame_conditions = []
    # Frames 0-9: clean
    for i in range(10):
        frame_conditions.append(("clean", lambda a=base_img: a))
    # Frames 10-19: gradually increasing fog (0.05 to 0.50)
    for i in range(10):
        alpha = 0.05 * (i + 1)
        frame_conditions.append((f"fog_{alpha:.2f}", lambda a=base_img, al=alpha: apply_fog(a, al)))
    # Frames 20-24: sudden night
    for i in range(5):
        frame_conditions.append(("night", lambda a=base_img: apply_night(a)))
    # Frames 25-29: back to clean
    for i in range(5):
        frame_conditions.append(("clean_after", lambda a=base_img: a))

    # Run with static threshold
    sigma = 3.0
    static_thresh = {l: cal_stats[l]["mean"] + sigma * cal_stats[l]["std"] for l in layers}

    # Run with adaptive threshold (EMA of distances)
    window_size = 5
    ema_alpha = 0.3  # exponential moving average weight

    static_results = []
    adaptive_results = []
    
    # EMA state
    ema_mean = {l: cal_stats[l]["mean"] for l in layers}
    ema_std = {l: cal_stats[l]["std"] for l in layers}
    recent_dists = {l: [] for l in layers}

    print("\n--- Running simulation ---", flush=True)
    for frame_idx, (cond_name, cond_fn) in enumerate(frame_conditions):
        img = cond_fn()
        h = extract_hidden(model, processor, Image.fromarray(img), prompt, layers)

        dists = {}
        for l in layers:
            dists[l] = cosine_distance(h[l], centroids[l])

        # Static detection
        static_flag = any(dists[l] > static_thresh[l] for l in layers)

        # Adaptive detection
        for l in layers:
            recent_dists[l].append(dists[l])
            if len(recent_dists[l]) > window_size:
                recent_dists[l] = recent_dists[l][-window_size:]
            # Only adapt if not currently flagged
            if not static_flag and len(recent_dists[l]) >= 2:
                ema_mean[l] = ema_alpha * np.mean(recent_dists[l]) + (1 - ema_alpha) * ema_mean[l]
                ema_std[l] = ema_alpha * np.std(recent_dists[l]) + (1 - ema_alpha) * ema_std[l]

        adaptive_thresh = {l: ema_mean[l] + sigma * max(ema_std[l], cal_stats[l]["std"] * 0.5) for l in layers}
        adaptive_flag = any(dists[l] > adaptive_thresh[l] for l in layers)

        static_results.append({
            "frame": frame_idx,
            "condition": cond_name,
            "dist_L3": dists[3],
            "dist_L32": dists[32],
            "flagged": static_flag,
            "thresh_L3": static_thresh[3],
            "thresh_L32": static_thresh[32],
        })
        adaptive_results.append({
            "frame": frame_idx,
            "condition": cond_name,
            "dist_L3": dists[3],
            "dist_L32": dists[32],
            "flagged": adaptive_flag,
            "thresh_L3": adaptive_thresh[3],
            "thresh_L32": adaptive_thresh[32],
            "ema_mean_L3": ema_mean[3],
            "ema_mean_L32": ema_mean[32],
        })

        s_flag = "OOD" if static_flag else "OK "
        a_flag = "OOD" if adaptive_flag else "OK "
        print(f"  Frame {frame_idx:2d} [{cond_name:>12s}]: L32={dists[32]:.4f} "
              f"static={s_flag} adaptive={a_flag}", flush=True)

    # Compute metrics
    def compute_metrics(results, true_ood_frames):
        flags = [r["flagged"] for r in results]
        tp = sum(1 for i, f in enumerate(flags) if f and i in true_ood_frames)
        fp = sum(1 for i, f in enumerate(flags) if f and i not in true_ood_frames)
        fn = sum(1 for i in true_ood_frames if not flags[i])
        tn = sum(1 for i, f in enumerate(flags) if not f and i not in true_ood_frames)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": precision, "recall": recall, "f1": f1}

    # True OOD: fog >= 0.30 (frames 15-19) + night (frames 20-24)
    true_ood = set(range(15, 25))
    static_metrics = compute_metrics(static_results, true_ood)
    adaptive_metrics = compute_metrics(adaptive_results, true_ood)

    print(f"\n  Static:   P={static_metrics['precision']:.3f} R={static_metrics['recall']:.3f} F1={static_metrics['f1']:.3f}")
    print(f"  Adaptive: P={adaptive_metrics['precision']:.3f} R={adaptive_metrics['recall']:.3f} F1={adaptive_metrics['f1']:.3f}")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "adaptive_threshold",
        "experiment_number": 169,
        "timestamp": ts,
        "n_cal": n_cal, "n_frames": len(frame_conditions),
        "sigma": sigma,
        "window_size": window_size,
        "ema_alpha": ema_alpha,
        "static_results": static_results,
        "adaptive_results": adaptive_results,
        "static_metrics": static_metrics,
        "adaptive_metrics": adaptive_metrics,
        "true_ood_frames": list(true_ood),
    }
    path = os.path.join(RESULTS_DIR, f"adaptive_threshold_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
