#!/usr/bin/env python3
"""Experiment 160: Deployment simulation pipeline.

End-to-end simulation: detection → severity estimation → graduated response.
Tests the full pipeline with temporal sequences and decision logic.
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
    print("Experiment 160: Deployment Simulation Pipeline")
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

    # === Phase 1: Calibration ===
    print("\n=== Phase 1: Calibration ===", flush=True)
    n_cal = 9  # 3 per scene type (diverse)
    cal_arrs = [creators[i%3](i) for i in range(n_cal)]
    centroids = {}
    cal_stats = {}
    for l in layers:
        cal_embs = []
        for arr in cal_arrs:
            h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
            cal_embs.append(h[l])
        embs = np.array(cal_embs)
        centroid = embs.mean(axis=0)
        dists = [cosine_distance(e, centroid) for e in embs]
        centroids[l] = centroid
        cal_stats[l] = {"mean": float(np.mean(dists)), "std": float(np.std(dists))}
    print(f"  L3 threshold (3σ): {cal_stats[3]['mean'] + 3*cal_stats[3]['std']:.6f}")
    print(f"  L32 threshold (3σ): {cal_stats[32]['mean'] + 3*cal_stats[32]['std']:.4f}")

    # === Phase 2: Define graduated response levels ===
    # Level 0: Normal operation
    # Level 1: Advisory (mild OOD, reduce speed by 20%)
    # Level 2: Warning (moderate OOD, reduce speed by 50%, increase sensor polling)
    # Level 3: Emergency (severe OOD, emergency stop)
    sigma = 3.0
    thresholds = {
        l: {
            "advisory": cal_stats[l]["mean"] + sigma * cal_stats[l]["std"],
            "warning": cal_stats[l]["mean"] + 2 * sigma * cal_stats[l]["std"],
            "emergency": cal_stats[l]["mean"] + 3 * sigma * cal_stats[l]["std"],
        }
        for l in layers
    }

    def classify_frame(dist_l3, dist_l32):
        """Classify a frame into response level using OR-gate logic."""
        # Check emergency first (either layer)
        if dist_l3 > thresholds[3]["emergency"] or dist_l32 > thresholds[32]["emergency"]:
            return 3, "EMERGENCY"
        if dist_l3 > thresholds[3]["warning"] or dist_l32 > thresholds[32]["warning"]:
            return 2, "WARNING"
        if dist_l3 > thresholds[3]["advisory"] or dist_l32 > thresholds[32]["advisory"]:
            return 1, "ADVISORY"
        return 0, "NORMAL"

    # === Phase 3: Simulate driving scenarios ===
    print("\n=== Phase 3: Simulation Scenarios ===", flush=True)

    scenarios = {}

    # Scenario A: Clear driving → fog rolls in → fog clears
    print("\n--- Scenario A: Fog onset and clearing ---", flush=True)
    base_img = creators[0](100)  # highway
    fog_sequence = (
        [0.0]*5 +  # clear
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] +  # fog intensifies
        [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0] +  # fog clears
        [0.0]*4  # clear again
    )
    scenario_a = []
    for frame_idx, fog_alpha in enumerate(fog_sequence):
        if fog_alpha > 0:
            img = apply_fog(base_img, fog_alpha)
        else:
            img = base_img
        h = extract_hidden(model, processor, Image.fromarray(img), prompt, layers)
        d3 = cosine_distance(h[3], centroids[3])
        d32 = cosine_distance(h[32], centroids[32])
        level, label = classify_frame(d3, d32)
        scenario_a.append({
            "frame": frame_idx,
            "fog_alpha": fog_alpha,
            "dist_L3": d3,
            "dist_L32": d32,
            "level": level,
            "label": label,
        })
        print(f"  Frame {frame_idx:2d}: fog={fog_alpha:.1f} → L3={d3:.6f} L32={d32:.4f} [{label}]", flush=True)
    scenarios["fog_cycle"] = scenario_a

    # Scenario B: Day → dusk → night → dawn
    print("\n--- Scenario B: Day/night transition ---", flush=True)
    brightness_sequence = (
        [1.0]*4 +  # day
        [0.8, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1] +  # dusk
        [0.1]*4 +  # night
        [0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0] +  # dawn
        [1.0]*3  # day again
    )
    scenario_b = []
    for frame_idx, brightness in enumerate(brightness_sequence):
        if brightness < 1.0:
            img = apply_night(base_img, brightness)
        else:
            img = base_img
        h = extract_hidden(model, processor, Image.fromarray(img), prompt, layers)
        d3 = cosine_distance(h[3], centroids[3])
        d32 = cosine_distance(h[32], centroids[32])
        level, label = classify_frame(d3, d32)
        scenario_b.append({
            "frame": frame_idx,
            "brightness": brightness,
            "dist_L3": d3,
            "dist_L32": d32,
            "level": level,
            "label": label,
        })
        print(f"  Frame {frame_idx:2d}: bright={brightness:.2f} → L3={d3:.6f} L32={d32:.4f} [{label}]", flush=True)
    scenarios["day_night_cycle"] = scenario_b

    # Scenario C: Mixed scene transitions (highway → urban → rural)
    print("\n--- Scenario C: Scene transitions ---", flush=True)
    scene_sequence = [
        ("highway", creators[0], 200),
        ("highway", creators[0], 201),
        ("highway", creators[0], 202),
        ("urban", creators[1], 203),
        ("urban", creators[1], 204),
        ("urban", creators[1], 205),
        ("rural", creators[2], 206),
        ("rural", creators[2], 207),
        ("rural", creators[2], 208),
        ("highway", creators[0], 209),
        ("highway", creators[0], 210),
    ]
    scenario_c = []
    for frame_idx, (scene, creator, seed) in enumerate(scene_sequence):
        img = creator(seed)
        h = extract_hidden(model, processor, Image.fromarray(img), prompt, layers)
        d3 = cosine_distance(h[3], centroids[3])
        d32 = cosine_distance(h[32], centroids[32])
        level, label = classify_frame(d3, d32)
        scenario_c.append({
            "frame": frame_idx,
            "scene": scene,
            "dist_L3": d3,
            "dist_L32": d32,
            "level": level,
            "label": label,
        })
        print(f"  Frame {frame_idx:2d}: {scene:>8s} → L3={d3:.6f} L32={d32:.4f} [{label}]", flush=True)
    scenarios["scene_transitions"] = scenario_c

    # === Phase 4: Compute deployment metrics ===
    print("\n=== Phase 4: Deployment Metrics ===", flush=True)
    metrics = {}
    for sname, frames in scenarios.items():
        levels = [f["level"] for f in frames]
        transitions = sum(1 for i in range(1, len(levels)) if levels[i] != levels[i-1])
        max_level = max(levels)
        normal_frac = levels.count(0) / len(levels)
        advisory_frac = levels.count(1) / len(levels)
        warning_frac = levels.count(2) / len(levels)
        emergency_frac = levels.count(3) / len(levels)
        metrics[sname] = {
            "n_frames": len(frames),
            "n_transitions": transitions,
            "max_level": max_level,
            "normal_frac": normal_frac,
            "advisory_frac": advisory_frac,
            "warning_frac": warning_frac,
            "emergency_frac": emergency_frac,
        }
        print(f"  {sname}: {len(frames)} frames, {transitions} transitions, "
              f"normal={normal_frac:.0%} adv={advisory_frac:.0%} warn={warning_frac:.0%} emerg={emergency_frac:.0%}")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "deployment_simulation",
        "experiment_number": 160,
        "timestamp": ts,
        "n_cal": n_cal,
        "sigma": sigma,
        "thresholds": {str(l): {k: float(v) for k, v in thresholds[l].items()} for l in layers},
        "cal_stats": {str(l): cal_stats[l] for l in layers},
        "scenarios": scenarios,
        "metrics": metrics,
    }
    path = os.path.join(RESULTS_DIR, f"deployment_sim_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
