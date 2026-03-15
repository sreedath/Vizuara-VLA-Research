#!/usr/bin/env python3
"""Experiment 153: Temporal consistency of OOD detection.

Simulates sequences of frames transitioning from ID to OOD (e.g., fog increasing
gradually). Tests if the detector flags transitions smoothly or with jitter.
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

def extract_hidden(model, processor, image, prompt, layers):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return {l: fwd.hidden_states[l][0, -1, :].float().cpu().numpy() for l in layers}

def cosine_distance(a, b):
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def main():
    print("=" * 60)
    print("Experiment 153: Temporal Consistency of OOD Detection")
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

    # Calibrate
    n_cal = 10
    cal_arrs = [create_highway(i) for i in range(n_cal)]
    print(f"\n--- Calibrating (n={n_cal}) ---", flush=True)
    cal_embs = {l: [] for l in layers}
    for arr in cal_arrs:
        h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
        for l in layers:
            cal_embs[l].append(h[l])

    centroids = {}; cal_stats = {}
    for l in layers:
        embs = np.array(cal_embs[l])
        centroids[l] = embs.mean(axis=0)
        dists = [cosine_distance(e, centroids[l]) for e in embs]
        cal_stats[l] = {"mean": float(np.mean(dists)), "std": float(np.std(dists))}

    sigma = 3.0
    thresholds = {l: cal_stats[l]["mean"] + sigma * cal_stats[l]["std"] for l in layers}

    # Define temporal sequences
    base_img = create_highway(100)
    sequences = {
        "fog_onset": {
            "description": "Fog increasing from 0% to 90%",
            "levels": np.linspace(0, 0.9, 20),
            "transform": lambda arr, alpha: np.clip(
                arr * (1 - alpha) + np.full_like(arr, [200, 200, 210]) * alpha, 0, 255
            ).astype(np.uint8),
        },
        "night_fall": {
            "description": "Brightness decreasing from 100% to 10%",
            "levels": np.linspace(1.0, 0.1, 20),
            "transform": lambda arr, b: np.clip(arr * b, 0, 255).astype(np.uint8),
        },
        "blur_increase": {
            "description": "Blur radius from 0 to 15",
            "levels": np.linspace(0, 15, 20),
            "transform": lambda arr, r: np.array(Image.fromarray(arr).filter(
                ImageFilter.GaussianBlur(radius=max(0.1, r))
            )) if r > 0.1 else arr,
        },
        "noise_increase": {
            "description": "Gaussian noise std from 0 to 100",
            "levels": np.linspace(0, 100, 20),
            "transform": lambda arr, s: np.clip(
                arr.astype(np.float32) + np.random.normal(0, max(0.1, s), arr.shape), 0, 255
            ).astype(np.uint8) if s > 0.5 else arr,
        },
    }

    # Run sequences
    print("\n--- Running temporal sequences ---", flush=True)
    results = {}

    for seq_name, seq_config in sequences.items():
        print(f"\n  Sequence: {seq_name}", flush=True)
        levels = seq_config["levels"]
        transform = seq_config["transform"]

        distances = {l: [] for l in layers}
        or_gate_flags = []

        for i, level in enumerate(levels):
            arr = transform(base_img, level)
            h = extract_hidden(model, processor, Image.fromarray(arr), prompt, layers)
            for l in layers:
                d = cosine_distance(h[l], centroids[l])
                distances[l].append(d)

            # OR-gate decision
            flagged = distances[3][-1] > thresholds[3] or distances[32][-1] > thresholds[32]
            or_gate_flags.append(flagged)

            status = "OOD" if flagged else "ID"
            if i % 5 == 0:
                print(f"    [{i+1}/20] level={level:.2f} L3={distances[3][-1]:.6f} L32={distances[32][-1]:.6f} → {status}", flush=True)

        # Compute transition point (first OOD flag)
        first_ood = next((i for i, f in enumerate(or_gate_flags) if f), len(or_gate_flags))
        # Compute jitter (number of ID→OOD→ID transitions after first OOD)
        transitions = sum(1 for i in range(first_ood, len(or_gate_flags)-1) if or_gate_flags[i] != or_gate_flags[i+1])

        results[seq_name] = {
            "description": seq_config["description"],
            "levels": [float(l) for l in levels],
            "distances": {f"L{l}": [float(d) for d in distances[l]] for l in layers},
            "thresholds": {f"L{l}": float(thresholds[l]) for l in layers},
            "or_gate_flags": [bool(f) for f in or_gate_flags],
            "first_ood_index": first_ood,
            "first_ood_level": float(levels[first_ood]) if first_ood < len(levels) else None,
            "jitter_after_first_ood": transitions,
            "n_flagged": sum(or_gate_flags),
        }
        print(f"    → First OOD at index {first_ood} (level={levels[first_ood] if first_ood < len(levels) else 'never':.2f}), jitter={transitions}", flush=True)

    # Summary
    print("\n" + "=" * 80)
    for sn, sr in results.items():
        print(f"  {sn:20s}: first_ood_idx={sr['first_ood_index']}, level={sr['first_ood_level']}, jitter={sr['jitter_after_first_ood']}, flagged={sr['n_flagged']}/20")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {"experiment": "temporal_consistency", "experiment_number": 153, "timestamp": ts,
              "n_cal": n_cal, "sigma": sigma, "layers": layers,
              "sequences": list(sequences.keys()), "results": results}
    path = os.path.join(RESULTS_DIR, f"temporal_consistency_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
