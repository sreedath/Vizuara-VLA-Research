#!/usr/bin/env python3
"""Experiment 193: Inference latency analysis — how much overhead does
OOD detection add to VLA inference?

Measures forward pass time with and without hidden state extraction,
and the computation time for centroid calibration and distance calculation.
"""

import json, os, sys, datetime, time
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

def cosine_distance(a, b):
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def main():
    print("=" * 60)
    print("Experiment 193: Inference Latency Analysis")
    print("=" * 60, flush=True)

    from transformers import AutoModelForVision2Seq, AutoProcessor
    print("Loading OpenVLA-7B...", flush=True)
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to drive forward at 25 m/s safely?\nOut:"
    img = Image.fromarray(create_highway(0))

    # Warmup
    print("\n--- Warmup ---", flush=True)
    for _ in range(3):
        inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            _ = model(**inputs)
    torch.cuda.synchronize()

    n_trials = 20

    # 1. Standard forward pass (no hidden states)
    print("--- Standard forward pass ---", flush=True)
    times_standard = []
    for i in range(n_trials):
        inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(**inputs)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_standard.append(t1 - t0)

    # 2. Forward pass with hidden states
    print("--- Forward pass with hidden states ---", flush=True)
    times_hidden = []
    for i in range(n_trials):
        inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            fwd = model(**inputs, output_hidden_states=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_hidden.append(t1 - t0)

    # 3. Hidden state extraction + cosine distance computation
    print("--- OOD detection overhead ---", flush=True)
    centroid = np.random.randn(4096).astype(np.float32)  # dummy centroid
    times_ood = []
    for i in range(n_trials):
        inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            fwd = model(**inputs, output_hidden_states=True)
        # Extract and compute distance
        emb = fwd.hidden_states[3][0, -1, :].float().cpu().numpy()
        dist = cosine_distance(emb, centroid)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_ood.append(t1 - t0)

    # 4. Just the cosine distance computation
    print("--- Pure cosine distance time ---", flush=True)
    emb = np.random.randn(4096).astype(np.float32)
    times_cosine = []
    for i in range(1000):
        t0 = time.perf_counter()
        _ = cosine_distance(emb, centroid)
        t1 = time.perf_counter()
        times_cosine.append(t1 - t0)

    # 5. Preprocessing time
    print("--- Preprocessing time ---", flush=True)
    times_preproc = []
    for i in range(n_trials):
        t0 = time.perf_counter()
        _ = processor(prompt, img)
        t1 = time.perf_counter()
        times_preproc.append(t1 - t0)

    results = {
        "standard_forward_ms": {
            "mean": float(np.mean(times_standard) * 1000),
            "std": float(np.std(times_standard) * 1000),
            "min": float(np.min(times_standard) * 1000),
            "max": float(np.max(times_standard) * 1000),
        },
        "hidden_states_forward_ms": {
            "mean": float(np.mean(times_hidden) * 1000),
            "std": float(np.std(times_hidden) * 1000),
            "min": float(np.min(times_hidden) * 1000),
            "max": float(np.max(times_hidden) * 1000),
        },
        "full_ood_detection_ms": {
            "mean": float(np.mean(times_ood) * 1000),
            "std": float(np.std(times_ood) * 1000),
            "min": float(np.min(times_ood) * 1000),
            "max": float(np.max(times_ood) * 1000),
        },
        "cosine_distance_us": {
            "mean": float(np.mean(times_cosine) * 1e6),
            "std": float(np.std(times_cosine) * 1e6),
        },
        "preprocessing_ms": {
            "mean": float(np.mean(times_preproc) * 1000),
            "std": float(np.std(times_preproc) * 1000),
        },
        "overhead_ms": float((np.mean(times_ood) - np.mean(times_standard)) * 1000),
        "overhead_pct": float((np.mean(times_ood) - np.mean(times_standard)) / np.mean(times_standard) * 100),
        "n_trials": n_trials,
    }

    print(f"\n  Standard forward:    {results['standard_forward_ms']['mean']:.1f} ± {results['standard_forward_ms']['std']:.1f} ms", flush=True)
    print(f"  Hidden states:       {results['hidden_states_forward_ms']['mean']:.1f} ± {results['hidden_states_forward_ms']['std']:.1f} ms", flush=True)
    print(f"  Full OOD detection:  {results['full_ood_detection_ms']['mean']:.1f} ± {results['full_ood_detection_ms']['std']:.1f} ms", flush=True)
    print(f"  Cosine distance:     {results['cosine_distance_us']['mean']:.1f} ± {results['cosine_distance_us']['std']:.1f} µs", flush=True)
    print(f"  Preprocessing:       {results['preprocessing_ms']['mean']:.1f} ± {results['preprocessing_ms']['std']:.1f} ms", flush=True)
    print(f"  OOD overhead:        {results['overhead_ms']:.1f} ms ({results['overhead_pct']:.1f}%)", flush=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "latency_analysis",
        "experiment_number": 193,
        "timestamp": ts,
        "results": results,
    }
    path = os.path.join(RESULTS_DIR, f"latency_analysis_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
