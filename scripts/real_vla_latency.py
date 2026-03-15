#!/usr/bin/env python3
"""Experiment 166: Detection latency measurement.

Measures computational overhead of OOD detection: forward pass time with
and without output_hidden_states, embedding extraction time, distance
computation time.
"""

import json, os, sys, datetime, time
import numpy as np
import torch
from pathlib import Path
from PIL import Image

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
    print("Experiment 166: Detection Latency Measurement")
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
    n_warmup = 3
    n_bench = 10

    img_arr = create_highway(0)
    image = Image.fromarray(img_arr)

    # Warmup
    print("\n--- Warmup ---", flush=True)
    for _ in range(n_warmup):
        inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            model(**inputs, output_hidden_states=True)
    torch.cuda.synchronize()
    print("  Warmup done.", flush=True)

    # Benchmark 1: Forward pass WITHOUT hidden states
    print("\n--- Benchmark: Forward pass (no hidden states) ---", flush=True)
    times_no_hs = []
    for i in range(n_bench):
        inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=False)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_no_hs.append(t1 - t0)
        print(f"  Run {i}: {(t1-t0)*1000:.1f} ms", flush=True)

    # Benchmark 2: Forward pass WITH hidden states
    print("\n--- Benchmark: Forward pass (with hidden states) ---", flush=True)
    times_with_hs = []
    for i in range(n_bench):
        inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_with_hs.append(t1 - t0)
        print(f"  Run {i}: {(t1-t0)*1000:.1f} ms", flush=True)

    # Benchmark 3: Hidden state extraction time
    print("\n--- Benchmark: Embedding extraction ---", flush=True)
    times_extract = []
    centroid = np.random.randn(4096).astype(np.float32)
    for i in range(n_bench):
        inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        embs = {l: out.hidden_states[l][0, -1, :].float().cpu().numpy() for l in layers}
        t1 = time.perf_counter()
        times_extract.append(t1 - t0)

    # Benchmark 4: Distance computation time
    print("\n--- Benchmark: Distance computation ---", flush=True)
    times_dist = []
    emb = np.random.randn(4096).astype(np.float32)
    for i in range(n_bench * 100):
        t0 = time.perf_counter()
        d = cosine_distance(emb, centroid)
        t1 = time.perf_counter()
        times_dist.append(t1 - t0)

    # Benchmark 5: Full pipeline (processor + forward + extract + distance)
    print("\n--- Benchmark: Full detection pipeline ---", flush=True)
    times_full = []
    for i in range(n_bench):
        t0 = time.perf_counter()
        inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        torch.cuda.synchronize()
        embs = {l: out.hidden_states[l][0, -1, :].float().cpu().numpy() for l in layers}
        d3 = cosine_distance(embs[3], centroid)
        d32 = cosine_distance(embs[32], centroid)
        is_ood = d3 > 0.001 or d32 > 0.15
        t1 = time.perf_counter()
        times_full.append(t1 - t0)
        print(f"  Run {i}: {(t1-t0)*1000:.1f} ms", flush=True)

    # Benchmark 6: Generate (7 action tokens)
    print("\n--- Benchmark: Action generation (7 tokens) ---", flush=True)
    times_gen = []
    for i in range(n_bench):
        inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
        input_len = inputs["input_ids"].shape[1]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=7, do_sample=False)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_gen.append(t1 - t0)
        print(f"  Run {i}: {(t1-t0)*1000:.1f} ms", flush=True)

    results = {
        "forward_no_hs_ms": {"mean": float(np.mean(times_no_hs))*1000, "std": float(np.std(times_no_hs))*1000},
        "forward_with_hs_ms": {"mean": float(np.mean(times_with_hs))*1000, "std": float(np.std(times_with_hs))*1000},
        "extract_ms": {"mean": float(np.mean(times_extract))*1000, "std": float(np.std(times_extract))*1000},
        "distance_us": {"mean": float(np.mean(times_dist))*1e6, "std": float(np.std(times_dist))*1e6},
        "full_pipeline_ms": {"mean": float(np.mean(times_full))*1000, "std": float(np.std(times_full))*1000},
        "generate_7tok_ms": {"mean": float(np.mean(times_gen))*1000, "std": float(np.std(times_gen))*1000},
    }

    overhead_ms = results["forward_with_hs_ms"]["mean"] - results["forward_no_hs_ms"]["mean"]
    overhead_pct = overhead_ms / results["forward_no_hs_ms"]["mean"] * 100
    results["hs_overhead_ms"] = overhead_ms
    results["hs_overhead_pct"] = overhead_pct

    detection_overhead = results["full_pipeline_ms"]["mean"] - results["forward_no_hs_ms"]["mean"]
    results["detection_overhead_ms"] = detection_overhead

    print("\n" + "=" * 80)
    print("LATENCY SUMMARY")
    print(f"  Forward (no HS):   {results['forward_no_hs_ms']['mean']:.1f} ± {results['forward_no_hs_ms']['std']:.1f} ms")
    print(f"  Forward (with HS): {results['forward_with_hs_ms']['mean']:.1f} ± {results['forward_with_hs_ms']['std']:.1f} ms")
    print(f"  HS overhead:       {overhead_ms:.1f} ms ({overhead_pct:.1f}%)")
    print(f"  Embedding extract: {results['extract_ms']['mean']:.3f} ± {results['extract_ms']['std']:.3f} ms")
    print(f"  Distance compute:  {results['distance_us']['mean']:.1f} ± {results['distance_us']['std']:.1f} μs")
    print(f"  Full pipeline:     {results['full_pipeline_ms']['mean']:.1f} ± {results['full_pipeline_ms']['std']:.1f} ms")
    print(f"  Generate (7 tok):  {results['generate_7tok_ms']['mean']:.1f} ± {results['generate_7tok_ms']['std']:.1f} ms")
    print(f"  Detection overhead: {detection_overhead:.1f} ms over baseline forward pass")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "detection_latency",
        "experiment_number": 166,
        "timestamp": ts,
        "n_warmup": n_warmup, "n_bench": n_bench,
        "gpu": "A40 48GB",
        "model": "openvla-7b",
        "results": results,
    }
    path = os.path.join(RESULTS_DIR, f"latency_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
