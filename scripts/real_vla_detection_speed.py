#!/usr/bin/env python3
"""Experiment 412: Detection Speed and Computational Cost Analysis

Precise benchmarking of the detection pipeline's computational overhead.
Critical for real-time deployment in autonomous systems.

Tests:
1. Full forward pass latency with hidden states
2. Hidden state extraction overhead vs without
3. Cosine distance computation latency
4. Random projection speedup at various dimensions
5. Memory footprint
6. Full detection pipeline latency
7. Scaling with calibration set size
"""

import json, time, os, sys
import numpy as np
import torch
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor

def apply_corruption(image, ctype, severity=1.0):
    arr = np.array(image).astype(np.float32) / 255.0
    if ctype == 'fog':
        arr = arr * (1 - 0.6 * severity) + 0.6 * severity
    elif ctype == 'night':
        arr = arr * max(0.01, 1.0 - 0.95 * severity)
    elif ctype == 'noise':
        arr = arr + np.random.RandomState(42).randn(*arr.shape) * 0.3 * severity
        arr = np.clip(arr, 0, 1)
    elif ctype == 'blur':
        return image.filter(ImageFilter.GaussianBlur(radius=10 * severity))
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))

def cosine_dist(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    return 1.0 - np.dot(a, b) / (na * nb)

def main():
    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    img = Image.fromarray(np.random.RandomState(42).randint(0, 255, (224, 224, 3), dtype=np.uint8))

    # Warmup
    print("Warmup...")
    for _ in range(3):
        inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            fwd = model(**inputs, output_hidden_states=True)
        emb = fwd.hidden_states[3][0, -1, :].float().cpu().numpy()

    results = {}
    n_trials = 20

    # === Test 1: Full forward pass latency ===
    print("\n=== Full Forward Pass Latency ===")
    forward_times = []
    for i in range(n_trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            fwd = model(**inputs, output_hidden_states=True)
        emb = fwd.hidden_states[3][0, -1, :].float().cpu().numpy()
        torch.cuda.synchronize()
        forward_times.append(time.perf_counter() - t0)

    results["forward_pass"] = {
        "mean_ms": float(np.mean(forward_times) * 1000),
        "std_ms": float(np.std(forward_times) * 1000),
        "p50_ms": float(np.percentile(forward_times, 50) * 1000),
        "p95_ms": float(np.percentile(forward_times, 95) * 1000),
        "p99_ms": float(np.percentile(forward_times, 99) * 1000),
        "min_ms": float(np.min(forward_times) * 1000),
        "max_ms": float(np.max(forward_times) * 1000),
        "n_trials": n_trials
    }
    print(f"  Forward pass: {np.mean(forward_times)*1000:.2f} ± {np.std(forward_times)*1000:.2f} ms")

    # === Test 2: Hidden state extraction overhead ===
    print("\n=== Hidden State Extraction Overhead ===")
    no_hidden_times = []
    for i in range(n_trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            fwd = model(**inputs)
        torch.cuda.synchronize()
        no_hidden_times.append(time.perf_counter() - t0)

    overhead = np.mean(forward_times) - np.mean(no_hidden_times)
    results["hidden_state_overhead"] = {
        "with_hidden_ms": float(np.mean(forward_times) * 1000),
        "without_hidden_ms": float(np.mean(no_hidden_times) * 1000),
        "overhead_ms": float(overhead * 1000),
        "overhead_pct": float(overhead / np.mean(no_hidden_times) * 100)
    }
    print(f"  Without hidden states: {np.mean(no_hidden_times)*1000:.2f} ms")
    print(f"  Hidden state overhead: {overhead*1000:.2f} ms ({overhead/np.mean(no_hidden_times)*100:.2f}%)")

    # === Test 3: Cosine distance computation ===
    print("\n=== Cosine Distance Computation ===")
    centroid = emb + np.random.randn(*emb.shape) * 0.001

    dist_times = []
    for i in range(10000):
        t0 = time.perf_counter()
        d = cosine_dist(emb, centroid)
        dist_times.append(time.perf_counter() - t0)

    results["cosine_distance"] = {
        "mean_us": float(np.mean(dist_times) * 1e6),
        "std_us": float(np.std(dist_times) * 1e6),
        "embedding_dim": int(emb.shape[0]),
        "n_trials": 10000
    }
    print(f"  Cosine distance: {np.mean(dist_times)*1e6:.1f} ± {np.std(dist_times)*1e6:.1f} μs (dim={emb.shape[0]})")

    # === Test 4: Random projection speedup ===
    print("\n=== Random Projection Speedup ===")
    proj_dims = [16, 32, 64, 128, 256, 512]
    proj_results = {}
    rng = np.random.RandomState(42)

    for d in proj_dims:
        proj_matrix = rng.randn(emb.shape[0], d).astype(np.float64) / np.sqrt(d)

        t0 = time.perf_counter()
        for _ in range(1000):
            emb_proj = emb @ proj_matrix
            cent_proj = centroid @ proj_matrix
        proj_time = (time.perf_counter() - t0) / 1000

        t0 = time.perf_counter()
        for _ in range(10000):
            d_proj = cosine_dist(emb_proj, cent_proj)
        dist_proj_time = (time.perf_counter() - t0) / 10000

        proj_results[str(d)] = {
            "projection_us": float(proj_time * 1e6),
            "distance_us": float(dist_proj_time * 1e6),
            "total_us": float((proj_time + dist_proj_time) * 1e6),
            "speedup_vs_full": float(np.mean(dist_times) / dist_proj_time)
        }

    results["random_projection"] = proj_results
    for d in proj_dims:
        r = proj_results[str(d)]
        print(f"  {d}D: proj={r['projection_us']:.1f}μs, dist={r['distance_us']:.1f}μs, speedup={r['speedup_vs_full']:.1f}×")

    # === Test 5: Memory footprint ===
    print("\n=== Memory Footprint ===")
    emb_bytes = emb.nbytes
    results["memory"] = {
        "single_embedding_bytes": int(emb_bytes),
        "single_embedding_kb": float(emb_bytes / 1024),
        "embedding_dim": int(emb.shape[0]),
        "dtype": str(emb.dtype),
        "centroid_bytes": int(emb_bytes),
        "projected_32d_bytes": int(32 * 8),
        "projection_matrix_bytes": int(emb.shape[0] * 32 * 8),
        "total_32d_system_kb": float((32 * 8 + emb.shape[0] * 32 * 8) / 1024)
    }
    print(f"  Full embedding: {emb_bytes} bytes ({emb_bytes/1024:.1f} KB), dim={emb.shape[0]}")
    print(f"  32D projected: {32*8} bytes")
    print(f"  32D system total: {(32*8 + emb.shape[0]*32*8)/1024:.1f} KB")

    # === Test 6: Full detection pipeline ===
    print("\n=== Full Detection Pipeline ===")
    pipeline_times = []
    for i in range(n_trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            fwd = model(**inputs, output_hidden_states=True)
        emb_i = fwd.hidden_states[3][0, -1, :].float().cpu().numpy()
        dist = cosine_dist(emb_i, centroid)
        is_ood = dist > 0.001
        torch.cuda.synchronize()
        pipeline_times.append(time.perf_counter() - t0)

    results["full_pipeline"] = {
        "mean_ms": float(np.mean(pipeline_times) * 1000),
        "std_ms": float(np.std(pipeline_times) * 1000),
        "p50_ms": float(np.percentile(pipeline_times, 50) * 1000),
        "p95_ms": float(np.percentile(pipeline_times, 95) * 1000),
        "detection_overhead_ms": float((np.mean(pipeline_times) - np.mean(no_hidden_times)) * 1000),
        "detection_overhead_pct": float((np.mean(pipeline_times) - np.mean(no_hidden_times)) / np.mean(no_hidden_times) * 100)
    }
    print(f"  Full pipeline: {np.mean(pipeline_times)*1000:.2f} ± {np.std(pipeline_times)*1000:.2f} ms")
    print(f"  Detection overhead: {(np.mean(pipeline_times)-np.mean(no_hidden_times))*1000:.2f} ms")

    # === Test 7: Scaling with calibration set size ===
    print("\n=== Scaling with Calibration Set Size ===")
    scaling_results = {}
    for n_cal in [1, 5, 10, 50, 100]:
        cal_set = [emb + np.random.randn(*emb.shape) * 0.0001 for _ in range(n_cal)]
        cal_centroid = np.mean(cal_set, axis=0)

        times = []
        for _ in range(1000):
            t0 = time.perf_counter()
            d = cosine_dist(emb, cal_centroid)
            times.append(time.perf_counter() - t0)

        scaling_results[str(n_cal)] = {
            "distance_us": float(np.mean(times) * 1e6)
        }

        t0 = time.perf_counter()
        for _ in range(100):
            c = np.mean(cal_set, axis=0)
        centroid_time = (time.perf_counter() - t0) / 100
        scaling_results[str(n_cal)]["centroid_compute_us"] = float(centroid_time * 1e6)

    results["scaling"] = scaling_results
    for n, r in scaling_results.items():
        print(f"  {n} cal images: dist={r['distance_us']:.1f}μs, centroid={r['centroid_compute_us']:.1f}μs")

    # Save
    out_path = "/workspace/Vizuara-VLA-Research/experiments/detection_speed_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
