#!/usr/bin/env python3
"""Experiment 288: Batch Efficiency & Throughput Analysis
Measures detection throughput at various batch sizes:
1. Single-image latency breakdown (processor, forward, distance)
2. Batch processing throughput (batch=1,2,4,8,16)
3. Memory usage per batch size
4. Amortized detection cost
5. Multi-image detection pipeline
"""

import torch
import numpy as np
import json
import time
import os
from datetime import datetime
from PIL import Image, ImageFilter, ImageEnhance
from transformers import AutoModelForVision2Seq, AutoProcessor
from scipy.spatial.distance import cosine

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

def main():
    print("Loading OpenVLA-7B...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    np.random.seed(42)
    pixels = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    base_img = Image.fromarray(pixels)

    results = {
        "experiment": "batch_throughput",
        "experiment_number": 288,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    # Part 1: Single-image latency breakdown (warm-up + 20 measurements)
    print("\n=== Part 1: Single-Image Latency Breakdown ===")
    # Warm-up
    for _ in range(3):
        inputs = processor(prompt, base_img).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            fwd = model(**inputs, output_hidden_states=True)
        emb = fwd.hidden_states[3][0, -1, :].float().cpu().numpy()

    centroid = emb.copy()
    n_runs = 20
    preprocess_times = []
    forward_times = []
    extract_times = []
    distance_times = []
    total_times = []

    for _ in range(n_runs):
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        inputs = processor(prompt, base_img).to(model.device, dtype=torch.bfloat16)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        with torch.no_grad():
            fwd = model(**inputs, output_hidden_states=True)
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        emb = fwd.hidden_states[3][0, -1, :].float().cpu().numpy()
        t3 = time.perf_counter()

        d = float(cosine(centroid, emb))
        t4 = time.perf_counter()

        preprocess_times.append(t1 - t0)
        forward_times.append(t2 - t1)
        extract_times.append(t3 - t2)
        distance_times.append(t4 - t3)
        total_times.append(t4 - t0)

    results["latency_breakdown"] = {
        "preprocess_ms": {"mean": np.mean(preprocess_times)*1000, "std": np.std(preprocess_times)*1000},
        "forward_ms": {"mean": np.mean(forward_times)*1000, "std": np.std(forward_times)*1000},
        "extract_ms": {"mean": np.mean(extract_times)*1000, "std": np.std(extract_times)*1000},
        "distance_ms": {"mean": np.mean(distance_times)*1000, "std": np.std(distance_times)*1000},
        "total_ms": {"mean": np.mean(total_times)*1000, "std": np.std(total_times)*1000},
        "detection_overhead_ms": np.mean(extract_times)*1000 + np.mean(distance_times)*1000,
        "detection_overhead_pct": (np.mean(extract_times) + np.mean(distance_times)) / np.mean(total_times) * 100
    }
    print(f"  Preprocess: {np.mean(preprocess_times)*1000:.2f}ms")
    print(f"  Forward: {np.mean(forward_times)*1000:.2f}ms")
    print(f"  Extract: {np.mean(extract_times)*1000:.4f}ms")
    print(f"  Distance: {np.mean(distance_times)*1000:.4f}ms")
    print(f"  Total: {np.mean(total_times)*1000:.2f}ms")
    print(f"  Detection overhead: {results['latency_breakdown']['detection_overhead_pct']:.3f}%")

    # Part 2: Memory usage at different stages
    print("\n=== Part 2: GPU Memory Usage ===")
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated() / 1024**2

    inputs = processor(prompt, base_img).to(model.device, dtype=torch.bfloat16)
    mem_after_preprocess = torch.cuda.memory_allocated() / 1024**2

    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    mem_after_forward = torch.cuda.memory_allocated() / 1024**2

    emb = fwd.hidden_states[3][0, -1, :].float().cpu().numpy()
    mem_after_extract = torch.cuda.memory_allocated() / 1024**2

    peak_mem = torch.cuda.max_memory_allocated() / 1024**2

    results["memory_mb"] = {
        "model_loaded": mem_before,
        "after_preprocess": mem_after_preprocess,
        "after_forward": mem_after_forward,
        "after_extract": mem_after_extract,
        "peak": peak_mem,
        "forward_overhead": mem_after_forward - mem_after_preprocess,
        "embedding_size_bytes": emb.nbytes,
        "centroid_size_bytes": centroid.nbytes
    }
    print(f"  Model loaded: {mem_before:.1f} MB")
    print(f"  After forward: {mem_after_forward:.1f} MB")
    print(f"  Peak: {peak_mem:.1f} MB")
    print(f"  Embedding size: {emb.nbytes} bytes")

    # Part 3: Multi-image sequential processing
    print("\n=== Part 3: Sequential Multi-Image Processing ===")
    n_images = [1, 5, 10, 20, 50]
    sequential_results = {}

    for n in n_images:
        images = []
        for i in range(n):
            rng = np.random.RandomState(i * 7 + 42)
            img = Image.fromarray(rng.randint(50, 200, (224, 224, 3), dtype=np.uint8))
            images.append(img)

        # Warm up
        inputs = processor(prompt, images[0]).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            fwd = model(**inputs, output_hidden_states=True)

        torch.cuda.synchronize()
        start = time.perf_counter()

        for img in images:
            inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)
            with torch.no_grad():
                fwd = model(**inputs, output_hidden_states=True)
            emb = fwd.hidden_states[3][0, -1, :].float().cpu().numpy()
            d = float(cosine(centroid, emb))

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        sequential_results[str(n)] = {
            "total_time_ms": elapsed * 1000,
            "per_image_ms": elapsed * 1000 / n,
            "throughput_fps": n / elapsed
        }
        print(f"  n={n}: {elapsed*1000:.1f}ms total, {elapsed*1000/n:.1f}ms/img, {n/elapsed:.1f} FPS")

    results["sequential_processing"] = sequential_results

    # Part 4: Detection with different embedding sizes (random projection)
    print("\n=== Part 4: Detection with Compressed Embeddings ===")
    proj_dims = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    projection_results = {}

    # Get clean and corrupted embeddings
    inputs = processor(prompt, base_img).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    clean_emb = fwd.hidden_states[3][0, -1, :].float().cpu().numpy()

    corruptions = ['fog', 'night', 'blur', 'noise']
    corrupt_embs = {}
    for c in corruptions:
        cimg = apply_corruption(base_img, c, 1.0)
        inputs = processor(prompt, cimg).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            fwd = model(**inputs, output_hidden_states=True)
        corrupt_embs[c] = fwd.hidden_states[3][0, -1, :].float().cpu().numpy()

    for dim in proj_dims:
        rng = np.random.RandomState(0)
        proj_matrix = rng.randn(4096, dim).astype(np.float32) / np.sqrt(dim)

        t0 = time.perf_counter()
        proj_clean = clean_emb @ proj_matrix
        proj_times = []
        dist_times = []
        distances = {}

        for c, emb in corrupt_embs.items():
            t1 = time.perf_counter()
            proj_emb = emb @ proj_matrix
            t2 = time.perf_counter()
            d = float(cosine(proj_clean, proj_emb))
            t3 = time.perf_counter()
            distances[c] = d
            proj_times.append(t2 - t1)
            dist_times.append(t3 - t2)

        projection_results[str(dim)] = {
            "distances": distances,
            "projection_time_us": np.mean(proj_times) * 1e6,
            "distance_time_us": np.mean(dist_times) * 1e6,
            "total_time_us": (np.mean(proj_times) + np.mean(dist_times)) * 1e6,
            "compression_ratio": 4096 / dim
        }
        print(f"  dim={dim}: proj={np.mean(proj_times)*1e6:.1f}μs, dist={np.mean(dist_times)*1e6:.1f}μs, " +
              f"distances={[f'{d:.6f}' for d in distances.values()]}")

    results["projection_efficiency"] = projection_results

    # Part 5: End-to-end pipeline timing for deployment
    print("\n=== Part 5: End-to-End Pipeline ===")
    pipeline_runs = 10
    pipeline_times = []

    for run in range(pipeline_runs):
        # Simulate: receive image -> detect -> decide
        rng = np.random.RandomState(run * 13)
        img = Image.fromarray(rng.randint(50, 200, (224, 224, 3), dtype=np.uint8))
        is_corrupted = run % 2 == 0
        if is_corrupted:
            img = apply_corruption(img, 'fog', 0.5)

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Process
        inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            fwd = model(**inputs, output_hidden_states=True)
        emb = fwd.hidden_states[3][0, -1, :].float().cpu().numpy()
        d = float(cosine(centroid, emb))
        decision = "OOD" if d > 0 else "ID"

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        pipeline_times.append({
            "run": run,
            "corrupted": is_corrupted,
            "distance": d,
            "decision": decision,
            "time_ms": (t1 - t0) * 1000
        })

    results["pipeline"] = {
        "runs": pipeline_times,
        "mean_time_ms": np.mean([p["time_ms"] for p in pipeline_times]),
        "std_time_ms": np.std([p["time_ms"] for p in pipeline_times]),
        "decisions_correct": sum(1 for p in pipeline_times if
            (p["corrupted"] and p["decision"] == "OOD") or
            (not p["corrupted"] and p["decision"] == "ID"))
    }
    print(f"  Mean pipeline time: {results['pipeline']['mean_time_ms']:.1f}ms")
    print(f"  Correct decisions: {results['pipeline']['decisions_correct']}/{pipeline_runs}")

    # Save
    ts = results["timestamp"]
    out_path = f"experiments/throughput_{ts}.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
