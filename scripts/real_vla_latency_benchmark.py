#!/usr/bin/env python3
"""Experiment 368: Real-Time Detection Latency Benchmark

Precise timing of each component in the detection pipeline:
1. Full inference time (embedding extraction)
2. Cosine distance computation time
3. Detection overhead as % of inference
4. Batch vs single-image throughput
5. Warm-up effects and timing stability
"""

import json, time, os, sys
import numpy as np
import torch
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor

def extract_hidden(model, processor, image, prompt, layer=3):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return fwd.hidden_states[layer][0, -1, :].float().cpu().numpy()

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
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return 1.0 - dot / (na * nb)

def main():
    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    results = {}

    # Generate test images
    print("Generating images...")
    rng = np.random.RandomState(42)
    px = rng.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    test_image = Image.fromarray(px)

    # Pre-compute calibration centroid
    cal_emb = extract_hidden(model, processor, test_image, prompt)

    # ========== 1. Warm-Up ==========
    print("\n=== Warm-Up (5 iterations) ===")
    for i in range(5):
        _ = extract_hidden(model, processor, test_image, prompt)
    print("  Done")

    # ========== 2. Full Inference Timing ==========
    print("\n=== Full Inference Timing (50 iterations) ===")

    inference_times = []
    for i in range(50):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        emb = extract_hidden(model, processor, test_image, prompt)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        inference_times.append(t1 - t0)

    results['inference'] = {
        'mean_ms': float(np.mean(inference_times) * 1000),
        'std_ms': float(np.std(inference_times) * 1000),
        'min_ms': float(min(inference_times) * 1000),
        'max_ms': float(max(inference_times) * 1000),
        'p50_ms': float(np.percentile(inference_times, 50) * 1000),
        'p95_ms': float(np.percentile(inference_times, 95) * 1000),
        'p99_ms': float(np.percentile(inference_times, 99) * 1000),
    }
    print(f"  Inference: {np.mean(inference_times)*1000:.2f} ms "
          f"(p50={np.percentile(inference_times, 50)*1000:.2f}, "
          f"p95={np.percentile(inference_times, 95)*1000:.2f})")

    # ========== 3. Cosine Distance Computation ==========
    print("\n=== Cosine Distance Timing (10000 iterations) ===")

    cosine_times = []
    for i in range(10000):
        t0 = time.perf_counter()
        d = cosine_dist(emb, cal_emb)
        t1 = time.perf_counter()
        cosine_times.append(t1 - t0)

    results['cosine_distance'] = {
        'mean_us': float(np.mean(cosine_times) * 1e6),
        'std_us': float(np.std(cosine_times) * 1e6),
        'min_us': float(min(cosine_times) * 1e6),
        'max_us': float(max(cosine_times) * 1e6),
        'p50_us': float(np.percentile(cosine_times, 50) * 1e6),
        'p95_us': float(np.percentile(cosine_times, 95) * 1e6),
    }
    print(f"  Cosine dist: {np.mean(cosine_times)*1e6:.2f} us "
          f"(p50={np.percentile(cosine_times, 50)*1e6:.2f})")

    # ========== 4. Preprocessing Timing ==========
    print("\n=== Preprocessing Timing (50 iterations) ===")

    preprocess_times = []
    for i in range(50):
        t0 = time.perf_counter()
        inputs = processor(prompt, test_image)
        t1 = time.perf_counter()
        preprocess_times.append(t1 - t0)

    results['preprocessing'] = {
        'mean_ms': float(np.mean(preprocess_times) * 1000),
        'std_ms': float(np.std(preprocess_times) * 1000),
        'min_ms': float(min(preprocess_times) * 1000),
        'max_ms': float(max(preprocess_times) * 1000),
    }
    print(f"  Preprocessing: {np.mean(preprocess_times)*1000:.2f} ms")

    # ========== 5. GPU Forward Pass Only ==========
    print("\n=== GPU Forward Pass Only (50 iterations) ===")

    inputs = processor(prompt, test_image).to(model.device, dtype=torch.bfloat16)
    forward_times = []
    for i in range(50):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            fwd = model(**inputs, output_hidden_states=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        forward_times.append(t1 - t0)

    results['forward_pass'] = {
        'mean_ms': float(np.mean(forward_times) * 1000),
        'std_ms': float(np.std(forward_times) * 1000),
        'min_ms': float(min(forward_times) * 1000),
        'max_ms': float(max(forward_times) * 1000),
        'p50_ms': float(np.percentile(forward_times, 50) * 1000),
        'p95_ms': float(np.percentile(forward_times, 95) * 1000),
    }
    print(f"  Forward: {np.mean(forward_times)*1000:.2f} ms")

    # ========== 6. Embedding Extraction from Hidden States ==========
    print("\n=== Embedding Extraction (10000 iterations) ===")

    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)

    extract_times = []
    for i in range(10000):
        t0 = time.perf_counter()
        e = fwd.hidden_states[3][0, -1, :].float().cpu().numpy()
        t1 = time.perf_counter()
        extract_times.append(t1 - t0)

    results['extraction'] = {
        'mean_us': float(np.mean(extract_times) * 1e6),
        'std_us': float(np.std(extract_times) * 1e6),
        'min_us': float(min(extract_times) * 1e6),
    }
    print(f"  Extraction: {np.mean(extract_times)*1e6:.2f} us")

    # ========== 7. Complete Detection Pipeline ==========
    print("\n=== Complete Detection Pipeline (50 iterations) ===")

    pipeline_times = []
    for i in range(50):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Step 1: Preprocess + Forward
        emb = extract_hidden(model, processor, test_image, prompt)

        # Step 2: Distance + Decision
        d = cosine_dist(emb, cal_emb)
        is_ood = d > 0.0001  # threshold

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        pipeline_times.append(t1 - t0)

    results['pipeline'] = {
        'mean_ms': float(np.mean(pipeline_times) * 1000),
        'std_ms': float(np.std(pipeline_times) * 1000),
        'min_ms': float(min(pipeline_times) * 1000),
        'max_ms': float(max(pipeline_times) * 1000),
    }
    print(f"  Pipeline: {np.mean(pipeline_times)*1000:.2f} ms")

    # ========== 8. Detection Overhead ==========
    print("\n=== Detection Overhead ===")

    detection_overhead_ms = np.mean(cosine_times) * 1000 + np.mean(extract_times) * 1000
    inference_ms = np.mean(forward_times) * 1000
    overhead_pct = detection_overhead_ms / inference_ms * 100

    results['overhead'] = {
        'detection_overhead_ms': float(detection_overhead_ms),
        'inference_ms': float(inference_ms),
        'overhead_pct': float(overhead_pct),
        'total_pipeline_ms': float(np.mean(pipeline_times) * 1000),
    }
    print(f"  Detection overhead: {detection_overhead_ms:.4f} ms ({overhead_pct:.3f}% of inference)")
    print(f"  Total pipeline: {np.mean(pipeline_times)*1000:.2f} ms")

    # ========== 9. Throughput Under Different Image Counts ==========
    print("\n=== Throughput (images/second) ===")

    # Generate multiple different images
    throughput = {}
    for n_images in [1, 5, 10, 20]:
        imgs = []
        for i in range(n_images):
            rng_i = np.random.RandomState(i * 100)
            px_i = rng_i.randint(50, 200, (224, 224, 3), dtype=np.uint8)
            imgs.append(Image.fromarray(px_i))

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for img in imgs:
            emb = extract_hidden(model, processor, img, prompt)
            d = cosine_dist(emb, cal_emb)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        total_time = t1 - t0
        fps = n_images / total_time
        throughput[str(n_images)] = {
            'total_time_ms': float(total_time * 1000),
            'fps': float(fps),
            'ms_per_image': float(total_time / n_images * 1000),
        }
        print(f"  {n_images} images: {fps:.2f} FPS, {total_time/n_images*1000:.2f} ms/img")

    results['throughput'] = throughput

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/latency_benchmark_{ts}.json"
    def convert(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)): return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return obj
    def recursive_convert(d):
        if isinstance(d, dict): return {k: recursive_convert(v) for k, v in d.items()}
        if isinstance(d, list): return [recursive_convert(x) for x in d]
        return convert(d)
    results = recursive_convert(results)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
