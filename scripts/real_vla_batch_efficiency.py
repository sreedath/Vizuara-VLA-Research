#!/usr/bin/env python3
"""Experiment 393: Batch Processing Efficiency and Scalability

Measures detection throughput under batch processing, tests how detection
scales with number of calibration samples, and profiles the complete
detection pipeline latency breakdown.

Tests:
1. Latency breakdown: embedding extraction vs cosine distance
2. Batch embedding extraction throughput
3. Detection AUROC vs number of calibration frames (1-50)
4. Centroid convergence rate
5. Memory footprint of detection system
6. End-to-end detection latency at deployment
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
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    return 1.0 - np.dot(a, b) / (na * nb)

def compute_auroc(id_scores, ood_scores):
    id_s = np.asarray(id_scores)
    ood_s = np.asarray(ood_scores)
    n_id, n_ood = len(id_s), len(ood_s)
    if n_id == 0 or n_ood == 0:
        return 0.5
    count = sum(float(np.sum(o > id_s) + 0.5 * np.sum(o == id_s)) for o in ood_s)
    return count / (n_id * n_ood)

def main():
    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    img = Image.fromarray(np.random.RandomState(42).randint(0, 255, (224, 224, 3), dtype=np.uint8))

    corruptions = ['fog', 'night', 'noise', 'blur']

    # 1. Latency breakdown
    print("=== Latency Breakdown ===")
    # Warm up
    _ = extract_hidden(model, processor, img, prompt)

    n_trials = 20
    embed_times = []
    for i in range(n_trials):
        t0 = time.time()
        emb = extract_hidden(model, processor, img, prompt)
        embed_times.append(time.time() - t0)

    centroid = emb  # Use as centroid for timing
    cosine_times = []
    for i in range(10000):
        t0 = time.time()
        cosine_dist(emb, centroid)
        cosine_times.append(time.time() - t0)

    results = {
        "latency_breakdown": {
            "embed_mean_ms": float(np.mean(embed_times) * 1000),
            "embed_std_ms": float(np.std(embed_times) * 1000),
            "embed_min_ms": float(np.min(embed_times) * 1000),
            "embed_max_ms": float(np.max(embed_times) * 1000),
            "cosine_mean_us": float(np.mean(cosine_times) * 1e6),
            "cosine_std_us": float(np.std(cosine_times) * 1e6),
            "total_mean_ms": float(np.mean(embed_times) * 1000 + np.mean(cosine_times) * 1000),
            "cosine_fraction": float(np.mean(cosine_times) / np.mean(embed_times)),
            "overhead_percent": float(np.mean(cosine_times) / np.mean(embed_times) * 100)
        }
    }
    print(f"  Embedding: {np.mean(embed_times)*1000:.2f} ± {np.std(embed_times)*1000:.2f} ms")
    print(f"  Cosine: {np.mean(cosine_times)*1e6:.1f} ± {np.std(cosine_times)*1e6:.1f} μs")
    print(f"  Overhead: {np.mean(cosine_times)/np.mean(embed_times)*100:.4f}%")

    # 2. Collect a large pool of embeddings for calibration scaling
    print("\n=== Collecting Embedding Pool ===")
    n_pool = 50
    clean_pool = []
    for i in range(n_pool):
        arr = np.array(img).astype(np.float32)
        arr += np.random.RandomState(100 + i).randn(*arr.shape) * 0.5
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        emb = extract_hidden(model, processor, Image.fromarray(arr), prompt)
        clean_pool.append(emb)
        if (i + 1) % 10 == 0:
            print(f"  Clean {i+1}/{n_pool}")

    corrupt_pool = {}
    for c in corruptions:
        corrupt_pool[c] = []
        for i in range(10):
            arr = np.array(img).astype(np.float32)
            arr += np.random.RandomState(200 + i).randn(*arr.shape) * 0.5
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            corrupted = apply_corruption(Image.fromarray(arr), c)
            emb = extract_hidden(model, processor, corrupted, prompt)
            corrupt_pool[c].append(emb)
        print(f"  {c}: 10 samples")

    # 3. AUROC vs calibration samples
    print("\n=== AUROC vs Calibration Samples ===")
    cal_sizes = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50]
    auroc_scaling = {}
    for n_cal in cal_sizes:
        aurocs = {}
        centroid = np.mean(clean_pool[:n_cal], axis=0)
        test_clean = clean_pool[n_cal:] if n_cal < len(clean_pool) else clean_pool
        for c in corruptions:
            id_scores = [cosine_dist(e, centroid) for e in test_clean]
            ood_scores = [cosine_dist(e, centroid) for e in corrupt_pool[c]]
            aurocs[c] = compute_auroc(id_scores, ood_scores)
        auroc_scaling[n_cal] = aurocs
        print(f"  n={n_cal}: {aurocs}")
    results["auroc_scaling"] = auroc_scaling

    # 4. Centroid convergence
    print("\n=== Centroid Convergence ===")
    final_centroid = np.mean(clean_pool, axis=0)
    convergence = []
    for n in range(1, n_pool + 1):
        partial_centroid = np.mean(clean_pool[:n], axis=0)
        dist_to_final = cosine_dist(partial_centroid, final_centroid)
        convergence.append({"n": n, "dist_to_final": float(dist_to_final)})
    results["centroid_convergence"] = convergence

    # Find n where convergence < 1e-6
    converge_n = n_pool
    for entry in convergence:
        if entry["dist_to_final"] < 1e-6:
            converge_n = entry["n"]
            break
    results["convergence_threshold_1e6"] = converge_n
    print(f"  Converges (< 1e-6) at n={converge_n}")

    # 5. Memory footprint
    print("\n=== Memory Footprint ===")
    emb_size = clean_pool[0].nbytes  # 4096 * 4 bytes
    centroid_size = emb_size
    results["memory"] = {
        "single_embedding_bytes": int(emb_size),
        "centroid_bytes": int(centroid_size),
        "detection_system_total_bytes": int(centroid_size),  # Just store centroid
        "detection_system_kb": float(centroid_size / 1024),
        "model_size_gb_approx": 14.0,  # 7B params * 2 bytes (bf16)
        "detection_fraction_of_model": float(centroid_size / (14 * 1024**3))
    }
    print(f"  Centroid: {centroid_size} bytes ({centroid_size/1024:.1f} KB)")
    print(f"  Fraction of model: {centroid_size / (14 * 1024**3):.2e}")

    # 6. End-to-end deployment timing
    print("\n=== End-to-End Deployment ===")
    centroid_deploy = np.mean(clean_pool[:10], axis=0)
    threshold = max(cosine_dist(e, centroid_deploy) for e in clean_pool[:10]) * 1.1

    e2e_times = []
    e2e_decisions = []
    for i in range(10):
        test_img = apply_corruption(img, 'fog', severity=0.5)
        t0 = time.time()
        emb = extract_hidden(model, processor, test_img, prompt)
        dist = cosine_dist(emb, centroid_deploy)
        decision = dist > threshold
        elapsed = time.time() - t0
        e2e_times.append(elapsed)
        e2e_decisions.append(bool(decision))

    results["e2e_deployment"] = {
        "mean_latency_ms": float(np.mean(e2e_times) * 1000),
        "std_latency_ms": float(np.std(e2e_times) * 1000),
        "all_detected": all(e2e_decisions),
        "detection_rate": float(np.mean(e2e_decisions))
    }
    print(f"  E2E: {np.mean(e2e_times)*1000:.2f} ± {np.std(e2e_times)*1000:.2f} ms")
    print(f"  All detected: {all(e2e_decisions)}")

    # 7. Detection at different model inference stages
    print("\n=== Detection Timing Within Inference ===")
    # The key insight: we can detect DURING inference (hidden states available
    # before generation completes)
    inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)

    # Time just the forward pass (no generation)
    fwd_times = []
    for _ in range(10):
        t0 = time.time()
        with torch.no_grad():
            fwd = model(**inputs, output_hidden_states=True)
        fwd_times.append(time.time() - t0)

    # Time generation
    gen_times = []
    for _ in range(10):
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=7, do_sample=False)
        gen_times.append(time.time() - t0)

    results["inference_timing"] = {
        "forward_only_ms": float(np.mean(fwd_times) * 1000),
        "generation_ms": float(np.mean(gen_times) * 1000),
        "generation_overhead_ms": float((np.mean(gen_times) - np.mean(fwd_times)) * 1000),
        "detection_available_at_fraction": float(np.mean(fwd_times) / np.mean(gen_times))
    }
    print(f"  Forward: {np.mean(fwd_times)*1000:.2f} ms")
    print(f"  Generation: {np.mean(gen_times)*1000:.2f} ms")
    print(f"  Detection available at {np.mean(fwd_times)/np.mean(gen_times)*100:.1f}% of inference")

    # Save
    out_path = "/workspace/Vizuara-VLA-Research/experiments/batch_efficiency_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
