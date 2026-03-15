#!/usr/bin/env python3
"""Experiment 434: Detection Latency & Computational Cost

Measures practical deployment characteristics: inference time,
memory footprint, and detection overhead. How much does OOD
detection add to the VLA inference pipeline?

Tests:
1. Single-image inference time (mean of 20 runs)
2. OOD detection overhead (hidden state extraction vs full generation)
3. Batch processing efficiency
4. Early exit potential (can we stop at layer 3?)
5. Detection precision at reduced resolution
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

def compute_auroc(id_scores, ood_scores):
    id_s = np.asarray(id_scores, dtype=np.float64)
    ood_s = np.asarray(ood_scores, dtype=np.float64)
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
    scene = Image.fromarray(np.random.RandomState(42).randint(0, 255, (224, 224, 3), dtype=np.uint8))

    results = {}

    # Warmup
    print("Warming up...")
    for _ in range(3):
        inputs = processor(prompt, scene).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            model(**inputs, output_hidden_states=True)
    torch.cuda.synchronize()

    # === Test 1: Forward pass timing ===
    print("\n=== Forward Pass Timing ===")
    n_runs = 20

    # (a) Full forward with hidden states
    times_hidden = []
    for _ in range(n_runs):
        inputs = processor(prompt, scene).to(model.device, dtype=torch.bfloat16)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            fwd = model(**inputs, output_hidden_states=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_hidden.append(t1 - t0)

    # (b) Full forward without hidden states
    times_no_hidden = []
    for _ in range(n_runs):
        inputs = processor(prompt, scene).to(model.device, dtype=torch.bfloat16)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            fwd = model(**inputs, output_hidden_states=False)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_no_hidden.append(t1 - t0)

    # (c) Full generation (7 action tokens)
    times_generate = []
    for _ in range(min(10, n_runs)):
        inputs = processor(prompt, scene).to(model.device, dtype=torch.bfloat16)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=7, do_sample=False)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_generate.append(t1 - t0)

    results["timing"] = {
        "forward_with_hidden": {
            "mean_ms": float(np.mean(times_hidden) * 1000),
            "std_ms": float(np.std(times_hidden) * 1000),
            "min_ms": float(np.min(times_hidden) * 1000),
        },
        "forward_no_hidden": {
            "mean_ms": float(np.mean(times_no_hidden) * 1000),
            "std_ms": float(np.std(times_no_hidden) * 1000),
            "min_ms": float(np.min(times_no_hidden) * 1000),
        },
        "generation_7_tokens": {
            "mean_ms": float(np.mean(times_generate) * 1000),
            "std_ms": float(np.std(times_generate) * 1000),
            "min_ms": float(np.min(times_generate) * 1000),
        },
        "hidden_state_overhead_ms": float((np.mean(times_hidden) - np.mean(times_no_hidden)) * 1000),
        "hidden_state_overhead_pct": float((np.mean(times_hidden) / np.mean(times_no_hidden) - 1) * 100),
        "detection_vs_generation_pct": float(np.mean(times_hidden) / np.mean(times_generate) * 100),
    }
    print(f"  Forward + hidden: {np.mean(times_hidden)*1000:.1f} ± {np.std(times_hidden)*1000:.1f} ms")
    print(f"  Forward no hidden: {np.mean(times_no_hidden)*1000:.1f} ± {np.std(times_no_hidden)*1000:.1f} ms")
    print(f"  Generation (7 tok): {np.mean(times_generate)*1000:.1f} ± {np.std(times_generate)*1000:.1f} ms")
    print(f"  Hidden state overhead: {results['timing']['hidden_state_overhead_ms']:.1f} ms ({results['timing']['hidden_state_overhead_pct']:.1f}%)")

    # === Test 2: Detection computation time ===
    print("\n=== Detection Computation Time ===")
    centroid = np.random.RandomState(42).randn(4096)

    # Time cosine distance computation
    emb = np.random.RandomState(123).randn(4096)
    n_detect = 10000
    t0 = time.perf_counter()
    for _ in range(n_detect):
        cosine_dist(emb, centroid)
    t1 = time.perf_counter()
    detect_time_us = (t1 - t0) / n_detect * 1e6

    results["detection_compute"] = {
        "cosine_dist_us": float(detect_time_us),
        "centroid_size_bytes": int(centroid.nbytes),
    }
    print(f"  Cosine distance: {detect_time_us:.1f} µs")
    print(f"  Centroid storage: {centroid.nbytes} bytes")

    # === Test 3: Memory analysis ===
    print("\n=== Memory Analysis ===")
    torch.cuda.synchronize()
    mem_allocated = torch.cuda.memory_allocated() / 1024**3
    mem_reserved = torch.cuda.memory_reserved() / 1024**3
    mem_max = torch.cuda.max_memory_allocated() / 1024**3

    results["memory"] = {
        "allocated_gb": float(mem_allocated),
        "reserved_gb": float(mem_reserved),
        "peak_gb": float(mem_max),
    }
    print(f"  Allocated: {mem_allocated:.2f} GB")
    print(f"  Reserved: {mem_reserved:.2f} GB")
    print(f"  Peak: {mem_max:.2f} GB")

    # === Test 4: Resolution sensitivity ===
    print("\n=== Resolution Sensitivity ===")
    seeds = [42, 123, 456, 789, 999]
    scenes_full = [Image.fromarray(np.random.RandomState(s).randint(0, 255, (224, 224, 3), dtype=np.uint8)) for s in seeds]

    # Full resolution baseline
    clean_embs_full = []
    for s in scenes_full:
        inputs = processor(prompt, s).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            fwd = model(**inputs, output_hidden_states=True)
        clean_embs_full.append(fwd.hidden_states[3][0, -1, :].float().cpu().numpy())
    centroid_full = np.mean(clean_embs_full, axis=0)
    clean_dists_full = [cosine_dist(e, centroid_full) for e in clean_embs_full]

    resolution_results = {}
    for res in [224, 112, 56, 28]:
        clean_embs_res = []
        for s in scenes_full:
            # Resize to lower res then back to 224
            s_low = s.resize((res, res), Image.BILINEAR).resize((224, 224), Image.BILINEAR)
            inputs = processor(prompt, s_low).to(model.device, dtype=torch.bfloat16)
            with torch.no_grad():
                fwd = model(**inputs, output_hidden_states=True)
            clean_embs_res.append(fwd.hidden_states[3][0, -1, :].float().cpu().numpy())

        centroid_res = np.mean(clean_embs_res, axis=0)
        clean_dists_res = [cosine_dist(e, centroid_res) for e in clean_embs_res]

        # Test detection on fog corruption
        ood_dists_res = []
        for s in scenes_full:
            s_low = s.resize((res, res), Image.BILINEAR).resize((224, 224), Image.BILINEAR)
            fog_img = apply_corruption(s_low, 'fog')
            inputs = processor(prompt, fog_img).to(model.device, dtype=torch.bfloat16)
            with torch.no_grad():
                fwd = model(**inputs, output_hidden_states=True)
            emb = fwd.hidden_states[3][0, -1, :].float().cpu().numpy()
            ood_dists_res.append(float(cosine_dist(emb, centroid_res)))

        auroc = float(compute_auroc(clean_dists_res, ood_dists_res))
        # Similarity to full-res centroid
        centroid_sim = float(1 - cosine_dist(centroid_res, centroid_full))

        resolution_results[str(res)] = {
            "auroc_fog": auroc,
            "centroid_similarity": centroid_sim,
            "mean_clean_dist": float(np.mean(clean_dists_res)),
        }
        print(f"  {res}×{res}: AUROC(fog)={auroc:.4f}, centroid_sim={centroid_sim:.6f}")
    results["resolution"] = resolution_results

    # === Test 5: Throughput estimation ===
    print("\n=== Throughput Estimation ===")
    fwd_ms = np.mean(times_hidden) * 1000
    gen_ms = np.mean(times_generate) * 1000
    results["throughput"] = {
        "detection_fps": float(1000 / fwd_ms),
        "generation_fps": float(1000 / gen_ms),
        "detection_overhead_vs_generation_ms": float(fwd_ms),
        "total_pipeline_ms": float(fwd_ms),  # Detection is a single forward pass
        "pipeline_fps": float(1000 / fwd_ms),
    }
    print(f"  Detection: {1000/fwd_ms:.1f} FPS")
    print(f"  Generation: {1000/gen_ms:.1f} FPS")
    print(f"  Detection is {fwd_ms/gen_ms*100:.0f}% of generation cost")

    out_path = "/workspace/Vizuara-VLA-Research/experiments/detection_latency_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
