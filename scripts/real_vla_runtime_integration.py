#!/usr/bin/env python3
"""Experiment 312: Runtime Integration & End-to-End Pipeline
Simulates a complete deployment pipeline:
1. Calibration phase: compute centroid from clean images
2. Online detection phase: process continuous frame stream
3. Alert system: trigger warnings at configurable thresholds
4. Latency breakdown: per-component timing
5. Memory footprint: storage requirements for deployment
"""

import torch
import numpy as np
import json
import time
from datetime import datetime
from PIL import Image, ImageFilter
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
    t0 = time.time()
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()
    model_load_time = time.time() - t0
    print(f"  Model loaded in {model_load_time:.1f}s")

    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    np.random.seed(42)
    pixels = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    base_img = Image.fromarray(pixels)

    results = {
        "experiment": "runtime_integration",
        "experiment_number": 312,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "model_load_time_s": model_load_time,
    }

    # Part 1: Calibration Phase
    print("\n=== Part 1: Calibration Phase ===")
    cal_times = []
    cal_embeddings = []

    for i in range(5):
        t0 = time.time()
        inputs = processor(prompt, base_img).to(model.device, dtype=torch.bfloat16)
        t_preprocess = time.time() - t0

        t0 = time.time()
        with torch.no_grad():
            fwd = model(**inputs, output_hidden_states=True)
        t_forward = time.time() - t0

        t0 = time.time()
        emb = fwd.hidden_states[3][0, -1, :].float().cpu().numpy()
        t_extract = time.time() - t0

        cal_embeddings.append(emb)
        cal_times.append({
            "preprocess_ms": t_preprocess * 1000,
            "forward_ms": t_forward * 1000,
            "extract_ms": t_extract * 1000,
            "total_ms": (t_preprocess + t_forward + t_extract) * 1000,
        })

    centroid = np.mean(cal_embeddings, axis=0)

    # Compute centroid
    t0 = time.time()
    centroid = np.mean(cal_embeddings, axis=0)
    t_centroid = (time.time() - t0) * 1000

    results["calibration"] = {
        "n_images": 5,
        "per_image_times": cal_times,
        "mean_time_ms": float(np.mean([t["total_ms"] for t in cal_times])),
        "centroid_compute_ms": t_centroid,
        "total_cal_time_ms": float(sum(t["total_ms"] for t in cal_times) + t_centroid),
        "centroid_storage_bytes": centroid.nbytes,
    }
    print(f"  Calibration: {results['calibration']['total_cal_time_ms']:.1f}ms total, "
          f"centroid storage: {centroid.nbytes} bytes")

    # Part 2: Online Detection Phase
    print("\n=== Part 2: Online Detection Phase ===")
    corruptions = ['fog', 'night', 'blur', 'noise']

    # Simulate 100-frame stream
    stream_frames = []
    # 30 clean, 20 fog, 10 night, 10 blur, 10 noise, 20 clean recovery
    for i in range(30):
        stream_frames.append(('clean', base_img, 0))
    for i in range(20):
        sev = (i + 1) / 20
        stream_frames.append(('fog', apply_corruption(base_img, 'fog', sev), sev))
    for i in range(10):
        sev = (i + 1) / 10
        stream_frames.append(('night', apply_corruption(base_img, 'night', sev), sev))
    for i in range(10):
        sev = (i + 1) / 10
        stream_frames.append(('blur', apply_corruption(base_img, 'blur', sev), sev))
    for i in range(10):
        sev = (i + 1) / 10
        stream_frames.append(('noise', apply_corruption(base_img, 'noise', sev), sev))
    for i in range(20):
        stream_frames.append(('clean', base_img, 0))

    detection_log = []
    frame_times = []

    for frame_idx, (condition, img, severity) in enumerate(stream_frames):
        t0 = time.time()

        # Preprocess
        inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)
        t_preprocess = time.time() - t0

        # Forward pass (shared with action inference)
        t0 = time.time()
        with torch.no_grad():
            fwd = model(**inputs, output_hidden_states=True)
        t_forward = time.time() - t0

        # Extract embedding + distance (detector overhead)
        t0 = time.time()
        emb = fwd.hidden_states[3][0, -1, :].float().cpu().numpy()
        d = float(cosine(centroid, emb))
        t_detect = time.time() - t0

        # Classification
        is_ood = d > 0
        alert_level = 'safe' if d == 0 else ('warning' if d < 0.001 else 'danger')

        detection_log.append({
            "frame": frame_idx,
            "condition": condition,
            "severity": severity,
            "distance": d,
            "is_ood": is_ood,
            "alert_level": alert_level,
        })

        frame_times.append({
            "preprocess_ms": t_preprocess * 1000,
            "forward_ms": t_forward * 1000,
            "detect_overhead_ms": t_detect * 1000,
            "total_ms": (t_preprocess + t_forward + t_detect) * 1000,
        })

        if frame_idx % 20 == 0 or condition != detection_log[max(0, frame_idx-1)]["condition"]:
            print(f"  Frame {frame_idx}: {condition} (sev={severity:.2f}), "
                  f"d={d:.6f}, alert={alert_level}")

    results["detection_stream"] = {
        "n_frames": len(stream_frames),
        "log": detection_log,
        "timing": {
            "mean_preprocess_ms": float(np.mean([t["preprocess_ms"] for t in frame_times])),
            "mean_forward_ms": float(np.mean([t["forward_ms"] for t in frame_times])),
            "mean_detect_overhead_ms": float(np.mean([t["detect_overhead_ms"] for t in frame_times])),
            "mean_total_ms": float(np.mean([t["total_ms"] for t in frame_times])),
            "detect_pct_of_total": float(np.mean([t["detect_overhead_ms"] for t in frame_times]) /
                                         np.mean([t["total_ms"] for t in frame_times]) * 100),
            "fps": float(1000 / np.mean([t["total_ms"] for t in frame_times])),
        },
    }

    # Detection accuracy
    tp = sum(1 for d in detection_log if d["condition"] != 'clean' and d["is_ood"])
    tn = sum(1 for d in detection_log if d["condition"] == 'clean' and not d["is_ood"])
    fp = sum(1 for d in detection_log if d["condition"] == 'clean' and d["is_ood"])
    fn = sum(1 for d in detection_log if d["condition"] != 'clean' and not d["is_ood"])

    results["detection_stream"]["accuracy"] = {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
    }

    print(f"\n  Stream results: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"  Sensitivity={results['detection_stream']['accuracy']['sensitivity']:.3f}, "
          f"Specificity={results['detection_stream']['accuracy']['specificity']:.3f}")
    print(f"  Timing: {results['detection_stream']['timing']['mean_total_ms']:.1f}ms/frame, "
          f"{results['detection_stream']['timing']['fps']:.1f} FPS, "
          f"detect overhead={results['detection_stream']['timing']['mean_detect_overhead_ms']:.3f}ms "
          f"({results['detection_stream']['timing']['detect_pct_of_total']:.2f}%)")

    # Part 3: Memory Footprint
    print("\n=== Part 3: Memory Footprint ===")
    memory = {
        "centroid_bytes": centroid.nbytes,
        "centroid_dims": len(centroid),
        "compressed_32d_bytes": 32 * 4,  # 32 floats
        "compressed_4d_bytes": 4 * 4,    # 4 floats
        "projection_matrix_bytes": 4096 * 32 * 4,  # for random projection
    }
    results["memory"] = memory
    print(f"  Full centroid: {centroid.nbytes} bytes ({centroid.nbytes/1024:.1f} KB)")
    print(f"  32D compressed: {32*4} bytes")
    print(f"  4D compressed: {4*4} bytes")

    # Part 4: Threshold Optimization
    print("\n=== Part 4: Threshold Optimization ===")
    clean_dists = [d["distance"] for d in detection_log if d["condition"] == "clean"]
    ood_dists = [d["distance"] for d in detection_log if d["condition"] != "clean"]

    threshold_analysis = {
        "clean_max": float(max(clean_dists)),
        "clean_mean": float(np.mean(clean_dists)),
        "ood_min": float(min(ood_dists)) if ood_dists else 0,
        "ood_mean": float(np.mean(ood_dists)) if ood_dists else 0,
        "gap": float(min(ood_dists) - max(clean_dists)) if ood_dists else 0,
        "optimal_threshold": float((max(clean_dists) + min(ood_dists)) / 2) if ood_dists else 0,
    }
    results["threshold_analysis"] = threshold_analysis
    print(f"  Clean max: {threshold_analysis['clean_max']:.8f}")
    print(f"  OOD min: {threshold_analysis['ood_min']:.8f}")
    print(f"  Gap: {threshold_analysis['gap']:.8f}")

    # Save
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(v) for v in obj]
        return obj

    ts = results["timestamp"]
    out_path = f"experiments/runtime_{ts}.json"
    with open(out_path, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
