#!/usr/bin/env python3
"""Experiment 373: Temporal Sequence Detection Analysis

How does the detector behave when corruptions appear/disappear over time?
1. Sudden onset: clean→corrupt transition detection latency
2. Gradual onset: slow corruption ramp detection threshold
3. Intermittent corruption: flickering corruption pattern
4. Recovery dynamics: corrupt→clean transition
5. Sequence-level statistics: running average, CUSUM, EMA detectors
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
    ctypes = ['fog', 'night', 'noise', 'blur']

    # Generate test images
    print("Generating images...")
    seeds = list(range(0, 500, 100))[:5]
    images = {}
    clean_embs = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        px = rng.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        images[seed] = Image.fromarray(px)
        clean_embs[seed] = extract_hidden(model, processor, images[seed], prompt)

    centroid = np.mean(list(clean_embs.values()), axis=0)
    clean_dists = [cosine_dist(centroid, clean_embs[s]) for s in seeds]
    threshold = max(clean_dists)
    print(f"  Detection threshold: {threshold:.6f}")

    # ========== 1. Sudden Onset: Clean→Corrupt Transition ==========
    print("\n=== Sudden Onset Detection ===")

    sudden_onset = {}
    for ct in ctypes:
        for seed in seeds[:3]:
            sequence_dists = []
            sequence_labels = []

            for t in range(10):
                if t < 5:
                    emb = clean_embs[seed]
                    sequence_labels.append('clean')
                else:
                    corrupt_img = apply_corruption(images[seed], ct, 0.5)
                    emb = extract_hidden(model, processor, corrupt_img, prompt)
                    sequence_labels.append('corrupt')

                d = cosine_dist(emb, centroid)
                sequence_dists.append(d)

            onset_frame = 5
            detection_frame = None
            for t in range(onset_frame, len(sequence_dists)):
                if sequence_dists[t] > threshold:
                    detection_frame = t
                    break

            latency = (detection_frame - onset_frame) if detection_frame is not None else -1

            key = f"{ct}_seed{seed}"
            sudden_onset[key] = {
                'sequence_dists': [float(d) for d in sequence_dists],
                'onset_frame': onset_frame,
                'detection_frame': detection_frame,
                'detection_latency': latency,
                'clean_mean_dist': float(np.mean(sequence_dists[:5])),
                'corrupt_mean_dist': float(np.mean(sequence_dists[5:])),
                'jump_ratio': float(np.mean(sequence_dists[5:]) / max(np.mean(sequence_dists[:5]), 1e-10)),
            }
            print(f"  {ct}_s{seed}: latency={latency} frames, "
                  f"jump_ratio={sudden_onset[key]['jump_ratio']:.2f}x")

    results['sudden_onset'] = sudden_onset

    # ========== 2. Gradual Onset: Slow Corruption Ramp ==========
    print("\n=== Gradual Onset (Severity Ramp) ===")

    gradual_onset = {}
    for ct in ctypes:
        seed = seeds[0]
        severities = np.linspace(0, 1.0, 20)
        ramp_dists = []
        ramp_detected = []

        for sev in severities:
            if sev == 0:
                emb = clean_embs[seed]
            else:
                corrupt_img = apply_corruption(images[seed], ct, sev)
                emb = extract_hidden(model, processor, corrupt_img, prompt)
            d = cosine_dist(emb, centroid)
            ramp_dists.append(d)
            ramp_detected.append(d > threshold)

        detection_severity = None
        for i, sev in enumerate(severities):
            if ramp_detected[i]:
                detection_severity = float(sev)
                break

        gradual_onset[ct] = {
            'severities': [float(s) for s in severities],
            'distances': [float(d) for d in ramp_dists],
            'detected': [bool(d) for d in ramp_detected],
            'detection_severity': detection_severity,
            'max_dist': float(max(ramp_dists)),
            'monotonic': all(ramp_dists[i] <= ramp_dists[i+1] for i in range(len(ramp_dists)-1)),
        }
        print(f"  {ct}: detection_severity={detection_severity}, "
              f"monotonic={gradual_onset[ct]['monotonic']}, max_dist={max(ramp_dists):.6f}")

    results['gradual_onset'] = gradual_onset

    # ========== 3. Intermittent Corruption (Flickering) ==========
    print("\n=== Intermittent Corruption ===")

    intermittent = {}
    for ct in ctypes:
        seed = seeds[0]
        pattern_dists = []
        pattern_labels = []

        for t in range(20):
            if t % 2 == 0:
                emb = clean_embs[seed]
                pattern_labels.append('clean')
            else:
                corrupt_img = apply_corruption(images[seed], ct, 0.5)
                emb = extract_hidden(model, processor, corrupt_img, prompt)
                pattern_labels.append('corrupt')
            d = cosine_dist(emb, centroid)
            pattern_dists.append(d)

        clean_frames = [pattern_dists[t] for t in range(20) if t % 2 == 0]
        corrupt_frames = [pattern_dists[t] for t in range(20) if t % 2 == 1]

        # Running average detector (window=3)
        running_avg = []
        for t in range(len(pattern_dists)):
            window = pattern_dists[max(0, t-2):t+1]
            running_avg.append(float(np.mean(window)))

        # CUSUM detector
        cusum = [0.0]
        cusum_threshold = threshold * 2
        cusum_detections = []
        for t in range(1, len(pattern_dists)):
            s = max(0, cusum[-1] + (pattern_dists[t] - threshold))
            cusum.append(s)
            if s > cusum_threshold:
                cusum_detections.append(t)

        intermittent[ct] = {
            'pattern_dists': [float(d) for d in pattern_dists],
            'clean_frame_mean': float(np.mean(clean_frames)),
            'corrupt_frame_mean': float(np.mean(corrupt_frames)),
            'running_avg': running_avg,
            'running_avg_detections': sum(1 for r in running_avg if r > threshold),
            'cusum_values': [float(c) for c in cusum],
            'cusum_detections': cusum_detections,
            'instant_detection_rate': sum(1 for d in corrupt_frames if d > threshold) / len(corrupt_frames),
        }
        print(f"  {ct}: instant_det_rate={intermittent[ct]['instant_detection_rate']:.2f}, "
              f"running_avg_dets={intermittent[ct]['running_avg_detections']}/20, "
              f"cusum_dets={len(cusum_detections)}")

    results['intermittent'] = intermittent

    # ========== 4. Recovery Dynamics: Corrupt→Clean Transition ==========
    print("\n=== Recovery Dynamics ===")

    recovery = {}
    for ct in ctypes:
        for seed in seeds[:3]:
            sequence_dists = []
            for t in range(10):
                if t < 5:
                    corrupt_img = apply_corruption(images[seed], ct, 0.5)
                    emb = extract_hidden(model, processor, corrupt_img, prompt)
                else:
                    emb = clean_embs[seed]
                d = cosine_dist(emb, centroid)
                sequence_dists.append(d)

            recovery_frame = None
            for t in range(5, len(sequence_dists)):
                if sequence_dists[t] <= threshold:
                    recovery_frame = t
                    break

            recovery_latency = (recovery_frame - 5) if recovery_frame is not None else -1

            key = f"{ct}_seed{seed}"
            recovery[key] = {
                'sequence_dists': [float(d) for d in sequence_dists],
                'recovery_frame': recovery_frame,
                'recovery_latency': recovery_latency,
                'corrupt_mean_dist': float(np.mean(sequence_dists[:5])),
                'clean_mean_dist': float(np.mean(sequence_dists[5:])),
            }
            print(f"  {ct}_s{seed}: recovery_latency={recovery_latency} frames")

    results['recovery'] = recovery

    # ========== 5. Sequence-Level Statistics ==========
    print("\n=== Sequence-Level Detection Comparison ===")

    seq_stats = {}
    for ct in ctypes:
        seed = seeds[0]
        long_seq_dists = []
        long_seq_labels = []

        for t in range(30):
            if t < 10 or t >= 20:
                emb = clean_embs[seed]
                long_seq_labels.append('clean')
            else:
                corrupt_img = apply_corruption(images[seed], ct, 0.5)
                emb = extract_hidden(model, processor, corrupt_img, prompt)
                long_seq_labels.append('corrupt')
            d = cosine_dist(emb, centroid)
            long_seq_dists.append(d)

        # Detector 1: Instant threshold
        instant_tp = sum(1 for t in range(10, 20) if long_seq_dists[t] > threshold)
        instant_fp = sum(1 for t in list(range(10)) + list(range(20, 30)) if long_seq_dists[t] > threshold)

        # Detector 2: Running average (window=5)
        running5 = []
        for t in range(len(long_seq_dists)):
            window = long_seq_dists[max(0, t-4):t+1]
            running5.append(float(np.mean(window)))
        ra5_tp = sum(1 for t in range(10, 20) if running5[t] > threshold)
        ra5_fp = sum(1 for t in list(range(10)) + list(range(20, 30)) if running5[t] > threshold)

        # Detector 3: CUSUM
        cusum = [0.0]
        for t in range(1, len(long_seq_dists)):
            s = max(0, cusum[-1] + (long_seq_dists[t] - threshold))
            cusum.append(s)
        cusum_thresh = threshold * 3
        cusum_tp = sum(1 for t in range(10, 20) if cusum[t] > cusum_thresh)
        cusum_fp = sum(1 for t in list(range(10)) + list(range(20, 30)) if cusum[t] > cusum_thresh)

        # Detector 4: Exponential moving average
        ema = [long_seq_dists[0]]
        alpha = 0.3
        for t in range(1, len(long_seq_dists)):
            ema.append(alpha * long_seq_dists[t] + (1 - alpha) * ema[-1])
        ema_tp = sum(1 for t in range(10, 20) if ema[t] > threshold)
        ema_fp = sum(1 for t in list(range(10)) + list(range(20, 30)) if ema[t] > threshold)

        seq_stats[ct] = {
            'sequence_dists': [float(d) for d in long_seq_dists],
            'instant': {'tp': instant_tp, 'fp': instant_fp, 'sensitivity': instant_tp / 10, 'specificity': 1 - instant_fp / 20},
            'running_avg5': {'tp': ra5_tp, 'fp': ra5_fp, 'sensitivity': ra5_tp / 10, 'specificity': 1 - ra5_fp / 20},
            'cusum': {'tp': cusum_tp, 'fp': cusum_fp, 'sensitivity': cusum_tp / 10, 'specificity': 1 - cusum_fp / 20},
            'ema': {'tp': ema_tp, 'fp': ema_fp, 'sensitivity': ema_tp / 10, 'specificity': 1 - ema_fp / 20},
            'running_avg5_values': running5,
            'cusum_values': [float(c) for c in cusum],
            'ema_values': [float(e) for e in ema],
        }
        print(f"  {ct}: instant={instant_tp}/10 TP {instant_fp}/20 FP, "
              f"RA5={ra5_tp}/10 TP {ra5_fp}/20 FP, "
              f"CUSUM={cusum_tp}/10 TP {cusum_fp}/20 FP, "
              f"EMA={ema_tp}/10 TP {ema_fp}/20 FP")

    results['sequence_detectors'] = seq_stats

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/temporal_sequence_{ts}.json"
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
