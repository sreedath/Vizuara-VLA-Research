#!/usr/bin/env python3
"""Experiment 347: Temporal Stability and Sequential Detection

How stable is detection over sequences of frames?
1. Frame-to-frame embedding consistency (clean sequence)
2. Gradual corruption onset (fog/night ramp-up over frames)
3. Intermittent corruption (every Nth frame corrupted)
4. Detection latency (how many consecutive corrupted frames before alarm?)
5. EWMA (exponentially weighted moving average) detector vs single-frame
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

    # Base scene
    np.random.seed(42)
    pixels = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    base_img = Image.fromarray(pixels)
    cal_emb = extract_hidden(model, processor, base_img, prompt)

    ctypes = ['fog', 'night', 'noise', 'blur']

    # ========== 1. Frame-to-frame consistency (clean sequence) ==========
    print("\n=== Clean Sequence Consistency ===")

    # Simulate 20 "frames" — same scene but with tiny random pixel jitter
    frame_embs = []
    frame_dists = []

    for frame_idx in range(20):
        if frame_idx == 0:
            img = base_img
        else:
            # Add sub-pixel jitter (+-1 pixel value)
            rng = np.random.RandomState(1000 + frame_idx)
            jitter = rng.randint(-1, 2, pixels.shape).astype(np.int16)
            jittered = np.clip(pixels.astype(np.int16) + jitter, 0, 255).astype(np.uint8)
            img = Image.fromarray(jittered)

        emb = extract_hidden(model, processor, img, prompt)
        d = cosine_dist(cal_emb, emb)
        frame_embs.append(emb)
        frame_dists.append(float(d))

    # Frame-to-frame distances
    consecutive_dists = []
    for i in range(1, len(frame_embs)):
        consecutive_dists.append(float(cosine_dist(frame_embs[i-1], frame_embs[i])))

    clean_consistency = {
        'n_frames': 20,
        'distances_from_cal': frame_dists,
        'consecutive_distances': consecutive_dists,
        'max_dist_from_cal': float(max(frame_dists)),
        'mean_dist_from_cal': float(np.mean(frame_dists)),
        'max_consecutive': float(max(consecutive_dists)) if consecutive_dists else 0,
        'all_identical': all(d == 0.0 for d in frame_dists),
    }
    print(f"  Max dist from cal: {max(frame_dists):.8f}")
    print(f"  Max consecutive dist: {max(consecutive_dists):.8f}")
    print(f"  All identical to cal: {all(d == 0.0 for d in frame_dists)}")

    results['clean_consistency'] = clean_consistency

    # ========== 2. Gradual corruption onset ==========
    print("\n=== Gradual Corruption Onset ===")

    gradual_results = {}
    for ct in ctypes:
        # Simulate 30 frames: first 10 clean, then severity ramps 0->1 over frames 10-30
        frame_distances = []
        frame_severities = []

        for frame_idx in range(30):
            if frame_idx < 10:
                sev = 0.0
                img = base_img
            else:
                sev = (frame_idx - 10) / 20.0  # 0 to 1 over 20 frames
                img = apply_corruption(base_img, ct, sev)

            emb = extract_hidden(model, processor, img, prompt)
            d = cosine_dist(cal_emb, emb)
            frame_distances.append(float(d))
            frame_severities.append(float(sev))

        # Find first frame where d > various thresholds
        thresholds = [0.0, 1e-6, 1e-5, 1e-4, 1e-3]
        first_detection = {}
        for thresh in thresholds:
            first_frame = None
            for i, d in enumerate(frame_distances):
                if d > thresh:
                    first_frame = i
                    break
            first_detection[str(thresh)] = first_frame

        gradual_results[ct] = {
            'severities': frame_severities,
            'distances': frame_distances,
            'first_detection': first_detection,
            'max_clean_dist': float(max(frame_distances[:10])),
            'min_corrupt_dist': float(min(frame_distances[10:])),
        }
        print(f"  {ct}: first d>0 at frame {first_detection['0.0']}, "
              f"first d>1e-4 at frame {first_detection['0.0001']}")

    results['gradual_onset'] = gradual_results

    # ========== 3. Intermittent corruption ==========
    print("\n=== Intermittent Corruption ===")

    intermittent_results = {}
    for ct in ['fog', 'night']:  # Just two types to save time
        # 40 frames, every Nth frame is corrupted at severity 0.5
        for N in [2, 3, 5, 10]:
            frame_distances = []
            is_corrupt = []

            for frame_idx in range(40):
                if frame_idx % N == 0 and frame_idx > 0:
                    img = apply_corruption(base_img, ct, 0.5)
                    corrupt = True
                else:
                    img = base_img
                    corrupt = False

                emb = extract_hidden(model, processor, img, prompt)
                d = cosine_dist(cal_emb, emb)
                frame_distances.append(float(d))
                is_corrupt.append(corrupt)

            # Detection stats
            corrupt_dists = [d for d, c in zip(frame_distances, is_corrupt) if c]
            clean_dists = [d for d, c in zip(frame_distances, is_corrupt) if not c]

            key = f"{ct}_every_{N}"
            intermittent_results[key] = {
                'n_corrupt': sum(is_corrupt),
                'n_clean': sum(not c for c in is_corrupt),
                'mean_corrupt_dist': float(np.mean(corrupt_dists)) if corrupt_dists else 0,
                'mean_clean_dist': float(np.mean(clean_dists)),
                'all_corrupt_detected': all(d > 0 for d in corrupt_dists),
                'any_false_alarms': any(d > 0 for d in clean_dists),
            }
            print(f"  {key}: corrupt detected={all(d > 0 for d in corrupt_dists)}, "
                  f"false alarms={any(d > 0 for d in clean_dists)}")

    results['intermittent'] = intermittent_results

    # ========== 4. EWMA detector vs single-frame ==========
    print("\n=== EWMA Detector ===")

    ewma_results = {}
    for ct in ctypes:
        # 50 frames: clean(20), corrupt(10), clean(10), corrupt(10)
        frame_distances = []
        ground_truth = []

        for frame_idx in range(50):
            if frame_idx < 20:
                img = base_img
                gt = False
            elif frame_idx < 30:
                img = apply_corruption(base_img, ct, 0.3)  # mild corruption
                gt = True
            elif frame_idx < 40:
                img = base_img
                gt = False
            else:
                img = apply_corruption(base_img, ct, 0.7)  # strong corruption
                gt = True

            emb = extract_hidden(model, processor, img, prompt)
            d = cosine_dist(cal_emb, emb)
            frame_distances.append(float(d))
            ground_truth.append(gt)

        # Single-frame detector at various thresholds
        single_frame = {}
        for thresh in [0.0, 1e-5, 1e-4, 5e-4, 1e-3]:
            preds = [d > thresh for d in frame_distances]
            tp = sum(p and g for p, g in zip(preds, ground_truth))
            fp = sum(p and not g for p, g in zip(preds, ground_truth))
            fn = sum(not p and g for p, g in zip(preds, ground_truth))
            tn = sum(not p and not g for p, g in zip(preds, ground_truth))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            single_frame[str(thresh)] = {
                'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                'precision': float(precision),
                'recall': float(recall),
            }

        # EWMA detector
        ewma_detector = {}
        for alpha in [0.1, 0.3, 0.5, 0.8]:
            ewma = 0.0
            ewma_values = []
            for d in frame_distances:
                ewma = alpha * d + (1 - alpha) * ewma
                ewma_values.append(ewma)

            # Use threshold = mean of clean EWMA + 3*std
            clean_ewma = ewma_values[:20]
            if np.std(clean_ewma) > 0:
                ewma_thresh = np.mean(clean_ewma) + 3 * np.std(clean_ewma)
            else:
                ewma_thresh = np.mean(clean_ewma) + 1e-7

            preds = [e > ewma_thresh for e in ewma_values]
            tp = sum(p and g for p, g in zip(preds, ground_truth))
            fp = sum(p and not g for p, g in zip(preds, ground_truth))
            fn = sum(not p and g for p, g in zip(preds, ground_truth))
            tn = sum(not p and not g for p, g in zip(preds, ground_truth))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            # Detection latency: first corrupt frame (20) to first detection
            first_detect = None
            for i, (p, g) in enumerate(zip(preds, ground_truth)):
                if p and g and i >= 20:
                    first_detect = i - 20  # latency in frames
                    break

            # Recovery latency: first clean after corrupt (30) to false alarm clear
            recovery = None
            for i in range(30, 40):
                if not preds[i]:
                    recovery = i - 30
                    break

            ewma_detector[str(alpha)] = {
                'threshold': float(ewma_thresh),
                'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                'precision': float(precision),
                'recall': float(recall),
                'detection_latency': first_detect,
                'recovery_latency': recovery,
                'ewma_values': ewma_values,
            }

        ewma_results[ct] = {
            'frame_distances': frame_distances,
            'ground_truth': ground_truth,
            'single_frame': single_frame,
            'ewma': ewma_detector,
        }
        print(f"  {ct}: single@d>0 recall={single_frame['0.0']['recall']:.2f}, "
              f"EWMA(0.3) recall={ewma_detector['0.3']['recall']:.2f} "
              f"latency={ewma_detector['0.3']['detection_latency']}")

    results['ewma_detector'] = ewma_results

    # ========== 5. Corruption transition dynamics ==========
    print("\n=== Corruption Transition Dynamics ===")

    transition_results = {}
    for ct in ctypes:
        # Abrupt transition: clean -> corrupt -> clean
        # How quickly does embedding respond?
        transition_dists = []
        for frame_idx in range(20):
            if frame_idx < 5:
                img = base_img
            elif frame_idx < 15:
                img = apply_corruption(base_img, ct, 0.5)
            else:
                img = base_img

            emb = extract_hidden(model, processor, img, prompt)
            d = cosine_dist(cal_emb, emb)
            transition_dists.append(float(d))

        # Rise time: frames from first corrupt to stable corrupt distance
        corrupt_dists = transition_dists[5:15]
        stable_corrupt = np.mean(corrupt_dists[-3:])

        rise_frame = None
        for i, d in enumerate(corrupt_dists):
            if d >= 0.9 * stable_corrupt:
                rise_frame = i
                break

        # Fall time: frames from first clean-again to stable clean
        recovery_dists = transition_dists[15:]
        fall_frame = None
        for i, d in enumerate(recovery_dists):
            if d <= 0.1 * stable_corrupt:
                fall_frame = i
                break

        transition_results[ct] = {
            'distances': transition_dists,
            'stable_corrupt_dist': float(stable_corrupt),
            'rise_frames': rise_frame,
            'fall_frames': fall_frame,
            'is_instantaneous': rise_frame == 0 and fall_frame == 0,
        }
        print(f"  {ct}: rise={rise_frame} frames, fall={fall_frame} frames, "
              f"instant={rise_frame == 0 and fall_frame == 0}")

    results['transition_dynamics'] = transition_results

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/temporal_stability_{ts}.json"
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
