#!/usr/bin/env python3
"""Experiment 442: Temporal Consistency Analysis

Simulates sequential frame processing to test detection stability over time.
In real deployment, the robot processes frames continuously — does detection
remain stable? Can we use temporal smoothing to improve detection?

Tests:
1. Frame-to-frame embedding stability (clean sequences)
2. Detection during gradual corruption onset
3. Temporal smoothing (exponential moving average) vs single-frame
4. Detection latency (frames until corruption is flagged)
5. False alarm rate under natural frame variation
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

def extract_hidden(model, processor, image, prompt, layer=3):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return fwd.hidden_states[layer][0, -1, :].float().cpu().numpy()

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
    corruptions = ['fog', 'night', 'noise', 'blur']

    # Create base scenes
    seeds = [42, 123, 456, 789, 999, 1111, 2222, 3333]
    scenes = [Image.fromarray(np.random.RandomState(s).randint(0, 255, (224, 224, 3), dtype=np.uint8)) for s in seeds]

    print("Extracting calibration embeddings...")
    clean_embs = [extract_hidden(model, processor, s, prompt) for s in scenes]
    centroid = np.mean(clean_embs, axis=0)
    clean_dists = [cosine_dist(e, centroid) for e in clean_embs]
    threshold_3sigma = np.mean(clean_dists) + 3 * np.std(clean_dists)

    results = {"n_scenes": len(scenes), "threshold_3sigma": float(threshold_3sigma)}

    # === Test 1: Frame-to-frame stability ===
    print("\n=== Frame-to-Frame Stability ===")
    # Simulate slight variations by adding tiny noise to each frame
    stability_results = {}
    for s_idx in range(3):
        base = scenes[s_idx]
        base_emb = clean_embs[s_idx]
        frame_dists = []
        frame_to_frame = []
        prev_emb = base_emb
        for frame in range(10):
            # Small random perturbation (simulating camera noise, slight motion)
            arr = np.array(base).astype(np.float32)
            arr += np.random.RandomState(frame * 100 + s_idx).randn(*arr.shape) * 2.0  # tiny noise
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            frame_img = Image.fromarray(arr)
            emb = extract_hidden(model, processor, frame_img, prompt)
            d = cosine_dist(emb, centroid)
            f2f = cosine_dist(emb, prev_emb)
            frame_dists.append(float(d))
            frame_to_frame.append(float(f2f))
            prev_emb = emb

        stability_results[f"scene_{s_idx}"] = {
            "frame_dists": frame_dists,
            "frame_to_frame_dists": frame_to_frame,
            "mean_dist": float(np.mean(frame_dists)),
            "std_dist": float(np.std(frame_dists)),
            "mean_f2f": float(np.mean(frame_to_frame)),
            "max_f2f": float(np.max(frame_to_frame)),
        }
        print(f"  Scene {s_idx}: mean_dist={np.mean(frame_dists):.6f}, std={np.std(frame_dists):.6f}, mean_f2f={np.mean(frame_to_frame):.6f}")
    results["frame_stability"] = stability_results

    # === Test 2: Gradual corruption onset ===
    print("\n=== Gradual Corruption Onset ===")
    onset_results = {}
    severity_steps = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]

    for c in corruptions:
        per_step = []
        for sev in severity_steps:
            dists = []
            for s in scenes[:4]:
                if sev == 0.0:
                    emb = extract_hidden(model, processor, s, prompt)
                else:
                    emb = extract_hidden(model, processor, apply_corruption(s, c, severity=sev), prompt)
                dists.append(cosine_dist(emb, centroid))
            per_step.append({
                "severity": sev,
                "mean_dist": float(np.mean(dists)),
                "std_dist": float(np.std(dists)),
                "above_threshold": float(np.mean(np.array(dists) > threshold_3sigma)),
            })
        onset_results[c] = per_step
        # Find detection onset severity
        first_detect = next((s["severity"] for s in per_step if s["above_threshold"] > 0.5), None)
        print(f"  {c}: detection onset at severity={first_detect}")
    results["gradual_onset"] = onset_results

    # === Test 3: Temporal smoothing (EMA) ===
    print("\n=== Temporal Smoothing (EMA) ===")
    ema_results = {}
    alphas = [0.1, 0.3, 0.5, 0.7, 1.0]  # 1.0 = no smoothing

    for c in ['fog', 'night']:
        per_alpha = {}
        # Simulate: 5 clean frames, then 10 corrupted frames
        base = scenes[0]
        clean_frames = [base] * 5
        corr_frames = [apply_corruption(base, c)] * 10
        all_frames = clean_frames + corr_frames

        frame_embs = [extract_hidden(model, processor, f, prompt) for f in all_frames]
        raw_dists = [cosine_dist(e, centroid) for e in frame_embs]

        for alpha in alphas:
            ema_dists = []
            ema = raw_dists[0]
            for d in raw_dists:
                ema = alpha * d + (1 - alpha) * ema
                ema_dists.append(float(ema))

            # Frames to detect (first frame above threshold after corruption starts at frame 5)
            detect_frame = None
            for i in range(5, len(ema_dists)):
                if ema_dists[i] > threshold_3sigma:
                    detect_frame = i - 5  # frames after corruption onset
                    break

            per_alpha[str(alpha)] = {
                "ema_dists": ema_dists,
                "raw_dists": raw_dists,
                "detect_frame_after_onset": detect_frame,
                "final_ema": float(ema_dists[-1]),
            }
        ema_results[c] = per_alpha
        print(f"  {c}: detect frames after onset = " +
              ", ".join(f"α={a}: {per_alpha[str(a)]['detect_frame_after_onset']}" for a in alphas))
    results["temporal_smoothing"] = ema_results

    # === Test 4: False alarm rate ===
    print("\n=== False Alarm Rate Under Natural Variation ===")
    n_trials = 50
    rng = np.random.RandomState(42)
    false_alarms = 0
    all_clean_dists = []

    for trial in range(n_trials):
        # Random scene with tiny noise
        s_idx = rng.randint(0, len(scenes))
        arr = np.array(scenes[s_idx]).astype(np.float32)
        arr += rng.randn(*arr.shape) * 3.0  # natural camera noise
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        emb = extract_hidden(model, processor, Image.fromarray(arr), prompt)
        d = cosine_dist(emb, centroid)
        all_clean_dists.append(float(d))
        if d > threshold_3sigma:
            false_alarms += 1

    results["false_alarm_rate"] = {
        "n_trials": n_trials,
        "false_alarms": false_alarms,
        "false_alarm_rate": float(false_alarms / n_trials),
        "mean_dist": float(np.mean(all_clean_dists)),
        "max_dist": float(np.max(all_clean_dists)),
        "threshold": float(threshold_3sigma),
    }
    print(f"  False alarm rate: {false_alarms}/{n_trials} = {false_alarms/n_trials:.4f}")

    # === Test 5: Detection with intermittent corruption ===
    print("\n=== Intermittent Corruption Detection ===")
    intermittent = {}
    for c in ['fog', 'night']:
        # Pattern: clean, corrupt, clean, corrupt, clean...
        base = scenes[0]
        pattern = [0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1]  # 0=clean, 1=corrupt
        frame_dists = []
        for p in pattern:
            if p == 0:
                emb = extract_hidden(model, processor, base, prompt)
            else:
                emb = extract_hidden(model, processor, apply_corruption(base, c), prompt)
            frame_dists.append(float(cosine_dist(emb, centroid)))

        detected = [1 if d > threshold_3sigma else 0 for d in frame_dists]
        tp = sum(1 for p, d in zip(pattern, detected) if p == 1 and d == 1)
        fp = sum(1 for p, d in zip(pattern, detected) if p == 0 and d == 1)
        fn = sum(1 for p, d in zip(pattern, detected) if p == 1 and d == 0)
        tn = sum(1 for p, d in zip(pattern, detected) if p == 0 and d == 0)
        n_corrupt = sum(pattern)
        n_clean = len(pattern) - n_corrupt

        intermittent[c] = {
            "pattern": pattern,
            "frame_dists": frame_dists,
            "detected": detected,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0,
            "recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
        }
        print(f"  {c}: TP={tp}/{n_corrupt}, FP={fp}/{n_clean}, precision={tp/(tp+fp) if tp+fp > 0 else 0:.4f}, recall={tp/(tp+fn) if tp+fn > 0 else 0:.4f}")
    results["intermittent_detection"] = intermittent

    out_path = "/workspace/Vizuara-VLA-Research/experiments/temporal_consistency_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
