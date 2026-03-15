#!/usr/bin/env python3
"""Experiment 356: Online Threshold Adaptation

How should detection thresholds be set and adapted?
1. Fixed threshold from calibration vs adaptive (running percentile)
2. CUSUM (cumulative sum) change detection on embedding distances
3. Page-Hinkley test for distribution shift detection
4. Threshold sensitivity: FPR/FNR tradeoffs across threshold values
5. Mixed-severity streams: detection delay vs false alarm rate
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
    results = {}
    ctypes = ['fog', 'night', 'noise', 'blur']

    # Generate calibration and test images
    print("Generating images...")
    cal_seeds = list(range(0, 1000, 100))[:10]
    test_seeds = list(range(2000, 3000, 100))[:10]

    cal_embs = {}
    for seed in cal_seeds:
        rng = np.random.RandomState(seed)
        px = rng.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(px)
        cal_embs[seed] = extract_hidden(model, processor, img, prompt)

    centroid = np.mean([cal_embs[s] for s in cal_seeds], axis=0)

    test_images = {}
    test_embs = {}
    for seed in test_seeds:
        rng = np.random.RandomState(seed)
        px = rng.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        test_images[seed] = Image.fromarray(px)
        test_embs[seed] = extract_hidden(model, processor, test_images[seed], prompt)

    # Compute clean baseline distances
    clean_dists = [float(cosine_dist(centroid, test_embs[s])) for s in test_seeds]
    clean_max = max(clean_dists)
    clean_mean = float(np.mean(clean_dists))
    clean_std = float(np.std(clean_dists))

    print(f"  Clean distances: mean={clean_mean:.6f}, max={clean_max:.6f}, std={clean_std:.6f}")

    # ========== 1. ROC Curve / Threshold Sweep ==========
    print("\n=== Threshold Sweep (ROC Analysis) ===")

    threshold_results = {}
    for ct in ctypes:
        ood_dists = []
        for seed in test_seeds:
            img = apply_corruption(test_images[seed], ct, 0.5)
            emb = extract_hidden(model, processor, img, prompt)
            ood_dists.append(float(cosine_dist(centroid, emb)))

        # Sweep thresholds
        all_scores = clean_dists + ood_dists
        thresholds = sorted(set(all_scores))
        # Add boundary thresholds
        thresholds = [min(all_scores) - 0.001] + thresholds + [max(all_scores) + 0.001]

        roc_points = []
        for thresh in thresholds:
            tp = sum(1 for d in ood_dists if d > thresh)
            fp = sum(1 for d in clean_dists if d > thresh)
            fn = sum(1 for d in ood_dists if d <= thresh)
            tn = sum(1 for d in clean_dists if d <= thresh)

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            roc_points.append({'threshold': float(thresh), 'tpr': tpr, 'fpr': fpr})

        # Find optimal threshold (Youden's J)
        best_j = -1
        best_thresh = 0
        for pt in roc_points:
            j = pt['tpr'] - pt['fpr']
            if j > best_j:
                best_j = j
                best_thresh = pt['threshold']

        # Gap between clean max and OOD min
        ood_min = min(ood_dists)
        gap = ood_min - clean_max

        threshold_results[ct] = {
            'clean_max': clean_max,
            'ood_min': float(ood_min),
            'gap': float(gap),
            'gap_relative': float(gap / clean_max) if clean_max > 0 else float('inf'),
            'optimal_threshold': float(best_thresh),
            'youden_j': float(best_j),
            'auroc': float(compute_auroc(clean_dists, ood_dists)),
            'n_roc_points': len(roc_points),
        }
        print(f"  {ct}: gap={gap:.6f} ({gap/clean_max*100:.1f}%), AUROC={compute_auroc(clean_dists, ood_dists):.4f}")

    results['threshold_sweep'] = threshold_results

    # ========== 2. Simulated Stream with CUSUM ==========
    print("\n=== CUSUM Change Detection ===")

    cusum_results = {}
    for ct in ctypes:
        # Create a stream: 20 clean frames, then 20 corrupted, then 20 clean
        stream_dists = []
        stream_labels = []  # 0=clean, 1=corrupted

        for i in range(20):
            seed = test_seeds[i % len(test_seeds)]
            stream_dists.append(float(cosine_dist(centroid, test_embs[seed])))
            stream_labels.append(0)

        for i in range(20):
            seed = test_seeds[i % len(test_seeds)]
            img = apply_corruption(test_images[seed], ct, 0.5)
            emb = extract_hidden(model, processor, img, prompt)
            stream_dists.append(float(cosine_dist(centroid, emb)))
            stream_labels.append(1)

        for i in range(20):
            seed = test_seeds[i % len(test_seeds)]
            stream_dists.append(float(cosine_dist(centroid, test_embs[seed])))
            stream_labels.append(0)

        # CUSUM detection
        # Reference value = mean of calibration distances
        mu0 = clean_mean
        # Allowable slack
        k_val = clean_max * 1.5  # threshold shift

        cusum_pos = [0.0]
        cusum_neg = [0.0]
        for d in stream_dists:
            sp = max(0, cusum_pos[-1] + (d - mu0) - k_val)
            sn = max(0, cusum_neg[-1] + (mu0 - d) - k_val)
            cusum_pos.append(sp)
            cusum_neg.append(sn)

        # Find first detection (CUSUM > h)
        h_values = [clean_max * 2, clean_max * 5, clean_max * 10]
        detections = {}
        for h in h_values:
            first_detect = None
            for t, sp in enumerate(cusum_pos):
                if sp > h:
                    first_detect = t
                    break
            h_key = f"h={h:.6f}"
            detections[h_key] = {
                'first_detection': first_detect,
                'detection_delay': first_detect - 20 if first_detect is not None and first_detect >= 20 else None,
                'detected_in_clean': first_detect is not None and first_detect < 20,
            }

        cusum_results[ct] = {
            'stream_length': len(stream_dists),
            'clean_mean_dist': float(np.mean(stream_dists[:20])),
            'corrupt_mean_dist': float(np.mean(stream_dists[20:40])),
            'recovery_mean_dist': float(np.mean(stream_dists[40:])),
            'cusum_max': float(max(cusum_pos)),
            'detections': detections,
        }
        print(f"  {ct}: clean={np.mean(stream_dists[:20]):.6f}, corrupt={np.mean(stream_dists[20:40]):.6f}")

    results['cusum'] = cusum_results

    # ========== 3. Page-Hinkley Test ==========
    print("\n=== Page-Hinkley Test ===")

    ph_results = {}
    for ct in ctypes:
        # Same stream as CUSUM
        stream_dists = []
        stream_labels = []

        for i in range(20):
            seed = test_seeds[i % len(test_seeds)]
            stream_dists.append(float(cosine_dist(centroid, test_embs[seed])))
            stream_labels.append(0)

        for i in range(20):
            seed = test_seeds[i % len(test_seeds)]
            img = apply_corruption(test_images[seed], ct, 0.5)
            emb = extract_hidden(model, processor, img, prompt)
            stream_dists.append(float(cosine_dist(centroid, emb)))
            stream_labels.append(1)

        for i in range(20):
            seed = test_seeds[i % len(test_seeds)]
            stream_dists.append(float(cosine_dist(centroid, test_embs[seed])))
            stream_labels.append(0)

        # Page-Hinkley: track cumulative deviation from running mean
        delta = clean_max  # minimum allowed deviation
        lambdas = [clean_max * 5, clean_max * 10, clean_max * 20]

        for lam in lambdas:
            m_t = 0
            M_t = 0
            running_sum = 0
            first_detect = None

            for t, d in enumerate(stream_dists):
                running_sum += d
                m_t = running_sum / (t + 1)
                # PH statistic
                ph_stat = running_sum - (t + 1) * m_t - delta
                if t == 0:
                    M_t = ph_stat
                else:
                    M_t = min(M_t, ph_stat)

                if ph_stat - M_t > lam and first_detect is None:
                    first_detect = t

            lam_key = f"lambda={lam:.6f}"
            if ct not in ph_results:
                ph_results[ct] = {}
            ph_results[ct][lam_key] = {
                'first_detection': first_detect,
                'detection_delay': first_detect - 20 if first_detect is not None and first_detect >= 20 else None,
                'false_alarm': first_detect is not None and first_detect < 20,
            }

        print(f"  {ct}: PH detections = " + str({k: v['first_detection'] for k, v in ph_results[ct].items()}))

    results['page_hinkley'] = ph_results

    # ========== 4. Severity-Dependent Detection Delay ==========
    print("\n=== Severity vs Detection Delay ===")

    severity_delay = {}
    for ct in ctypes:
        per_sev = {}
        for sev in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
            # Stream: 10 clean, 10 corrupted
            stream_dists = []

            for i in range(10):
                seed = test_seeds[i % len(test_seeds)]
                stream_dists.append(float(cosine_dist(centroid, test_embs[seed])))

            for i in range(10):
                seed = test_seeds[i % len(test_seeds)]
                img = apply_corruption(test_images[seed], ct, sev)
                emb = extract_hidden(model, processor, img, prompt)
                stream_dists.append(float(cosine_dist(centroid, emb)))

            # Simple threshold detection at clean_max * 1.1
            thresh = clean_max * 1.1 if clean_max > 0 else 1e-6
            first_above = None
            for t in range(10, 20):
                if stream_dists[t] > thresh:
                    first_above = t - 10  # delay from corruption start
                    break

            # Also check if all corrupted frames detected
            n_detected = sum(1 for t in range(10, 20) if stream_dists[t] > thresh)

            per_sev[str(sev)] = {
                'detection_delay': first_above,
                'n_detected': n_detected,
                'n_total': 10,
                'detection_rate': n_detected / 10.0,
                'corrupt_mean_dist': float(np.mean(stream_dists[10:])),
                'margin': float(np.mean(stream_dists[10:]) - thresh),
            }

        severity_delay[ct] = per_sev
        # Print summary
        delays = [per_sev[str(s)]['detection_delay'] for s in [0.05, 0.1, 0.5, 1.0]]
        rates = [per_sev[str(s)]['detection_rate'] for s in [0.05, 0.1, 0.5, 1.0]]
        print(f"  {ct}: delays@[0.05,0.1,0.5,1.0]={delays}, rates={rates}")

    results['severity_delay'] = severity_delay

    # ========== 5. Running Percentile Threshold ==========
    print("\n=== Running Percentile Adaptive Threshold ===")

    adaptive_results = {}
    for ct in ctypes:
        # Long stream: 30 clean, 20 corrupted, 30 clean
        stream_dists = []
        stream_labels = []

        for i in range(30):
            seed = test_seeds[i % len(test_seeds)]
            stream_dists.append(float(cosine_dist(centroid, test_embs[seed])))
            stream_labels.append(0)

        for i in range(20):
            seed = test_seeds[i % len(test_seeds)]
            img = apply_corruption(test_images[seed], ct, 0.5)
            emb = extract_hidden(model, processor, img, prompt)
            stream_dists.append(float(cosine_dist(centroid, emb)))
            stream_labels.append(1)

        for i in range(30):
            seed = test_seeds[i % len(test_seeds)]
            stream_dists.append(float(cosine_dist(centroid, test_embs[seed])))
            stream_labels.append(0)

        # Fixed threshold
        fixed_thresh = clean_max * 1.1 if clean_max > 0 else 1e-6
        fixed_detections = [1 if d > fixed_thresh else 0 for d in stream_dists]

        # Running percentile (95th of last W frames)
        W = 10
        adaptive_detections = []
        for t in range(len(stream_dists)):
            if t < W:
                # Use calibration threshold during warmup
                adaptive_detections.append(1 if stream_dists[t] > fixed_thresh else 0)
            else:
                window = stream_dists[max(0, t-W):t]
                p95 = float(np.percentile(window, 95))
                adaptive_detections.append(1 if stream_dists[t] > p95 * 1.1 else 0)

        # Compare performance
        fixed_tp = sum(1 for t in range(len(stream_dists)) if fixed_detections[t] == 1 and stream_labels[t] == 1)
        fixed_fp = sum(1 for t in range(len(stream_dists)) if fixed_detections[t] == 1 and stream_labels[t] == 0)
        fixed_fn = sum(1 for t in range(len(stream_dists)) if fixed_detections[t] == 0 and stream_labels[t] == 1)
        fixed_tn = sum(1 for t in range(len(stream_dists)) if fixed_detections[t] == 0 and stream_labels[t] == 0)

        adap_tp = sum(1 for t in range(len(stream_dists)) if adaptive_detections[t] == 1 and stream_labels[t] == 1)
        adap_fp = sum(1 for t in range(len(stream_dists)) if adaptive_detections[t] == 1 and stream_labels[t] == 0)
        adap_fn = sum(1 for t in range(len(stream_dists)) if adaptive_detections[t] == 0 and stream_labels[t] == 1)
        adap_tn = sum(1 for t in range(len(stream_dists)) if adaptive_detections[t] == 0 and stream_labels[t] == 0)

        adaptive_results[ct] = {
            'fixed': {
                'tp': fixed_tp, 'fp': fixed_fp, 'fn': fixed_fn, 'tn': fixed_tn,
                'precision': fixed_tp / (fixed_tp + fixed_fp) if (fixed_tp + fixed_fp) > 0 else 0,
                'recall': fixed_tp / (fixed_tp + fixed_fn) if (fixed_tp + fixed_fn) > 0 else 0,
            },
            'adaptive': {
                'tp': adap_tp, 'fp': adap_fp, 'fn': adap_fn, 'tn': adap_tn,
                'precision': adap_tp / (adap_tp + adap_fp) if (adap_tp + adap_fp) > 0 else 0,
                'recall': adap_tp / (adap_tp + adap_fn) if (adap_tp + adap_fn) > 0 else 0,
            },
        }
        print(f"  {ct}: fixed F1={2*fixed_tp/(2*fixed_tp+fixed_fp+fixed_fn):.3f} vs " +
              f"adaptive F1={2*adap_tp/(2*adap_tp+adap_fp+adap_fn):.3f}" if (2*adap_tp+adap_fp+adap_fn) > 0 else f"  {ct}: adaptive no detections")

    results['adaptive_threshold'] = adaptive_results

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/online_threshold_{ts}.json"
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
