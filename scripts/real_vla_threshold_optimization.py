#!/usr/bin/env python3
"""Experiment 340: Threshold Optimization for Deployment

Characterize the ROC curve shape and find optimal thresholds that:
1. Separate true corruption from benign perturbations (JPEG, resize)
2. Maximize F1 under realistic deployment conditions
3. Analyze threshold stability across scenes and corruption types
4. Study the distance distribution shapes (clean vs benign vs corrupt)
"""

import json, time, os, sys, io
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
    elif ctype == 'rain':
        rng = np.random.RandomState(42)
        for _ in range(200):
            x, y = rng.randint(0, 224), rng.randint(0, 224)
            length = rng.randint(5, 20)
            for k in range(length):
                if y + k < 224:
                    arr[y+k, x, :] = np.clip(arr[y+k, x, :] + 0.3 * severity, 0, 1)
        return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    elif ctype == 'frost':
        rng = np.random.RandomState(42)
        frost = rng.random((224, 224, 3)) * 0.4 * severity
        arr = np.clip(arr + frost, 0, 1) * (1 - 0.3 * severity) + 0.6 * severity * 0.3
        arr = np.clip(arr, 0, 1)
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))

def cosine_dist(a, b):
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return 1.0 - dot / (na * nb)

def apply_jpeg(image, quality):
    buf = io.BytesIO()
    image.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return Image.open(buf).convert('RGB')

def apply_resize(image, size):
    return image.resize((size, size), Image.BILINEAR).resize((224, 224), Image.BILINEAR)

def compute_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def main():
    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    results = {}

    # Create diverse scenes
    seeds = [0, 100, 200, 300, 400, 500, 600, 700]
    scenes = {}
    cal_embs = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        px = rng.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        scenes[seed] = Image.fromarray(px)
        cal_embs[seed] = extract_hidden(model, processor, scenes[seed], prompt)
        print(f"  Scene {seed} calibrated")

    ctypes = ['fog', 'night', 'noise', 'blur', 'rain', 'frost']

    # ========== 1. Distance distributions ==========
    print("\n=== Distance Distributions ===")

    dist_data = {'clean': [], 'benign': [], 'corrupt_low': [], 'corrupt_mid': [], 'corrupt_high': []}

    for seed in seeds:
        cal = cal_embs[seed]
        img = scenes[seed]

        # Clean: re-embed same image (should be 0)
        emb = extract_hidden(model, processor, img, prompt)
        d = cosine_dist(cal, emb)
        dist_data['clean'].append(float(d))

        # Benign perturbations: JPEG, slight resize
        for q in [95, 90, 85, 80]:
            jpeg_img = apply_jpeg(img, q)
            emb = extract_hidden(model, processor, jpeg_img, prompt)
            d = cosine_dist(cal, emb)
            dist_data['benign'].append(float(d))

        for sz in [200, 180, 160]:
            res_img = apply_resize(img, sz)
            emb = extract_hidden(model, processor, res_img, prompt)
            d = cosine_dist(cal, emb)
            dist_data['benign'].append(float(d))

        # Low severity corruption (0.05-0.1)
        for ct in ctypes:
            for sev in [0.05, 0.1]:
                corrupt_img = apply_corruption(img, ct, sev)
                emb = extract_hidden(model, processor, corrupt_img, prompt)
                d = cosine_dist(cal, emb)
                dist_data['corrupt_low'].append(float(d))

        # Mid severity corruption (0.3-0.5)
        for ct in ctypes:
            for sev in [0.3, 0.5]:
                corrupt_img = apply_corruption(img, ct, sev)
                emb = extract_hidden(model, processor, corrupt_img, prompt)
                d = cosine_dist(cal, emb)
                dist_data['corrupt_mid'].append(float(d))

        # High severity corruption (0.8-1.0)
        for ct in ctypes:
            for sev in [0.8, 1.0]:
                corrupt_img = apply_corruption(img, ct, sev)
                emb = extract_hidden(model, processor, corrupt_img, prompt)
                d = cosine_dist(cal, emb)
                dist_data['corrupt_high'].append(float(d))

        print(f"  Scene {seed}: clean={dist_data['clean'][-1]:.6f}, "
              f"benign_max={max(dist_data['benign'][-7:]):.6f}, "
              f"corrupt_min={min(dist_data['corrupt_low'][-12:]):.6f}")

    # Distribution statistics
    dist_stats = {}
    for category, dists in dist_data.items():
        arr = np.array(dists)
        dist_stats[category] = {
            'n': len(dists),
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'median': float(np.median(arr)),
            'p25': float(np.percentile(arr, 25)),
            'p75': float(np.percentile(arr, 75)),
            'p95': float(np.percentile(arr, 95)),
            'p99': float(np.percentile(arr, 99)),
        }
    results['distributions'] = dist_stats

    print("\n  Distribution summary:")
    for cat, s in dist_stats.items():
        print(f"    {cat}: mean={s['mean']:.6f}, std={s['std']:.6f}, "
              f"range=[{s['min']:.6f}, {s['max']:.6f}]")

    # ========== 2. ROC curve with benign vs corrupt ==========
    print("\n=== ROC Curve Analysis ===")

    negatives = dist_data['clean'] + dist_data['benign']
    positives = dist_data['corrupt_low'] + dist_data['corrupt_mid'] + dist_data['corrupt_high']

    all_scores = sorted(set(negatives + positives))
    thresholds = np.linspace(0, max(all_scores) * 1.1, 500)

    roc_points = []
    best_f1 = 0
    best_threshold = 0
    best_f1_details = {}

    for thresh in thresholds:
        tp = sum(1 for s in positives if s > thresh)
        fp = sum(1 for s in negatives if s > thresh)
        fn = sum(1 for s in positives if s <= thresh)
        tn = sum(1 for s in negatives if s <= thresh)

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        f1 = compute_f1(tp, fp, fn)

        roc_points.append({
            'threshold': float(thresh),
            'tpr': float(tpr),
            'fpr': float(fpr),
            'f1': float(f1),
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        })

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(thresh)
            best_f1_details = roc_points[-1]

    # Compute AUC
    fpr_list = [p['fpr'] for p in roc_points]
    tpr_list = [p['tpr'] for p in roc_points]
    auc = 0
    for i in range(1, len(fpr_list)):
        auc += abs(fpr_list[i] - fpr_list[i-1]) * (tpr_list[i] + tpr_list[i-1]) / 2

    results['roc'] = {
        'auc': float(auc),
        'n_negatives': len(negatives),
        'n_positives': len(positives),
        'best_f1': float(best_f1),
        'best_threshold': best_threshold,
        'best_f1_details': best_f1_details,
    }

    print(f"  AUC (benign-aware): {auc:.4f}")
    print(f"  Best F1: {best_f1:.4f} at threshold={best_threshold:.6f}")
    print(f"  Details: TP={best_f1_details['tp']}, FP={best_f1_details['fp']}, "
          f"FN={best_f1_details['fn']}, TN={best_f1_details['tn']}")

    # ========== 3. Gap analysis ==========
    print("\n=== Gap Analysis ===")

    max_benign = max(dist_data['benign'])
    min_corrupt_low = min(dist_data['corrupt_low'])
    min_corrupt_mid = min(dist_data['corrupt_mid'])
    min_corrupt_high = min(dist_data['corrupt_high'])

    gap_low = min_corrupt_low - max_benign
    gap_mid = min_corrupt_mid - max_benign
    gap_high = min_corrupt_high - max_benign

    results['gap_analysis'] = {
        'max_benign': float(max_benign),
        'min_corrupt_low': float(min_corrupt_low),
        'min_corrupt_mid': float(min_corrupt_mid),
        'min_corrupt_high': float(min_corrupt_high),
        'gap_to_low_corrupt': float(gap_low),
        'gap_to_mid_corrupt': float(gap_mid),
        'gap_to_high_corrupt': float(gap_high),
        'gap_ratio_low': float(min_corrupt_low / max_benign) if max_benign > 0 else float('inf'),
        'gap_ratio_mid': float(min_corrupt_mid / max_benign) if max_benign > 0 else float('inf'),
        'gap_ratio_high': float(min_corrupt_high / max_benign) if max_benign > 0 else float('inf'),
        'separable_low': bool(gap_low > 0),
        'separable_mid': bool(gap_mid > 0),
        'separable_high': bool(gap_high > 0),
    }

    print(f"  Max benign distance: {max_benign:.6f}")
    print(f"  Min low-corrupt distance: {min_corrupt_low:.6f} (gap: {gap_low:.6f})")
    print(f"  Min mid-corrupt distance: {min_corrupt_mid:.6f} (gap: {gap_mid:.6f})")
    print(f"  Gap ratio (mid/benign): {results['gap_analysis']['gap_ratio_mid']:.1f}x")

    # ========== 4. Per-corruption-type thresholds ==========
    print("\n=== Per-Type Threshold Analysis ===")

    per_type = {}
    for ct in ctypes:
        type_dists = []
        for seed in seeds:
            for sev in [0.05, 0.1, 0.3, 0.5, 0.8, 1.0]:
                corrupt_img = apply_corruption(scenes[seed], ct, sev)
                emb = extract_hidden(model, processor, corrupt_img, prompt)
                d = cosine_dist(cal_embs[seed], emb)
                type_dists.append({'severity': sev, 'distance': float(d)})

        arr = np.array([x['distance'] for x in type_dists])
        sev_groups = {}
        for item in type_dists:
            s = item['severity']
            if s not in sev_groups:
                sev_groups[s] = []
            sev_groups[s].append(item['distance'])

        per_type[ct] = {
            'overall_min': float(np.min(arr)),
            'overall_max': float(np.max(arr)),
            'overall_mean': float(np.mean(arr)),
            'by_severity': {
                str(sev): {
                    'mean': float(np.mean(vals)),
                    'min': float(np.min(vals)),
                    'max': float(np.max(vals)),
                }
                for sev, vals in sorted(sev_groups.items())
            },
            'min_detectable_above_benign': float(min(d for d in arr if d > max_benign)) if any(d > max_benign for d in arr) else None,
        }
        print(f"  {ct}: range=[{per_type[ct]['overall_min']:.6f}, {per_type[ct]['overall_max']:.6f}], "
              f"min_above_benign={per_type[ct]['min_detectable_above_benign']}")

    results['per_type'] = per_type

    # ========== 5. Threshold stability across scenes ==========
    print("\n=== Threshold Stability ===")

    scene_optimal = {}
    for seed in seeds:
        cal = cal_embs[seed]
        img = scenes[seed]

        # Benign for this scene
        scene_benign = []
        for q in [95, 90, 85, 80]:
            jpeg_img = apply_jpeg(img, q)
            emb = extract_hidden(model, processor, jpeg_img, prompt)
            scene_benign.append(float(cosine_dist(cal, emb)))

        # Corrupt for this scene
        scene_corrupt = []
        for ct in ctypes:
            for sev in [0.3, 0.5]:
                corrupt_img = apply_corruption(img, ct, sev)
                emb = extract_hidden(model, processor, corrupt_img, prompt)
                scene_corrupt.append(float(cosine_dist(cal, emb)))

        max_b = max(scene_benign) if scene_benign else 0
        min_c = min(scene_corrupt) if scene_corrupt else 0
        optimal_t = (max_b + min_c) / 2

        scene_optimal[str(seed)] = {
            'max_benign': float(max_b),
            'min_corrupt': float(min_c),
            'optimal_threshold': float(optimal_t),
            'gap': float(min_c - max_b),
            'gap_ratio': float(min_c / max_b) if max_b > 0 else float('inf'),
        }
        print(f"  Scene {seed}: optimal_t={optimal_t:.6f}, gap_ratio={scene_optimal[str(seed)]['gap_ratio']:.1f}x")

    thresholds_list = [v['optimal_threshold'] for v in scene_optimal.values()]
    results['threshold_stability'] = {
        'per_scene': scene_optimal,
        'mean_threshold': float(np.mean(thresholds_list)),
        'std_threshold': float(np.std(thresholds_list)),
        'cv_threshold': float(np.std(thresholds_list) / np.mean(thresholds_list)) if np.mean(thresholds_list) > 0 else 0,
        'min_threshold': float(np.min(thresholds_list)),
        'max_threshold': float(np.max(thresholds_list)),
    }

    print(f"\n  Threshold CV: {results['threshold_stability']['cv_threshold']:.3f}")
    print(f"  Range: [{results['threshold_stability']['min_threshold']:.6f}, "
          f"{results['threshold_stability']['max_threshold']:.6f}]")

    # ========== 6. Practical deployment thresholds ==========
    print("\n=== Deployment Thresholds ===")

    strategies = {}

    # Strategy 1: d > 0 (original)
    tp = sum(1 for s in positives if s > 0)
    fp = sum(1 for s in negatives if s > 0)
    fn = sum(1 for s in positives if s <= 0)
    tn = sum(1 for s in negatives if s <= 0)
    strategies['zero'] = {
        'threshold': 0.0,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision': float(tp/(tp+fp)) if (tp+fp) > 0 else 0,
        'recall': float(tp/(tp+fn)) if (tp+fn) > 0 else 0,
        'f1': float(compute_f1(tp, fp, fn)),
    }

    # Strategy 2: midpoint between max benign and min low-corrupt
    mid_t = (max_benign + min_corrupt_low) / 2
    tp = sum(1 for s in positives if s > mid_t)
    fp = sum(1 for s in negatives if s > mid_t)
    fn = sum(1 for s in positives if s <= mid_t)
    tn = sum(1 for s in negatives if s <= mid_t)
    strategies['midpoint'] = {
        'threshold': float(mid_t),
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision': float(tp/(tp+fp)) if (tp+fp) > 0 else 0,
        'recall': float(tp/(tp+fn)) if (tp+fn) > 0 else 0,
        'f1': float(compute_f1(tp, fp, fn)),
    }

    # Strategy 3: 2x max benign
    t_2x = max_benign * 2
    tp = sum(1 for s in positives if s > t_2x)
    fp = sum(1 for s in negatives if s > t_2x)
    fn = sum(1 for s in positives if s <= t_2x)
    tn = sum(1 for s in negatives if s <= t_2x)
    strategies['2x_benign'] = {
        'threshold': float(t_2x),
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision': float(tp/(tp+fp)) if (tp+fp) > 0 else 0,
        'recall': float(tp/(tp+fn)) if (tp+fn) > 0 else 0,
        'f1': float(compute_f1(tp, fp, fn)),
    }

    # Strategy 4: Best F1 threshold
    strategies['best_f1'] = {
        'threshold': best_threshold,
        'tp': best_f1_details['tp'],
        'fp': best_f1_details['fp'],
        'fn': best_f1_details['fn'],
        'tn': best_f1_details['tn'],
        'precision': float(best_f1_details['tp']/(best_f1_details['tp']+best_f1_details['fp'])) if (best_f1_details['tp']+best_f1_details['fp']) > 0 else 0,
        'recall': float(best_f1_details['tp']/(best_f1_details['tp']+best_f1_details['fn'])) if (best_f1_details['tp']+best_f1_details['fn']) > 0 else 0,
        'f1': float(best_f1),
    }

    results['strategies'] = strategies
    for name, s in strategies.items():
        print(f"  {name}: t={s['threshold']:.6f}, F1={s['f1']:.4f}, "
              f"prec={s['precision']:.4f}, rec={s['recall']:.4f}")

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/threshold_optimization_{ts}.json"
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
