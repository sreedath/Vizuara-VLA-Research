"""
Threshold Sensitivity Analysis.

Determines the optimal detection threshold and maps the full
ROC curve with detailed FPR/TPR trade-offs. Tests multiple
threshold selection strategies: Youden's J, FPR<1%, FPR<5%,
equal error rate (EER).

Experiment 84 in the CalibDrive series.
"""
import os
import json
import datetime
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve

RESULTS_DIR = "/workspace/Vizuara-VLA-Research/experiments"
os.makedirs(RESULTS_DIR, exist_ok=True)
SIZE = (256, 256)


def create_highway(idx):
    rng = np.random.default_rng(idx * 5001)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]
    img[SIZE[0]//2:] = [80, 80, 80]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [255, 255, 255]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_urban(idx):
    rng = np.random.default_rng(idx * 5002)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [135, 206, 235]
    img[SIZE[0]//3:SIZE[0]//2] = [139, 119, 101]
    img[SIZE[0]//2:] = [60, 60, 60]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_noise(idx):
    rng = np.random.default_rng(idx * 5003)
    return rng.integers(0, 256, (*SIZE, 3), dtype=np.uint8)

def create_indoor(idx):
    rng = np.random.default_rng(idx * 5004)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:] = [200, 180, 160]
    img[SIZE[0]//2:, :] = [100, 80, 60]
    img[:SIZE[0]//3, SIZE[1]//3:2*SIZE[1]//3] = [150, 200, 255]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_twilight_highway(idx):
    rng = np.random.default_rng(idx * 5010)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [70, 50, 80]
    img[SIZE[0]//2:] = [60, 60, 60]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [200, 200, 100]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_snow(idx):
    rng = np.random.default_rng(idx * 5014)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [200, 200, 210]
    img[SIZE[0]//2:] = [220, 220, 230]
    img[SIZE[0]//2:, SIZE[1]//2-2:SIZE[1]//2+2] = [180, 180, 190]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_blackout(idx):
    return np.zeros((*SIZE, 3), dtype=np.uint8)

def create_inverted(idx):
    img = create_highway(idx + 3000)
    return (255 - img).astype(np.uint8)


def extract_hidden(model, processor, image, prompt):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    if hasattr(fwd, 'hidden_states') and fwd.hidden_states:
        return fwd.hidden_states[-1][0, -1, :].float().cpu().numpy()
    return None


def cosine_dist(a, b):
    return 1.0 - float(np.dot(a / (np.linalg.norm(a) + 1e-10),
                               b / (np.linalg.norm(b) + 1e-10)))


def main():
    print("=" * 70, flush=True)
    print("THRESHOLD SENSITIVITY ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    from transformers import AutoModelForVision2Seq, AutoProcessor
    print("Loading OpenVLA-7B...", flush=True)
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        "openvla/openvla-7b", trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.", flush=True)

    prompt = "In: What action should the robot take to drive forward at 25 m/s safely?\nOut:"

    # Calibrate
    print("\nCalibrating...", flush=True)
    cal_hidden = []
    for fn in [create_highway, create_urban]:
        for i in range(15):
            h = extract_hidden(model, processor,
                               Image.fromarray(fn(i + 9000)), prompt)
            if h is not None:
                cal_hidden.append(h)
    centroid = np.mean(cal_hidden, axis=0)
    cal_dists = [cosine_dist(h, centroid) for h in cal_hidden]
    print(f"  {len(cal_hidden)} calibration samples", flush=True)
    print(f"  Cal dist range: [{min(cal_dists):.4f}, {max(cal_dists):.4f}]", flush=True)

    # Larger test set for better ROC curve
    print("\nCollecting test data...", flush=True)
    id_scores = []
    id_scenarios = []
    for fn in [create_highway, create_urban]:
        for i in range(20):
            h = extract_hidden(model, processor,
                               Image.fromarray(fn(i + 500)), prompt)
            if h is not None:
                id_scores.append(cosine_dist(h, centroid))
                id_scenarios.append('highway' if fn == create_highway else 'urban')

    ood_scores = []
    ood_scenarios = []
    ood_fns = {
        'noise': (create_noise, 10),
        'indoor': (create_indoor, 10),
        'twilight': (create_twilight_highway, 10),
        'snow': (create_snow, 10),
        'blackout': (create_blackout, 6),
        'inverted': (create_inverted, 8),
    }

    cnt = 0
    total_ood = sum(v[1] for v in ood_fns.values())
    for cat, (fn, n) in ood_fns.items():
        for i in range(n):
            cnt += 1
            h = extract_hidden(model, processor,
                               Image.fromarray(fn(i + 500)), prompt)
            if h is not None:
                ood_scores.append(cosine_dist(h, centroid))
                ood_scenarios.append(cat)
            if cnt % 10 == 0:
                print(f"  [{cnt}/{total_ood}] {cat}", flush=True)

    print(f"  ID: {len(id_scores)}, OOD: {len(ood_scores)}", flush=True)

    # Full ROC curve
    labels = [0]*len(id_scores) + [1]*len(ood_scores)
    scores = id_scores + ood_scores
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auroc = roc_auc_score(labels, scores)

    # Threshold strategies
    strategies = {}

    # 1. Youden's J statistic (maximizes TPR - FPR)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    strategies['youden'] = {
        'threshold': float(thresholds[best_idx]),
        'fpr': float(fpr[best_idx]),
        'tpr': float(tpr[best_idx]),
        'j_score': float(j_scores[best_idx]),
    }

    # 2. FPR < 1%
    idx_1pct = np.where(fpr <= 0.01)[0]
    if len(idx_1pct) > 0:
        best_1pct = idx_1pct[np.argmax(tpr[idx_1pct])]
        strategies['fpr_1pct'] = {
            'threshold': float(thresholds[best_1pct]),
            'fpr': float(fpr[best_1pct]),
            'tpr': float(tpr[best_1pct]),
        }

    # 3. FPR < 5%
    idx_5pct = np.where(fpr <= 0.05)[0]
    if len(idx_5pct) > 0:
        best_5pct = idx_5pct[np.argmax(tpr[idx_5pct])]
        strategies['fpr_5pct'] = {
            'threshold': float(thresholds[best_5pct]),
            'fpr': float(fpr[best_5pct]),
            'tpr': float(tpr[best_5pct]),
        }

    # 4. Equal Error Rate (EER)
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    strategies['eer'] = {
        'threshold': float(thresholds[eer_idx]),
        'fpr': float(fpr[eer_idx]),
        'tpr': float(tpr[eer_idx]),
        'eer': float((fpr[eer_idx] + fnr[eer_idx]) / 2),
    }

    # 5. Calibration-based: mean + k*std of cal distances
    cal_mean = float(np.mean(cal_dists))
    cal_std = float(np.std(cal_dists))
    for k in [2, 3, 5]:
        thresh = cal_mean + k * cal_std
        tp = sum(1 for s in ood_scores if s > thresh)
        fp = sum(1 for s in id_scores if s > thresh)
        strategies[f'cal_mean_{k}std'] = {
            'threshold': thresh,
            'fpr': fp / len(id_scores),
            'tpr': tp / len(ood_scores),
        }

    # Per-category TPR at Youden threshold
    youden_thresh = strategies['youden']['threshold']
    per_cat_tpr = {}
    for cat in set(ood_scenarios):
        cat_scores = [s for s, sc in zip(ood_scores, ood_scenarios) if sc == cat]
        tpr_cat = sum(1 for s in cat_scores if s > youden_thresh) / len(cat_scores)
        per_cat_tpr[cat] = float(tpr_cat)

    # Print results
    print("\n" + "=" * 70, flush=True)
    print("THRESHOLD STRATEGIES", flush=True)
    print("=" * 70, flush=True)
    for name, s in strategies.items():
        print(f"  {name:<16}: thresh={s['threshold']:.4f}, "
              f"FPR={s['fpr']:.3f}, TPR={s['tpr']:.3f}", flush=True)

    print(f"\n  AUROC: {auroc:.4f}", flush=True)
    print(f"\n  Per-category TPR at Youden ({youden_thresh:.4f}):", flush=True)
    for cat, t in sorted(per_cat_tpr.items()):
        print(f"    {cat:<12}: {t:.3f}", flush=True)

    # ID score distribution
    print(f"\n  ID scores: mean={np.mean(id_scores):.4f}, "
          f"std={np.std(id_scores):.4f}, "
          f"range=[{min(id_scores):.4f}, {max(id_scores):.4f}]", flush=True)
    print(f"  OOD scores: mean={np.mean(ood_scores):.4f}, "
          f"std={np.std(ood_scores):.4f}, "
          f"range=[{min(ood_scores):.4f}, {max(ood_scores):.4f}]", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'threshold_analysis',
        'experiment_number': 84,
        'timestamp': timestamp,
        'auroc': float(auroc),
        'n_id': len(id_scores),
        'n_ood': len(ood_scores),
        'n_cal': len(cal_hidden),
        'strategies': strategies,
        'per_category_tpr': per_cat_tpr,
        'id_stats': {
            'mean': float(np.mean(id_scores)),
            'std': float(np.std(id_scores)),
            'min': float(min(id_scores)),
            'max': float(max(id_scores)),
        },
        'ood_stats': {
            'mean': float(np.mean(ood_scores)),
            'std': float(np.std(ood_scores)),
            'min': float(min(ood_scores)),
            'max': float(max(ood_scores)),
        },
        'roc_curve': {
            'fpr': fpr.tolist()[:50],  # subsample for storage
            'tpr': tpr.tolist()[:50],
            'thresholds': thresholds.tolist()[:50],
        }
    }
    output_path = os.path.join(RESULTS_DIR, f"threshold_analysis_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
