"""
Threshold Selection and Operating Points.

Detailed analysis of the ROC curve, optimal threshold selection,
and precision/recall/F1 at various operating points. This is the
deployment-critical analysis: given a desired false-positive or
false-negative rate, what threshold should be used?

Experiment 130 in the CalibDrive series.
"""
import os
import json
import datetime
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve

RESULTS_DIR = "/workspace/Vizuara-VLA-Research/experiments"
os.makedirs(RESULTS_DIR, exist_ok=True)
SIZE = (256, 256)


def create_highway(idx):
    rng = np.random.default_rng(idx * 23001)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]
    img[SIZE[0]//2:] = [80, 80, 80]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [255, 255, 255]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_urban(idx):
    rng = np.random.default_rng(idx * 23002)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [135, 206, 235]
    img[SIZE[0]//3:SIZE[0]//2] = [139, 119, 101]
    img[SIZE[0]//2:] = [60, 60, 60]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_noise(idx):
    rng = np.random.default_rng(idx * 23003)
    return rng.integers(0, 256, (*SIZE, 3), dtype=np.uint8)

def create_indoor(idx):
    rng = np.random.default_rng(idx * 23004)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:] = [200, 180, 160]
    img[SIZE[0]//2:, :] = [100, 80, 60]
    img[:SIZE[0]//3, SIZE[1]//3:2*SIZE[1]//3] = [150, 200, 255]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_twilight_highway(idx):
    rng = np.random.default_rng(idx * 23010)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [70, 50, 80]
    img[SIZE[0]//2:] = [60, 60, 60]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [200, 200, 100]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_snow(idx):
    rng = np.random.default_rng(idx * 23014)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [200, 200, 210]
    img[SIZE[0]//2:] = [220, 220, 230]
    img[SIZE[0]//2:, SIZE[1]//2-2:SIZE[1]//2+2] = [180, 180, 190]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_fog_highway(idx):
    rng = np.random.default_rng(idx * 23022)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [180, 185, 190]
    img[SIZE[0]//2:] = [140, 140, 145]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [160, 160, 165]
    noise = rng.integers(-15, 16, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def extract_hidden(model, processor, image, prompt):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    if not hasattr(fwd, 'hidden_states') or not fwd.hidden_states:
        return None
    return fwd.hidden_states[-1][0, -1, :].float().cpu().numpy()


def cosine_dist(a, b):
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def main():
    print("=" * 70, flush=True)
    print("THRESHOLD SELECTION AND OPERATING POINTS", flush=True)
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

    categories = {
        'highway': (create_highway, 'ID'),
        'urban': (create_urban, 'ID'),
        'noise': (create_noise, 'OOD'),
        'indoor': (create_indoor, 'OOD'),
        'twilight': (create_twilight_highway, 'OOD'),
        'snow': (create_snow, 'OOD'),
        'fog': (create_fog_highway, 'OOD'),
    }

    N_CAL = 10
    N_TEST = 15

    print("\n--- Collecting embeddings ---", flush=True)
    embeddings = {}
    for cat_name, (fn, group) in categories.items():
        print(f"  {cat_name} ({group})...", flush=True)
        embeds = []
        for i in range(N_CAL + N_TEST if group == 'ID' else N_TEST):
            h = extract_hidden(model, processor, Image.fromarray(fn(i + 2900)), prompt)
            if h is not None:
                embeds.append(h)
        embeddings[cat_name] = {'embeds': np.array(embeds), 'group': group}

    # Build cal/test
    cal_embeds = []
    test_embeds = []
    test_labels = []
    test_cats = []

    for cat_name, data in embeddings.items():
        if data['group'] == 'ID':
            cal_embeds.extend(data['embeds'][:N_CAL])
            for e in data['embeds'][N_CAL:]:
                test_embeds.append(e)
                test_labels.append(0)
                test_cats.append(cat_name)
        else:
            for e in data['embeds']:
                test_embeds.append(e)
                test_labels.append(1)
                test_cats.append(cat_name)

    cal_embeds = np.array(cal_embeds)
    test_embeds = np.array(test_embeds)
    test_labels = np.array(test_labels)
    centroid = np.mean(cal_embeds, axis=0)

    print(f"\nCal: {len(cal_embeds)}, Test: {len(test_labels)} ({sum(test_labels==0)} ID, {sum(test_labels==1)} OOD)", flush=True)

    # Compute scores
    scores = np.array([cosine_dist(e, centroid) for e in test_embeds])
    id_scores = scores[test_labels == 0]
    ood_scores = scores[test_labels == 1]

    print(f"\nID scores: {np.mean(id_scores):.4f} +/- {np.std(id_scores):.4f} (range: {np.min(id_scores):.4f}-{np.max(id_scores):.4f})", flush=True)
    print(f"OOD scores: {np.mean(ood_scores):.4f} +/- {np.std(ood_scores):.4f} (range: {np.min(ood_scores):.4f}-{np.max(ood_scores):.4f})", flush=True)

    # ROC analysis
    auroc = float(roc_auc_score(test_labels, scores))
    fpr, tpr, thresholds = roc_curve(test_labels, scores)
    print(f"\nAUROC: {auroc:.4f}", flush=True)

    # Optimal threshold (Youden's J)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    optimal_threshold = float(thresholds[best_idx])
    print(f"Optimal threshold (Youden's J): {optimal_threshold:.4f}", flush=True)

    # Score gap
    gap = float(np.min(ood_scores) - np.max(id_scores))
    print(f"Score gap (min OOD - max ID): {gap:.4f}", flush=True)

    # Threshold sweep
    print("\n--- Threshold Operating Points ---", flush=True)
    sweep_thresholds = np.linspace(np.min(scores), np.max(scores), 50)
    operating_points = []

    for t in sweep_thresholds:
        tp = int(np.sum((scores >= t) & (test_labels == 1)))
        fp = int(np.sum((scores >= t) & (test_labels == 0)))
        tn = int(np.sum((scores < t) & (test_labels == 0)))
        fn = int(np.sum((scores < t) & (test_labels == 1)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr_val = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        operating_points.append({
            'threshold': float(t),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'fpr': fpr_val,
            'fnr': fnr_val,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        })

    # Key operating points
    print(f"{'Threshold':>10s} {'Prec':>6s} {'Recall':>6s} {'F1':>6s} {'FPR':>6s} {'FNR':>6s}", flush=True)
    for op in operating_points[::5]:
        print(f"  {op['threshold']:8.4f}   {op['precision']:5.3f}  {op['recall']:5.3f}  {op['f1']:5.3f}  {op['fpr']:5.3f}  {op['fnr']:5.3f}", flush=True)

    # Per-category scores
    print("\n--- Per-Category Scores ---", flush=True)
    per_cat_scores = {}
    for cat_name in set(test_cats):
        cat_mask = np.array([c == cat_name for c in test_cats])
        cat_scores = scores[cat_mask]
        per_cat_scores[cat_name] = {
            'mean': float(np.mean(cat_scores)),
            'std': float(np.std(cat_scores)),
            'min': float(np.min(cat_scores)),
            'max': float(np.max(cat_scores)),
        }
        print(f"  {cat_name:12s}: {np.mean(cat_scores):.4f} +/- {np.std(cat_scores):.4f} "
              f"(range: {np.min(cat_scores):.4f}-{np.max(cat_scores):.4f})", flush=True)

    # Recommended thresholds for different safety levels
    print("\n--- Recommended Thresholds ---", flush=True)
    id_max = float(np.max(id_scores))
    id_mean = float(np.mean(id_scores))
    id_std = float(np.std(id_scores))
    recommendations = {
        'conservative (id_mean + 3*std)': id_mean + 3 * id_std,
        'moderate (id_mean + 5*std)': id_mean + 5 * id_std,
        'relaxed (midpoint)': (id_max + float(np.min(ood_scores))) / 2,
    }
    for name, t in recommendations.items():
        tp = int(np.sum((scores >= t) & (test_labels == 1)))
        fp = int(np.sum((scores >= t) & (test_labels == 0)))
        fn = int(np.sum((scores < t) & (test_labels == 1)))
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr_val = fp / (fp + sum(test_labels == 0))
        print(f"  {name}: t={t:.4f}, recall={recall:.3f}, FPR={fpr_val:.3f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'threshold_analysis',
        'experiment_number': 130,
        'timestamp': timestamp,
        'auroc': auroc,
        'optimal_threshold': optimal_threshold,
        'score_gap': gap,
        'id_stats': {'mean': float(np.mean(id_scores)), 'std': float(np.std(id_scores)),
                     'min': float(np.min(id_scores)), 'max': float(np.max(id_scores))},
        'ood_stats': {'mean': float(np.mean(ood_scores)), 'std': float(np.std(ood_scores)),
                      'min': float(np.min(ood_scores)), 'max': float(np.max(ood_scores))},
        'per_category': per_cat_scores,
        'recommendations': {k: float(v) for k, v in recommendations.items()},
        'n_operating_points': len(operating_points),
    }
    output_path = os.path.join(RESULTS_DIR, f"threshold_analysis_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
