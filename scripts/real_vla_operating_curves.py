"""
Operating Characteristic Curves for the Full Pipeline.

Generates ROC curves, precision-recall curves, and analyzes the
coverage-safety tradeoff at different operating points for the
detection pipeline.

Tests:
1. ROC curves for each pipeline configuration
2. Precision-recall curves
3. Coverage vs safety rate at multiple thresholds
4. FPR vs TPR at specific operating points
5. Optimal threshold selection analysis

Uses realistic test images with trajectory aggregation.

Experiment 52 in the CalibDrive series.
"""
import os
import json
import datetime
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score

RESULTS_DIR = "/workspace/Vizuara-VLA-Research/experiments"
os.makedirs(RESULTS_DIR, exist_ok=True)
SIZE = (256, 256)


# Image generators (same as ablation study)
def create_highway_realistic(idx):
    rng = np.random.default_rng(idx * 7001)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]
    for row in range(SIZE[0]//2, SIZE[0]):
        t = (row - SIZE[0]//2) / (SIZE[0]//2)
        gray = int(80 + t * 30)
        img[row, :] = [gray, gray, gray]
    cx = SIZE[1]//2 + rng.integers(-5, 6)
    img[SIZE[0]//2:, cx-2:cx+2] = [255, 255, 200]
    noise = rng.integers(-8, 9, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_urban_realistic(idx):
    rng = np.random.default_rng(idx * 7002)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [135, 206, 235]
    img[SIZE[0]//3:SIZE[0]//2] = [139, 119, 101]
    for bx in range(0, SIZE[1], 40):
        for by in range(SIZE[0]//3 + 5, SIZE[0]//2 - 5, 15):
            if rng.random() > 0.3:
                img[by:by+8, bx+5:bx+15] = [200, 220, 255]
    img[SIZE[0]//2:] = [60, 60, 60]
    noise = rng.integers(-8, 9, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_night_realistic(idx):
    rng = np.random.default_rng(idx * 7003)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [15, 15, 30]
    img[SIZE[0]//2:] = [25, 25, 25]
    cy, cx = SIZE[0]*3//4, SIZE[1]//2
    for r in range(60, 0, -1):
        brightness = int(40 * (1 - r/60))
        for dy in range(-r, r):
            for dx in range(-r*2, r*2):
                py, px = cy + dy, cx + dx
                if 0 <= py < SIZE[0] and 0 <= px < SIZE[1]:
                    img[py, px] = np.clip(img[py, px].astype(int) + brightness, 0, 255)
    noise = rng.integers(-3, 4, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_foggy_realistic(idx):
    rng = np.random.default_rng(idx * 7004)
    img = create_highway_realistic(idx + 2000)
    fog = np.full_like(img, 200)
    alpha = 0.55 + rng.random() * 0.15
    return (alpha * fog + (1 - alpha) * img).astype(np.uint8)

def create_offroad(idx):
    rng = np.random.default_rng(idx * 7010)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]
    for row in range(SIZE[0]//2, SIZE[0]):
        t = (row - SIZE[0]//2) / (SIZE[0]//2)
        g = int(80 + rng.integers(-20, 20))
        img[row, :] = [34 + int(t*20), min(139, g + int(t*30)), 34]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_flooded(idx):
    rng = np.random.default_rng(idx * 7011)
    img = create_highway_realistic(idx + 3000)
    water_start = SIZE[0]*2//3
    for row in range(water_start, SIZE[0]):
        t = (row - water_start) / (SIZE[0] - water_start)
        blue = [int(30 + 60*t), int(60 + 80*t), int(120 + 80*t)]
        img[row, :] = np.clip(
            (0.4 * img[row, :].astype(float) + 0.6 * np.array(blue)).astype(int),
            0, 255).astype(np.uint8)
    return img

def create_tunnel(idx):
    rng = np.random.default_rng(idx * 7012)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:] = [30, 25, 20]
    for col in range(SIZE[1]//4, 3*SIZE[1]//4):
        for row in range(SIZE[0]//4, 3*SIZE[0]//4):
            img[row, col] = [40, 35, 30]
    img[SIZE[0]//2:, SIZE[1]//4:3*SIZE[1]//4] = [50, 50, 50]
    for lx in range(SIZE[1]//4 + 20, 3*SIZE[1]//4, 40):
        img[SIZE[0]//4:SIZE[0]//4+5, lx:lx+5] = [255, 200, 100]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_snow(idx):
    rng = np.random.default_rng(idx * 7013)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [200, 200, 210]
    img[SIZE[0]//2:] = [220, 220, 230]
    for _ in range(200):
        sy, sx = rng.integers(0, SIZE[0]), rng.integers(0, SIZE[1])
        img[sy:sy+2, sx:sx+2] = 255
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def extract_signals(model, processor, image, prompt):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=7, do_sample=False,
            output_scores=True, output_hidden_states=True,
            return_dict_in_generate=True,
        )
    if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
        last_step = outputs.hidden_states[-1]
        if isinstance(last_step, tuple):
            hidden = last_step[-1][0, -1, :].float().cpu().numpy()
        else:
            hidden = last_step[0, -1, :].float().cpu().numpy()
    else:
        hidden = np.zeros(4096)

    vocab_size = outputs.scores[0].shape[-1]
    action_start = vocab_size - 256
    masses = []
    for score in outputs.scores[:7]:
        probs = torch.softmax(score[0].float(), dim=0)
        masses.append(float(probs[action_start:].sum()))

    return {
        'hidden': hidden,
        'action_mass': float(np.mean(masses)),
    }


def cosine_dist(a, b):
    return 1.0 - float(np.dot(a / (np.linalg.norm(a) + 1e-10),
                               b / (np.linalg.norm(b) + 1e-10)))


def main():
    print("=" * 70, flush=True)
    print("OPERATING CHARACTERISTIC CURVES", flush=True)
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

    # Calibration
    print("\nCalibration...", flush=True)
    cal_scenes = {
        'highway': create_highway_realistic,
        'urban': create_urban_realistic,
        'night': create_night_realistic,
        'foggy': create_foggy_realistic,
    }
    cal_per_scene = 8
    cal_data = {scene: [] for scene in cal_scenes}
    all_cal = []

    for scene, fn in cal_scenes.items():
        for i in range(cal_per_scene):
            data = extract_signals(model, processor,
                                    Image.fromarray(fn(i + 9000)), prompt)
            cal_data[scene].append(data)
            all_cal.append(data)

    global_centroid = np.mean([d['hidden'] for d in all_cal], axis=0)
    per_scene_centroids = {
        scene: np.mean([d['hidden'] for d in cal_data[scene]], axis=0)
        for scene in cal_scenes
    }
    print(f"  Calibration complete ({len(all_cal)} samples)", flush=True)

    # Compute calibration scores for conformal threshold
    cal_cos_scores = [cosine_dist(d['hidden'], global_centroid) for d in all_cal]
    cal_scene_scores = [
        min(cosine_dist(d['hidden'], c) for c in per_scene_centroids.values())
        for d in all_cal
    ]

    # Larger test set: 8 trajectories × 5 steps per scene
    print("\nGenerating test set...", flush=True)
    id_scenes = {
        'highway': (create_highway_realistic, False),
        'urban': (create_urban_realistic, False),
        'night': (create_night_realistic, False),
        'foggy': (create_foggy_realistic, False),
    }
    ood_scenes = {
        'offroad': (create_offroad, True),
        'flooded': (create_flooded, True),
        'tunnel': (create_tunnel, True),
        'snow': (create_snow, True),
    }
    all_scenes = {**id_scenes, **ood_scenes}
    n_traj = 8
    traj_len = 5

    trajectories = []
    total = sum(n_traj * traj_len for _ in all_scenes)
    cnt = 0

    for scene, (fn, is_ood) in all_scenes.items():
        for t in range(n_traj):
            traj_frames = []
            for step in range(traj_len):
                cnt += 1
                idx = t * 100 + step + 300
                data = extract_signals(model, processor,
                                        Image.fromarray(fn(idx)), prompt)
                cos_global = cosine_dist(data['hidden'], global_centroid)
                cos_scene = min(
                    cosine_dist(data['hidden'], c) for c in per_scene_centroids.values()
                )
                traj_frames.append({
                    'cos_global': cos_global,
                    'cos_scene': cos_scene,
                    'action_mass': data['action_mass'],
                })
                if cnt % 40 == 0:
                    print(f"  [{cnt}/{total}] {scene}_t{t}_s{step}", flush=True)

            trajectories.append({
                'scenario': scene,
                'is_ood': is_ood,
                'frames': traj_frames,
            })

    labels = [1 if tr['is_ood'] else 0 for tr in trajectories]

    # ===================================================================
    # Generate scores for all methods
    # ===================================================================
    methods = {
        'Global cosine (1f)': [tr['frames'][0]['cos_global'] for tr in trajectories],
        'Per-scene cosine (1f)': [tr['frames'][0]['cos_scene'] for tr in trajectories],
        'Action mass (1f)': [1 - tr['frames'][0]['action_mass'] for tr in trajectories],
        'Per-scene cosine (5s)': [np.mean([f['cos_scene'] for f in tr['frames']]) for tr in trajectories],
        'Full pipeline': [
            0.7 * np.mean([f['cos_scene'] for f in tr['frames']]) +
            0.3 * np.mean([1 - f['action_mass'] for f in tr['frames']])
            for tr in trajectories
        ],
    }

    # ===================================================================
    # Analysis
    # ===================================================================
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    # 1. AUROC
    print("\n1. AUROC Summary", flush=True)
    print("-" * 60, flush=True)

    roc_data = {}
    for name, scores in methods.items():
        auroc = roc_auc_score(labels, scores)
        fpr, tpr, thresholds = roc_curve(labels, scores)
        prec, recall, pr_thresholds = precision_recall_curve(labels, scores)
        ap = average_precision_score(labels, scores)

        # Find operating points
        # 95% TPR
        idx_95 = np.argmin(np.abs(tpr - 0.95))
        fpr_at_95tpr = fpr[idx_95]

        # 99% TPR
        idx_99 = np.argmin(np.abs(tpr - 0.99))
        fpr_at_99tpr = fpr[idx_99]

        roc_data[name] = {
            'auroc': float(auroc),
            'ap': float(ap),
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist(),
            'prec': prec.tolist(),
            'recall': recall.tolist(),
            'fpr_at_95tpr': float(fpr_at_95tpr),
            'fpr_at_99tpr': float(fpr_at_99tpr),
        }

        print(f"  {name:<30} AUROC={auroc:.3f}  AP={ap:.3f}  "
              f"FPR@95TPR={fpr_at_95tpr:.3f}  FPR@99TPR={fpr_at_99tpr:.3f}", flush=True)

    # 2. Coverage-safety tradeoff
    print("\n2. Coverage-Safety Tradeoff (Full Pipeline)", flush=True)
    print("-" * 60, flush=True)

    full_scores = np.array(methods['Full pipeline'])
    labels_arr = np.array(labels)
    id_scores = full_scores[labels_arr == 0]
    ood_scores = full_scores[labels_arr == 1]

    # Different threshold percentiles from calibration
    cal_full = [
        0.7 * min(cosine_dist(d['hidden'], c) for c in per_scene_centroids.values()) +
        0.3 * (1 - d['action_mass'])
        for d in all_cal
    ]

    print(f"\n  {'Alpha':>8} {'Threshold':>10} {'Coverage':>10} {'Safety':>10} {'FPR':>8} {'FNR':>8}",
          flush=True)
    print("  " + "-" * 58, flush=True)

    operating_points = {}
    for alpha in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        threshold = np.quantile(cal_full, 1 - alpha)
        n_id_pass = np.sum(id_scores <= threshold)
        n_ood_caught = np.sum(ood_scores > threshold)
        coverage = n_id_pass / len(id_scores)
        safety = n_ood_caught / len(ood_scores)
        fpr = 1 - coverage  # false positive rate = flagging ID as OOD
        fnr = 1 - safety  # false negative rate = missing OOD

        operating_points[str(alpha)] = {
            'threshold': float(threshold),
            'coverage': float(coverage),
            'safety': float(safety),
            'fpr': float(fpr),
            'fnr': float(fnr),
        }

        print(f"  {alpha:>8.2f} {threshold:>10.4f} {coverage:>10.3f} {safety:>10.3f} "
              f"{fpr:>8.3f} {fnr:>8.3f}", flush=True)

    # 3. Youden's J statistic for optimal threshold
    print("\n3. Optimal Threshold (Youden's J)", flush=True)
    print("-" * 60, flush=True)

    fpr_full, tpr_full, thresholds_full = roc_curve(labels, methods['Full pipeline'])
    j_scores = tpr_full - fpr_full
    best_j_idx = np.argmax(j_scores)
    best_threshold = thresholds_full[best_j_idx]
    best_j = j_scores[best_j_idx]

    print(f"  Best threshold: {best_threshold:.4f}", flush=True)
    print(f"  Youden's J: {best_j:.3f}", flush=True)
    print(f"  TPR at best: {tpr_full[best_j_idx]:.3f}", flush=True)
    print(f"  FPR at best: {fpr_full[best_j_idx]:.3f}", flush=True)

    # 4. EER (Equal Error Rate)
    print("\n4. Equal Error Rate", flush=True)
    print("-" * 60, flush=True)

    fnr_full = 1 - tpr_full
    eer_idx = np.argmin(np.abs(fpr_full - fnr_full))
    eer = (fpr_full[eer_idx] + fnr_full[eer_idx]) / 2
    print(f"  EER: {eer:.3f}", flush=True)
    print(f"  FPR at EER: {fpr_full[eer_idx]:.3f}", flush=True)
    print(f"  FNR at EER: {fnr_full[eer_idx]:.3f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'operating_curves',
        'experiment_number': 52,
        'timestamp': timestamp,
        'n_cal': len(all_cal),
        'n_trajectories': len(trajectories),
        'n_id': int(np.sum(labels_arr == 0)),
        'n_ood': int(np.sum(labels_arr == 1)),
        'roc_data': {k: {kk: vv for kk, vv in v.items() if kk not in ['fpr', 'tpr', 'thresholds', 'prec', 'recall']}
                     for k, v in roc_data.items()},
        'operating_points': operating_points,
        'optimal_threshold': {
            'value': float(best_threshold),
            'youdens_j': float(best_j),
            'tpr': float(tpr_full[best_j_idx]),
            'fpr': float(fpr_full[best_j_idx]),
        },
        'eer': float(eer),
        'full_roc': {
            'fpr': fpr_full.tolist(),
            'tpr': tpr_full.tolist(),
        },
    }
    output_path = os.path.join(RESULTS_DIR, f"operating_curves_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
