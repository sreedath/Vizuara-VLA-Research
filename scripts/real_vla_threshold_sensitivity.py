"""
Detection Threshold Sensitivity Analysis.

Comprehensive analysis of how the detection threshold affects:
1. True positive rate (recall) and false positive rate
2. Precision-recall tradeoff at different operating points
3. Coverage (fraction of inputs we act on) vs safety (fraction of OOD caught)
4. Practical threshold selection strategies

Experiment 61 in the CalibDrive series.
"""
import os
import json
import datetime
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve

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
    img[SIZE[0]//2:] = [139, 90, 43]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_inverted(idx):
    return 255 - create_highway(idx + 3000)

def create_blackout(idx):
    return np.zeros((*SIZE, 3), dtype=np.uint8)


def extract_hidden(model, processor, image, prompt):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=7, do_sample=False,
            output_hidden_states=True, output_scores=True,
            return_dict_in_generate=True,
        )
    # Hidden state
    if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
        last_step = outputs.hidden_states[-1]
        if isinstance(last_step, tuple):
            hidden = last_step[-1][0, -1, :].float().cpu().numpy()
        else:
            hidden = last_step[0, -1, :].float().cpu().numpy()
    else:
        hidden = np.zeros(4096)

    # Action mass
    vocab_size = outputs.scores[0].shape[-1]
    action_start = vocab_size - 256
    masses = []
    for score in outputs.scores[:7]:
        probs = torch.softmax(score[0].float(), dim=0)
        masses.append(float(probs[action_start:].sum()))
    action_mass = np.mean(masses)

    return hidden, action_mass


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

    # Calibration
    print("\nCalibrating...", flush=True)
    cal_hidden = []
    cal_masses = []
    for fn in [create_highway, create_urban]:
        for i in range(15):
            h, m = extract_hidden(model, processor,
                                  Image.fromarray(fn(i + 9000)), prompt)
            cal_hidden.append(h)
            cal_masses.append(m)
    centroid = np.mean(cal_hidden, axis=0)
    cal_cosine = [cosine_dist(h, centroid) for h in cal_hidden]
    print(f"  Calibration: {len(cal_hidden)} samples", flush=True)
    print(f"  Cal cosine: {np.mean(cal_cosine):.4f} ± {np.std(cal_cosine):.4f}", flush=True)
    print(f"  Cal mass: {np.mean(cal_masses):.4f} ± {np.std(cal_masses):.4f}", flush=True)

    # Test set (larger for better threshold analysis)
    print("\nCollecting test set...", flush=True)
    test_fns = {
        'highway': (create_highway, False, 15),
        'urban': (create_urban, False, 15),
        'noise': (create_noise, True, 10),
        'indoor': (create_indoor, True, 10),
        'inverted': (create_inverted, True, 10),
        'blackout': (create_blackout, True, 10),
    }

    test_data = []
    cnt = 0
    total = sum(v[2] for v in test_fns.values())
    for scene, (fn, is_ood, n) in test_fns.items():
        for i in range(n):
            cnt += 1
            h, m = extract_hidden(model, processor,
                                  Image.fromarray(fn(i + 200)), prompt)
            cos = cosine_dist(h, centroid)
            test_data.append({
                'scenario': scene, 'is_ood': is_ood,
                'cosine': cos, 'mass': m,
            })
            if cnt % 10 == 0:
                print(f"  [{cnt}/{total}] {scene}_{i}", flush=True)

    labels = [d['is_ood'] for d in test_data]
    labels_int = [1 if l else 0 for l in labels]

    # Combined score
    cos_scores = np.array([d['cosine'] for d in test_data])
    mass_scores = np.array([1 - d['mass'] for d in test_data])  # Invert so higher = more OOD
    cos_norm = (cos_scores - cos_scores.min()) / (cos_scores.max() - cos_scores.min() + 1e-10)
    mass_norm = (mass_scores - mass_scores.min()) / (mass_scores.max() - mass_scores.min() + 1e-10)
    combined = 0.7 * cos_norm + 0.3 * mass_norm

    print("\n" + "=" * 70, flush=True)
    print("THRESHOLD ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    for score_name, scores in [('cosine', cos_scores), ('1-mass', mass_scores),
                                ('combined', combined)]:
        auroc = roc_auc_score(labels_int, scores)
        print(f"\n  {score_name} (AUROC={auroc:.3f}):", flush=True)

        # ROC curve
        fpr, tpr, roc_thresholds = roc_curve(labels_int, scores)
        # Find operating points
        print(f"    {'Threshold':>10} {'TPR':>8} {'FPR':>8} {'Precision':>10} {'F1':>8}", flush=True)
        print("    " + "-" * 50, flush=True)

        # Quantile-based thresholds from calibration
        id_scores = [s for s, l in zip(scores, labels) if not l]
        ood_scores = [s for s, l in zip(scores, labels) if l]

        percentiles = [50, 75, 90, 95, 99]
        for p in percentiles:
            thresh = np.percentile(id_scores, p)
            tp = sum(1 for s, l in zip(scores, labels) if s >= thresh and l)
            fp = sum(1 for s, l in zip(scores, labels) if s >= thresh and not l)
            fn = sum(1 for s, l in zip(scores, labels) if s < thresh and l)
            tn = sum(1 for s, l in zip(scores, labels) if s < thresh and not l)
            tpr_val = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * prec * tpr_val / (prec + tpr_val) if (prec + tpr_val) > 0 else 0
            print(f"    p{p:>2}={thresh:.4f} {tpr_val:>8.3f} {fpr_val:>8.3f} "
                  f"{prec:>10.3f} {f1:>8.3f}", flush=True)

        # Youden's J
        j_scores = tpr - fpr
        best_j_idx = np.argmax(j_scores)
        best_j_thresh = roc_thresholds[best_j_idx]
        print(f"\n    Youden's J optimal: thresh={best_j_thresh:.4f}, "
              f"TPR={tpr[best_j_idx]:.3f}, FPR={fpr[best_j_idx]:.3f}, "
              f"J={j_scores[best_j_idx]:.3f}", flush=True)

    # Per-scenario threshold analysis
    print("\n  Per-scenario catch rate at different thresholds:", flush=True)
    p95_thresh = np.percentile([d['cosine'] for d in test_data if not d['is_ood']], 95)
    p90_thresh = np.percentile([d['cosine'] for d in test_data if not d['is_ood']], 90)
    j_idx = np.argmax(tpr - fpr)
    j_thresh = roc_thresholds[j_idx]  # From combined score

    for scene in ['noise', 'indoor', 'inverted', 'blackout']:
        scene_scores = [d['cosine'] for d in test_data if d['scenario'] == scene]
        caught_p90 = sum(1 for s in scene_scores if s >= p90_thresh) / len(scene_scores)
        caught_p95 = sum(1 for s in scene_scores if s >= p95_thresh) / len(scene_scores)
        print(f"    {scene:<12}: p90={caught_p90:.0%}, p95={caught_p95:.0%} "
              f"(mean={np.mean(scene_scores):.4f})", flush=True)

    # Conformal prediction thresholds
    print("\n  Conformal prediction thresholds:", flush=True)
    for alpha in [0.01, 0.05, 0.10, 0.20]:
        q = np.quantile(cal_cosine, 1 - alpha)
        caught = sum(1 for d in test_data if d['is_ood'] and d['cosine'] >= q)
        total_ood = sum(1 for d in test_data if d['is_ood'])
        false_alarm = sum(1 for d in test_data if not d['is_ood'] and d['cosine'] >= q)
        total_id = sum(1 for d in test_data if not d['is_ood'])
        print(f"    α={alpha:.2f}: q={q:.4f}, OOD caught={caught}/{total_ood} ({caught/total_ood:.0%}), "
              f"false alarm={false_alarm}/{total_id} ({false_alarm/total_id:.0%})", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'threshold_sensitivity',
        'experiment_number': 61,
        'timestamp': timestamp,
        'n_cal': len(cal_hidden),
        'n_test': len(test_data),
        'auroc_cosine': float(roc_auc_score(labels_int, cos_scores)),
        'auroc_mass': float(roc_auc_score(labels_int, mass_scores)),
        'auroc_combined': float(roc_auc_score(labels_int, combined)),
        'cal_cosine_mean': float(np.mean(cal_cosine)),
        'cal_cosine_std': float(np.std(cal_cosine)),
        'test_data': [{'scenario': d['scenario'], 'is_ood': d['is_ood'],
                        'cosine': float(d['cosine']), 'mass': float(d['mass'])}
                       for d in test_data],
    }
    output_path = os.path.join(RESULTS_DIR, f"threshold_sensitivity_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
