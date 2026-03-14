"""
Temperature Scaling Effect on OOD Detection.

Tests how temperature scaling of the output logits affects OOD
detection via entropy and probability-based features. Also compares
whether temperature scaling interacts with hidden-state features.

Experiment 93 in the CalibDrive series.
"""
import os
import json
import datetime
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

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


def extract_features(model, processor, image, prompt, temperatures):
    """Extract hidden state and temperature-scaled output features."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)

    result = {}

    # Hidden state
    if hasattr(fwd, 'hidden_states') and fwd.hidden_states:
        result['hidden'] = fwd.hidden_states[-1][0, -1, :].float().cpu().numpy()

    # Logits at last position
    logits = fwd.logits[0, -1, :].float().cpu()

    for T in temperatures:
        scaled = logits / T
        probs = torch.softmax(scaled, dim=0).numpy()
        entropy = -float(np.sum(probs * np.log(probs + 1e-10)))
        top_prob = float(np.max(probs))
        top5_prob = float(np.sort(probs)[-5:].sum())
        result[f'T_{T}'] = {
            'entropy': entropy,
            'top_prob': top_prob,
            'top5_prob': top5_prob,
        }

    return result


def cosine_dist(a, b):
    return 1.0 - float(np.dot(a / (np.linalg.norm(a) + 1e-10),
                               b / (np.linalg.norm(b) + 1e-10)))


def main():
    print("=" * 70, flush=True)
    print("TEMPERATURE SCALING EFFECT ON OOD DETECTION", flush=True)
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
    temperatures = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]

    # Collect features
    print("\nCollecting calibration features...", flush=True)
    cal_features = []
    for fn in [create_highway, create_urban]:
        for i in range(10):
            feat = extract_features(model, processor,
                                     Image.fromarray(fn(i + 9000)), prompt, temperatures)
            cal_features.append(feat)
    print(f"  Calibration: {len(cal_features)}", flush=True)

    print("\nCollecting ID test features...", flush=True)
    id_features = []
    for fn in [create_highway, create_urban]:
        for i in range(8):
            feat = extract_features(model, processor,
                                     Image.fromarray(fn(i + 500)), prompt, temperatures)
            id_features.append(feat)

    print("\nCollecting OOD test features...", flush=True)
    ood_features = []
    ood_labels = []
    for fn, name in [(create_noise, 'noise'), (create_indoor, 'indoor'),
                     (create_twilight_highway, 'twilight'), (create_snow, 'snow')]:
        for i in range(6):
            feat = extract_features(model, processor,
                                     Image.fromarray(fn(i + 500)), prompt, temperatures)
            ood_features.append(feat)
            ood_labels.append(name)

    print(f"  ID: {len(id_features)}, OOD: {len(ood_features)}", flush=True)

    # Evaluate hidden state baseline
    cal_hidden = [f['hidden'] for f in cal_features]
    centroid = np.mean(cal_hidden, axis=0)

    id_cosine = [cosine_dist(f['hidden'], centroid) for f in id_features]
    ood_cosine = [cosine_dist(f['hidden'], centroid) for f in ood_features]

    labels = [0]*len(id_cosine) + [1]*len(ood_cosine)
    hidden_auroc = roc_auc_score(labels, id_cosine + ood_cosine)

    results = {
        'hidden_state_auroc': float(hidden_auroc),
    }

    # Evaluate each temperature for each output feature
    print("\nEvaluating temperature scaling...", flush=True)
    temp_results = {}
    for T in temperatures:
        key = f'T_{T}'
        for feat_name in ['entropy', 'top_prob', 'top5_prob']:
            id_scores = [f[key][feat_name] for f in id_features]
            ood_scores = [f[key][feat_name] for f in ood_features]

            # For entropy, higher = more uncertain = OOD
            # For top_prob/top5_prob, lower = more uncertain = OOD (negate)
            if feat_name in ['top_prob', 'top5_prob']:
                id_neg = [-s for s in id_scores]
                ood_neg = [-s for s in ood_scores]
                auroc = roc_auc_score(labels, id_neg + ood_neg)
            else:
                auroc = roc_auc_score(labels, id_scores + ood_scores)

            id_arr = np.array(id_scores)
            ood_arr = np.array(ood_scores)

            temp_results[f'{key}_{feat_name}'] = {
                'auroc': float(auroc),
                'id_mean': float(id_arr.mean()),
                'id_std': float(id_arr.std()),
                'ood_mean': float(ood_arr.mean()),
                'ood_std': float(ood_arr.std()),
            }
            print(f"  T={T:<5} {feat_name:<10}: AUROC={auroc:.3f}, ID={id_arr.mean():.3f}±{id_arr.std():.3f}, OOD={ood_arr.mean():.3f}±{ood_arr.std():.3f}", flush=True)

    results['temperature_results'] = temp_results

    # Find optimal temperature per feature
    for feat_name in ['entropy', 'top_prob', 'top5_prob']:
        best_T = None
        best_auroc = 0
        for T in temperatures:
            key = f'T_{T}_{feat_name}'
            if temp_results[key]['auroc'] > best_auroc:
                best_auroc = temp_results[key]['auroc']
                best_T = T
        results[f'best_T_{feat_name}'] = {'temperature': best_T, 'auroc': best_auroc}
        print(f"\n  Best T for {feat_name}: T={best_T}, AUROC={best_auroc:.3f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'temperature_scaling',
        'experiment_number': 93,
        'timestamp': timestamp,
        'n_cal': len(cal_features),
        'n_id': len(id_features),
        'n_ood': len(ood_features),
        'temperatures': temperatures,
        'results': results,
    }
    output_path = os.path.join(RESULTS_DIR, f"temperature_scaling_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
