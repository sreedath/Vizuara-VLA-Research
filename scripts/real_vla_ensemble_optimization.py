"""
Ensemble Weight Optimization.

Grid search over cosine distance, attention max, and embedding norm
weights to find optimal ensemble for OOD detection. Tests whether
combining multiple signals provides additive benefit.

Experiment 98 in the CalibDrive series.
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


def extract_features(model, processor, image, prompt):
    """Extract hidden state and attention features."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True, output_attentions=True)

    result = {}
    if hasattr(fwd, 'hidden_states') and fwd.hidden_states:
        hidden = fwd.hidden_states[-1][0, -1, :].float().cpu().numpy()
        result['hidden'] = hidden
        result['norm'] = float(np.linalg.norm(hidden))

    if hasattr(fwd, 'attentions') and fwd.attentions:
        last_attn = fwd.attentions[-1][0].float().cpu().numpy()
        result['attn_max'] = float(np.max(last_attn[:, -1, :]))
        attn_row = last_attn[:, -1, :].flatten()
        attn_row = attn_row / (attn_row.sum() + 1e-10)
        result['attn_entropy'] = float(-np.sum(attn_row * np.log(attn_row + 1e-10)))

    return result


def cosine_dist(a, b):
    return 1.0 - float(np.dot(a / (np.linalg.norm(a) + 1e-10),
                               b / (np.linalg.norm(b) + 1e-10)))


def main():
    print("=" * 70, flush=True)
    print("ENSEMBLE WEIGHT OPTIMIZATION", flush=True)
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

    # Collect features
    print("\nCollecting features...", flush=True)
    cal_features = []
    for fn in [create_highway, create_urban]:
        for i in range(10):
            feat = extract_features(model, processor,
                                     Image.fromarray(fn(i + 9000)), prompt)
            cal_features.append(feat)

    id_features = []
    for fn in [create_highway, create_urban]:
        for i in range(10):
            feat = extract_features(model, processor,
                                     Image.fromarray(fn(i + 500)), prompt)
            id_features.append(feat)

    ood_features = []
    ood_labels = []
    for fn, name in [(create_noise, 'noise'), (create_indoor, 'indoor'),
                     (create_twilight_highway, 'twilight'), (create_snow, 'snow')]:
        for i in range(8):
            feat = extract_features(model, processor,
                                     Image.fromarray(fn(i + 500)), prompt)
            ood_features.append(feat)
            ood_labels.append(name)

    print(f"  Cal: {len(cal_features)}, ID: {len(id_features)}, OOD: {len(ood_features)}", flush=True)

    # Compute per-feature scores
    centroid = np.mean([f['hidden'] for f in cal_features], axis=0)
    cal_norms = np.array([f['norm'] for f in cal_features])
    cal_mean_norm = cal_norms.mean()

    labels = [0]*len(id_features) + [1]*len(ood_features)

    # Feature 1: Cosine distance
    id_cos = np.array([cosine_dist(f['hidden'], centroid) for f in id_features])
    ood_cos = np.array([cosine_dist(f['hidden'], centroid) for f in ood_features])

    # Feature 2: Attention max (negate — higher max = more focused = ID-like)
    id_attn = np.array([f.get('attn_max', 0.5) for f in id_features])
    ood_attn = np.array([f.get('attn_max', 0.5) for f in ood_features])

    # Feature 3: Norm deviation
    id_norm = np.array([abs(f['norm'] - cal_mean_norm) for f in id_features])
    ood_norm = np.array([abs(f['norm'] - cal_mean_norm) for f in ood_features])

    # Z-score normalize
    def zscore(id_vals, ood_vals):
        all_vals = np.concatenate([id_vals, ood_vals])
        mu, sigma = all_vals.mean(), all_vals.std() + 1e-10
        return (id_vals - mu) / sigma, (ood_vals - mu) / sigma

    id_cos_z, ood_cos_z = zscore(id_cos, ood_cos)
    id_attn_z, ood_attn_z = zscore(id_attn, ood_attn)
    id_norm_z, ood_norm_z = zscore(id_norm, ood_norm)

    # Individual AUROCs
    cos_auroc = roc_auc_score(labels, list(id_cos) + list(ood_cos))
    # For attn_max, lower values might indicate OOD — test both directions
    attn_auroc_pos = roc_auc_score(labels, list(id_attn) + list(ood_attn))
    attn_auroc_neg = roc_auc_score(labels, list(-id_attn) + list(-ood_attn))
    attn_auroc = max(attn_auroc_pos, attn_auroc_neg)
    attn_sign = 1.0 if attn_auroc_pos >= attn_auroc_neg else -1.0

    norm_auroc = roc_auc_score(labels, list(id_norm) + list(ood_norm))

    print(f"\n  Individual AUROCs:", flush=True)
    print(f"    Cosine: {cos_auroc:.3f}", flush=True)
    print(f"    Attn max: {attn_auroc:.3f} (sign={attn_sign})", flush=True)
    print(f"    Norm dev: {norm_auroc:.3f}", flush=True)

    # Grid search over weights
    print("\n  Grid search...", flush=True)
    grid_results = []
    for w_cos in np.arange(0, 1.05, 0.1):
        for w_attn in np.arange(0, 1.05 - w_cos, 0.1):
            w_norm = round(1.0 - w_cos - w_attn, 2)
            if w_norm < -0.01:
                continue

            id_ensemble = w_cos * id_cos_z + w_attn * attn_sign * id_attn_z + w_norm * id_norm_z
            ood_ensemble = w_cos * ood_cos_z + w_attn * attn_sign * ood_attn_z + w_norm * ood_norm_z

            auroc = roc_auc_score(labels, list(id_ensemble) + list(ood_ensemble))
            id_arr = id_ensemble
            ood_arr = ood_ensemble
            pooled = np.sqrt((id_arr.var() + ood_arr.var()) / 2)
            d = (ood_arr.mean() - id_arr.mean()) / (pooled + 1e-10)

            grid_results.append({
                'w_cos': round(float(w_cos), 2),
                'w_attn': round(float(w_attn), 2),
                'w_norm': round(float(w_norm), 2),
                'auroc': float(auroc),
                'cohens_d': float(d),
            })

    # Sort by Cohen's d
    grid_results.sort(key=lambda x: x['cohens_d'], reverse=True)
    print(f"\n  Top 10 ensembles by Cohen's d:", flush=True)
    for r in grid_results[:10]:
        print(f"    cos={r['w_cos']:.1f} attn={r['w_attn']:.1f} norm={r['w_norm']:.1f}: "
              f"AUROC={r['auroc']:.3f}, d={r['cohens_d']:.2f}", flush=True)

    print(f"\n  Bottom 5:", flush=True)
    for r in grid_results[-5:]:
        print(f"    cos={r['w_cos']:.1f} attn={r['w_attn']:.1f} norm={r['w_norm']:.1f}: "
              f"AUROC={r['auroc']:.3f}, d={r['cohens_d']:.2f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'ensemble_optimization',
        'experiment_number': 98,
        'timestamp': timestamp,
        'n_cal': len(cal_features),
        'n_id': len(id_features),
        'n_ood': len(ood_features),
        'individual_aurocs': {
            'cosine': float(cos_auroc),
            'attn_max': float(attn_auroc),
            'norm_deviation': float(norm_auroc),
            'attn_sign': float(attn_sign),
        },
        'grid_results': grid_results,
        'top_10': grid_results[:10],
        'best': grid_results[0],
    }
    output_path = os.path.join(RESULTS_DIR, f"ensemble_optimization_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
