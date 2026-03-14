"""
Activation Statistics Analysis.

Tests whether activation-level statistics differ between ID and OOD:
sparsity (fraction of zero/near-zero activations), kurtosis,
skewness, mean, variance, and top-k activation patterns.

Experiment 102 in the CalibDrive series.
"""
import os
import json
import datetime
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from scipy import stats as scipy_stats

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


def extract_activation_stats(model, processor, image, prompt):
    """Extract last hidden state and compute activation statistics."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)

    if not hasattr(fwd, 'hidden_states') or not fwd.hidden_states:
        return None

    hidden = fwd.hidden_states[-1][0, -1, :].float().cpu().numpy()

    # Activation statistics
    abs_h = np.abs(hidden)
    stats = {
        'hidden': hidden,
        'mean': float(np.mean(hidden)),
        'std': float(np.std(hidden)),
        'abs_mean': float(np.mean(abs_h)),
        'sparsity_0': float(np.mean(abs_h < 0.01)),
        'sparsity_01': float(np.mean(abs_h < 0.1)),
        'sparsity_1': float(np.mean(abs_h < 1.0)),
        'max_abs': float(np.max(abs_h)),
        'kurtosis': float(scipy_stats.kurtosis(hidden)),
        'skewness': float(scipy_stats.skew(hidden)),
        'top10_mean': float(np.sort(abs_h)[-10:].mean()),
        'top100_mean': float(np.sort(abs_h)[-100:].mean()),
        'positive_frac': float(np.mean(hidden > 0)),
        'l1_norm': float(np.linalg.norm(hidden, ord=1)),
        'l2_norm': float(np.linalg.norm(hidden, ord=2)),
    }
    return stats


def main():
    print("=" * 70, flush=True)
    print("ACTIVATION STATISTICS ANALYSIS", flush=True)
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
    }

    all_stats = {}
    for cat_name, (fn, group) in categories.items():
        print(f"\n  {cat_name} ({group})...", flush=True)
        cat_stats = []
        for i in range(10):
            stats = extract_activation_stats(model, processor,
                                             Image.fromarray(fn(i + 500)), prompt)
            if stats is not None:
                cat_stats.append({k: v for k, v in stats.items() if k != 'hidden'})
        all_stats[cat_name] = {'group': group, 'stats': cat_stats}

        # Print averages
        for key in ['mean', 'std', 'abs_mean', 'sparsity_0', 'kurtosis', 'skewness', 'l2_norm']:
            vals = [s[key] for s in cat_stats]
            print(f"    {key}: {np.mean(vals):.4f}±{np.std(vals):.4f}", flush=True)

    # Per-feature AUROC
    print("\n--- Per-feature AUROC ---", flush=True)
    id_stats = []
    ood_stats = []
    for cat_name, data in all_stats.items():
        if data['group'] == 'ID':
            id_stats.extend(data['stats'])
        else:
            ood_stats.extend(data['stats'])

    labels = [0]*len(id_stats) + [1]*len(ood_stats)
    feature_aurocs = {}
    for key in ['mean', 'std', 'abs_mean', 'sparsity_0', 'sparsity_01', 'sparsity_1',
                'max_abs', 'kurtosis', 'skewness', 'top10_mean', 'top100_mean',
                'positive_frac', 'l1_norm', 'l2_norm']:
        id_vals = [s[key] for s in id_stats]
        ood_vals = [s[key] for s in ood_stats]
        # Try both directions
        auroc_pos = roc_auc_score(labels, id_vals + ood_vals)
        auroc_neg = roc_auc_score(labels, [-v for v in id_vals] + [-v for v in ood_vals])
        auroc = max(auroc_pos, auroc_neg)
        direction = 'higher=OOD' if auroc_pos >= auroc_neg else 'lower=OOD'
        feature_aurocs[key] = {
            'auroc': float(auroc),
            'direction': direction,
            'id_mean': float(np.mean(id_vals)),
            'ood_mean': float(np.mean(ood_vals)),
        }
        print(f"  {key:<15}: AUROC={auroc:.3f} ({direction})", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'activation_stats',
        'experiment_number': 102,
        'timestamp': timestamp,
        'categories': {k: {
            'group': v['group'],
            'stats_summary': {
                key: {'mean': float(np.mean([s[key] for s in v['stats']])),
                      'std': float(np.std([s[key] for s in v['stats']]))}
                for key in v['stats'][0].keys()
            }
        } for k, v in all_stats.items()},
        'feature_aurocs': feature_aurocs,
    }
    output_path = os.path.join(RESULTS_DIR, f"activation_stats_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
