"""
Feature Correlation Analysis.

Computes the full correlation matrix between all OOD detection
features (cosine distance, attention max, output entropy, top-1
probability, attention entropy) to identify redundant vs complementary
features and understand the latent structure of OOD signals.

Experiment 83 in the CalibDrive series.
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

def create_blackout(idx):
    return np.zeros((*SIZE, 3), dtype=np.uint8)


def extract_all_features(model, processor, image, prompt, centroid):
    """Extract all OOD detection features from a single forward pass."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True, output_attentions=True)

    features = {}

    # 1. Hidden state cosine distance
    if hasattr(fwd, 'hidden_states') and fwd.hidden_states:
        h = fwd.hidden_states[-1][0, -1, :].float().cpu().numpy()
        cos_dist = 1.0 - float(np.dot(h / (np.linalg.norm(h) + 1e-10),
                                       centroid / (np.linalg.norm(centroid) + 1e-10)))
        features['cosine_dist'] = cos_dist
        features['hidden_norm'] = float(np.linalg.norm(h))

    # 2. Attention features
    if hasattr(fwd, 'attentions') and fwd.attentions:
        last_attn = fwd.attentions[-1][0].float().cpu().numpy()
        features['attn_max'] = float(last_attn[:, -1, :].max())
        features['attn_mean'] = float(last_attn[:, -1, :].mean())
        attn_last = last_attn[:, -1, :]
        attn_flat = attn_last.flatten()
        attn_flat = attn_flat / (attn_flat.sum() + 1e-10)
        features['attn_entropy'] = -float(np.sum(attn_flat * np.log(attn_flat + 1e-10)))

    # 3. Output logit features
    logits = fwd.logits[0, -1, :].float().cpu().numpy()
    probs = np.exp(logits - np.max(logits))
    probs = probs / probs.sum()
    features['output_entropy'] = -float(np.sum(probs * np.log(probs + 1e-10)))
    features['top1_prob'] = float(np.max(probs))
    top5_probs = np.sort(probs)[-5:]
    features['top5_prob_sum'] = float(top5_probs.sum())

    return features


def main():
    print("=" * 70, flush=True)
    print("FEATURE CORRELATION ANALYSIS", flush=True)
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

    # Calibrate for cosine distance
    print("\nCalibrating...", flush=True)
    cal_hidden = []
    for fn in [create_highway, create_urban]:
        for i in range(15):
            inputs = processor(prompt, Image.fromarray(fn(i + 9000))).to(
                model.device, dtype=torch.bfloat16)
            with torch.no_grad():
                fwd = model(**inputs, output_hidden_states=True)
            if hasattr(fwd, 'hidden_states') and fwd.hidden_states:
                h = fwd.hidden_states[-1][0, -1, :].float().cpu().numpy()
                cal_hidden.append(h)
    centroid = np.mean(cal_hidden, axis=0)
    print(f"  {len(cal_hidden)} calibration samples", flush=True)

    # Collect features for all test images
    test_configs = [
        ('highway', create_highway, range(500, 512), False),
        ('urban', create_urban, range(500, 512), False),
        ('noise', create_noise, range(500, 510), True),
        ('indoor', create_indoor, range(500, 510), True),
        ('twilight', create_twilight_highway, range(500, 510), True),
        ('snow', create_snow, range(500, 510), True),
        ('blackout', create_blackout, range(500, 506), True),
    ]

    all_features = []
    all_labels = []
    all_scenarios = []
    cnt = 0
    total = sum(len(list(ids)) for _, _, ids, _ in test_configs)

    print("\nExtracting features...", flush=True)
    for name, fn, indices, is_ood in test_configs:
        for i in indices:
            cnt += 1
            feats = extract_all_features(model, processor,
                                          Image.fromarray(fn(i)), prompt, centroid)
            all_features.append(feats)
            all_labels.append(1 if is_ood else 0)
            all_scenarios.append(name)
            if cnt % 10 == 0:
                print(f"  [{cnt}/{total}] {name}", flush=True)

    print(f"\n  Total samples: {len(all_features)}", flush=True)

    # Build feature matrix
    feature_names = ['cosine_dist', 'attn_max', 'attn_entropy', 'output_entropy',
                     'top1_prob', 'top5_prob_sum', 'hidden_norm', 'attn_mean']
    n = len(all_features)
    matrix = np.zeros((n, len(feature_names)))
    for i, feats in enumerate(all_features):
        for j, fname in enumerate(feature_names):
            matrix[i, j] = feats.get(fname, 0.0)

    # Compute correlation matrix
    corr = np.corrcoef(matrix.T)

    print("\nCorrelation Matrix:", flush=True)
    header = "           " + "  ".join(f"{fn[:8]:>8}" for fn in feature_names)
    print(header, flush=True)
    for i, fn in enumerate(feature_names):
        row = f"  {fn[:10]:<10}"
        for j in range(len(feature_names)):
            row += f"  {corr[i,j]:>8.3f}"
        print(row, flush=True)

    # Per-feature AUROC
    labels = np.array(all_labels)
    print("\nPer-Feature AUROC:", flush=True)
    feature_aurocs = {}
    for j, fname in enumerate(feature_names):
        scores = matrix[:, j]
        try:
            auroc = roc_auc_score(labels, scores)
            # For features where lower = OOD, flip
            auroc_flipped = roc_auc_score(labels, -scores)
            best_auroc = max(auroc, auroc_flipped)
            direction = "higher=OOD" if auroc >= auroc_flipped else "lower=OOD"
            feature_aurocs[fname] = {
                'auroc': float(best_auroc),
                'direction': direction,
            }
            print(f"  {fname:<16}: AUROC={best_auroc:.3f} ({direction})", flush=True)
        except Exception as e:
            print(f"  {fname:<16}: error - {e}", flush=True)
            feature_aurocs[fname] = {'auroc': 0.5, 'direction': 'unknown'}

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'feature_correlation',
        'experiment_number': 83,
        'timestamp': timestamp,
        'n_samples': len(all_features),
        'n_id': int((labels == 0).sum()),
        'n_ood': int((labels == 1).sum()),
        'feature_names': feature_names,
        'correlation_matrix': corr.tolist(),
        'feature_aurocs': feature_aurocs,
        'per_scenario_means': {},
    }

    # Per-scenario means
    for name in set(all_scenarios):
        mask = [s == name for s in all_scenarios]
        scenario_matrix = matrix[mask]
        means = {fn: float(scenario_matrix[:, j].mean()) for j, fn in enumerate(feature_names)}
        output['per_scenario_means'][name] = means

    output_path = os.path.join(RESULTS_DIR, f"feature_correlation_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
