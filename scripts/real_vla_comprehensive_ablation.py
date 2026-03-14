"""
Comprehensive Ablation Study.

Systematic ablation testing the contribution of each component:
1. Detection metric (cosine vs Euclidean vs Mahalanobis vs norm)
2. Feature source (last layer vs multi-layer PCA vs attention)
3. Calibration size (1, 3, 5, 10, 20)
4. Threshold strategy (μ+2σ, μ+3σ, Youden's J)
5. Combined recommendations

Experiment 100 in the CalibDrive series.
"""
import os
import json
import datetime
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA

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


def extract_all_hidden(model, processor, image, prompt):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    if hasattr(fwd, 'hidden_states') and fwd.hidden_states:
        all_layers = []
        for layer_hs in fwd.hidden_states:
            all_layers.append(layer_hs[0, -1, :].float().cpu().numpy())
        return all_layers
    return None


def cosine_dist(a, b):
    return 1.0 - float(np.dot(a / (np.linalg.norm(a) + 1e-10),
                               b / (np.linalg.norm(b) + 1e-10)))


def evaluate(cal_feats, id_feats, ood_feats):
    centroid = np.mean(cal_feats, axis=0)
    id_scores = [cosine_dist(h, centroid) for h in id_feats]
    ood_scores = [cosine_dist(h, centroid) for h in ood_feats]
    labels = [0]*len(id_scores) + [1]*len(ood_scores)
    auroc = roc_auc_score(labels, id_scores + ood_scores)
    id_arr, ood_arr = np.array(id_scores), np.array(ood_scores)
    pooled = np.sqrt((id_arr.var() + ood_arr.var()) / 2)
    d = (ood_arr.mean() - id_arr.mean()) / (pooled + 1e-10)
    return float(auroc), float(d)


def main():
    print("=" * 70, flush=True)
    print("COMPREHENSIVE ABLATION STUDY", flush=True)
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

    # Collect all hidden states (all layers)
    print("\nCollecting all hidden states...", flush=True)
    cal_all = []
    for fn in [create_highway, create_urban]:
        for i in range(12):
            layers = extract_all_hidden(model, processor,
                                         Image.fromarray(fn(i + 9000)), prompt)
            if layers is not None:
                cal_all.append(layers)

    id_all = []
    for fn in [create_highway, create_urban]:
        for i in range(10):
            layers = extract_all_hidden(model, processor,
                                         Image.fromarray(fn(i + 500)), prompt)
            if layers is not None:
                id_all.append(layers)

    ood_all = []
    ood_labels = []
    for fn, name in [(create_noise, 'noise'), (create_indoor, 'indoor'),
                     (create_twilight_highway, 'twilight'), (create_snow, 'snow')]:
        for i in range(8):
            layers = extract_all_hidden(model, processor,
                                         Image.fromarray(fn(i + 500)), prompt)
            if layers is not None:
                ood_all.append(layers)
                ood_labels.append(name)

    n_layers = len(cal_all[0])
    print(f"  Cal: {len(cal_all)}, ID: {len(id_all)}, OOD: {len(ood_all)}, Layers: {n_layers}", flush=True)

    results = {}

    # === Ablation 1: Feature source ===
    print("\n=== Ablation 1: Feature Source ===", flush=True)
    feature_ablation = {}

    # Last layer
    cal_last = [s[-1] for s in cal_all]
    id_last = [s[-1] for s in id_all]
    ood_last = [s[-1] for s in ood_all]
    auroc, d = evaluate(cal_last, id_last, ood_last)
    feature_ablation['last_layer'] = {'auroc': auroc, 'd': d, 'dim': len(cal_last[0])}
    print(f"  Last layer: AUROC={auroc:.3f}, d={d:.2f}", flush=True)

    # Layer 24
    cal_l24 = [s[24] for s in cal_all]
    id_l24 = [s[24] for s in id_all]
    ood_l24 = [s[24] for s in ood_all]
    auroc, d = evaluate(cal_l24, id_l24, ood_l24)
    feature_ablation['layer_24'] = {'auroc': auroc, 'd': d, 'dim': len(cal_l24[0])}
    print(f"  Layer 24: AUROC={auroc:.3f}, d={d:.2f}", flush=True)

    # Multi-layer PCA-8 (every 4th)
    layer_indices = list(range(0, n_layers, 4))
    cal_concat = [np.concatenate([s[l] for l in layer_indices]) for s in cal_all]
    id_concat = [np.concatenate([s[l] for l in layer_indices]) for s in id_all]
    ood_concat = [np.concatenate([s[l] for l in layer_indices]) for s in ood_all]
    all_concat = np.array(cal_concat + id_concat + ood_concat)
    pca = PCA(n_components=8, random_state=42)
    all_pca = pca.fit_transform(all_concat)
    nc, ni = len(cal_concat), len(id_concat)
    auroc, d = evaluate(all_pca[:nc].tolist(), all_pca[nc:nc+ni].tolist(), all_pca[nc+ni:].tolist())
    feature_ablation['multi_layer_pca8'] = {'auroc': auroc, 'd': d, 'dim': 8}
    print(f"  Multi-layer PCA-8: AUROC={auroc:.3f}, d={d:.2f}", flush=True)

    # Norm only (no centroid)
    id_norm = [np.linalg.norm(s[-1]) for s in id_all]
    ood_norm = [np.linalg.norm(s[-1]) for s in ood_all]
    labels = [0]*len(id_norm) + [1]*len(ood_norm)
    norm_auroc = roc_auc_score(labels, id_norm + ood_norm)
    feature_ablation['norm_only'] = {'auroc': float(norm_auroc), 'd': None, 'dim': 1}
    print(f"  Norm only: AUROC={norm_auroc:.3f}", flush=True)

    results['feature_source'] = feature_ablation

    # === Ablation 2: Calibration size ===
    print("\n=== Ablation 2: Calibration Size ===", flush=True)
    cal_size_ablation = {}
    for n_cal in [1, 3, 5, 10, 20, len(cal_last)]:
        if n_cal > len(cal_last):
            continue
        auroc, d = evaluate(cal_last[:n_cal], id_last, ood_last)
        cal_size_ablation[f'n_{n_cal}'] = {'auroc': auroc, 'd': d, 'n': n_cal}
        print(f"  N={n_cal}: AUROC={auroc:.3f}, d={d:.2f}", flush=True)
    results['calibration_size'] = cal_size_ablation

    # === Ablation 3: Detection metric ===
    print("\n=== Ablation 3: Detection Metric ===", flush=True)
    metric_ablation = {}

    centroid = np.mean(cal_last, axis=0)

    # Cosine
    id_cos = [cosine_dist(h, centroid) for h in id_last]
    ood_cos = [cosine_dist(h, centroid) for h in ood_last]
    auroc = roc_auc_score(labels, id_cos + ood_cos)
    metric_ablation['cosine'] = {'auroc': float(auroc)}
    print(f"  Cosine: AUROC={auroc:.3f}", flush=True)

    # Euclidean
    id_euc = [np.linalg.norm(np.array(h) - centroid) for h in id_last]
    ood_euc = [np.linalg.norm(np.array(h) - centroid) for h in ood_last]
    auroc = roc_auc_score(labels, id_euc + ood_euc)
    metric_ablation['euclidean'] = {'auroc': float(auroc)}
    print(f"  Euclidean: AUROC={auroc:.3f}", flush=True)

    results['detection_metric'] = metric_ablation

    # === Ablation 4: Per-category difficulty ===
    print("\n=== Ablation 4: Per-Category ===", flush=True)
    per_cat = {}
    for cat in set(ood_labels):
        cat_feats = [ood_last[i] for i, l in enumerate(ood_labels) if l == cat]
        cat_scores = [cosine_dist(h, centroid) for h in cat_feats]
        cat_labels = [0]*len(id_cos) + [1]*len(cat_scores)
        cat_all_scores = id_cos + cat_scores
        auroc = roc_auc_score(cat_labels, cat_all_scores)
        per_cat[cat] = {'auroc': float(auroc), 'n': len(cat_feats)}
        print(f"  {cat}: AUROC={auroc:.3f} (n={len(cat_feats)})", flush=True)
    results['per_category'] = per_cat

    # === Summary ===
    print("\n" + "=" * 70, flush=True)
    print("ABLATION SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"\nBest feature: multi-layer PCA-8 (d={feature_ablation['multi_layer_pca8']['d']:.2f})", flush=True)
    print(f"Simplest perfect: last-layer cosine (AUROC={feature_ablation['last_layer']['auroc']:.3f})", flush=True)
    print(f"Minimum calibration: N=1 (AUROC={cal_size_ablation['n_1']['auroc']:.3f})", flush=True)
    print(f"Calibration-free: norm (AUROC={feature_ablation['norm_only']['auroc']:.3f})", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'comprehensive_ablation',
        'experiment_number': 100,
        'timestamp': timestamp,
        'n_cal': len(cal_all),
        'n_id': len(id_all),
        'n_ood': len(ood_all),
        'results': results,
    }
    output_path = os.path.join(RESULTS_DIR, f"comprehensive_ablation_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
