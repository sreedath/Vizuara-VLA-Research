"""
Multi-Layer Hidden State Fusion for OOD Detection.

Tests whether combining hidden states from multiple layers improves
detection over using only the last layer. Compares: last-layer only,
early+late concat, every-4th-layer concat, weighted average, PCA of
concatenated multi-layer features.

Experiment 92 in the CalibDrive series.
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
    """Extract hidden states from ALL layers."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    if hasattr(fwd, 'hidden_states') and fwd.hidden_states:
        # Return last-token hidden state from each layer
        all_layers = []
        for layer_hs in fwd.hidden_states:
            all_layers.append(layer_hs[0, -1, :].float().cpu().numpy())
        return all_layers
    return None


def cosine_dist(a, b):
    return 1.0 - float(np.dot(a / (np.linalg.norm(a) + 1e-10),
                               b / (np.linalg.norm(b) + 1e-10)))


def evaluate_features(cal_features, id_features, ood_features, ood_labels):
    """Evaluate a set of features using centroid-based cosine detection."""
    centroid = np.mean(cal_features, axis=0)

    id_scores = [cosine_dist(h, centroid) for h in id_features]
    ood_scores = [cosine_dist(h, centroid) for h in ood_features]

    labels = [0]*len(id_scores) + [1]*len(ood_scores)
    scores = id_scores + ood_scores
    auroc = roc_auc_score(labels, scores)

    id_arr = np.array(id_scores)
    ood_arr = np.array(ood_scores)
    pooled = np.sqrt((id_arr.var() + ood_arr.var()) / 2)
    d = (ood_arr.mean() - id_arr.mean()) / (pooled + 1e-10)

    # Per-category
    cat_aurocs = {}
    for cat in set(ood_labels):
        cat_scores = [cosine_dist(h, centroid)
                      for h, l in zip(ood_features, ood_labels) if l == cat]
        cat_labels = [0]*len(id_scores) + [1]*len(cat_scores)
        cat_all = id_scores + cat_scores
        cat_aurocs[cat] = float(roc_auc_score(cat_labels, cat_all))

    return {
        'auroc': float(auroc),
        'cohens_d': float(d),
        'id_mean': float(id_arr.mean()),
        'ood_mean': float(ood_arr.mean()),
        'per_category': cat_aurocs,
    }


def main():
    print("=" * 70, flush=True)
    print("MULTI-LAYER HIDDEN STATE FUSION", flush=True)
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

    # Collect all hidden states
    print("\nCollecting calibration hidden states...", flush=True)
    cal_all_layers = []
    for fn in [create_highway, create_urban]:
        for i in range(10):
            layers = extract_all_hidden(model, processor,
                                         Image.fromarray(fn(i + 9000)), prompt)
            if layers is not None:
                cal_all_layers.append(layers)
    print(f"  Calibration: {len(cal_all_layers)} samples, {len(cal_all_layers[0])} layers", flush=True)

    print("\nCollecting ID test hidden states...", flush=True)
    id_all_layers = []
    for fn in [create_highway, create_urban]:
        for i in range(8):
            layers = extract_all_hidden(model, processor,
                                         Image.fromarray(fn(i + 500)), prompt)
            if layers is not None:
                id_all_layers.append(layers)

    print("\nCollecting OOD test hidden states...", flush=True)
    ood_all_layers = []
    ood_labels = []
    for fn, name in [(create_noise, 'noise'), (create_indoor, 'indoor'),
                     (create_twilight_highway, 'twilight'), (create_snow, 'snow')]:
        for i in range(6):
            layers = extract_all_hidden(model, processor,
                                         Image.fromarray(fn(i + 500)), prompt)
            if layers is not None:
                ood_all_layers.append(layers)
                ood_labels.append(name)

    print(f"  ID: {len(id_all_layers)}, OOD: {len(ood_all_layers)}", flush=True)
    n_layers = len(cal_all_layers[0])
    print(f"  Layers per sample: {n_layers}", flush=True)

    # Define fusion strategies
    strategies = {}

    # 1. Last layer only (baseline)
    strategies['last_layer'] = {'layers': [n_layers - 1]}

    # 2. Layer 24 only (peak from Exp 72)
    strategies['layer_24'] = {'layers': [24]}

    # 3. Early + Late (layers 0, 8, 16, 24, 31)
    strategies['early_late'] = {'layers': [0, 8, 16, 24, n_layers - 1]}

    # 4. Every 4th layer
    strategies['every_4th'] = {'layers': list(range(0, n_layers, 4))}

    # 5. Last 4 layers
    strategies['last_4'] = {'layers': list(range(n_layers - 4, n_layers))}

    # 6. Last 8 layers
    strategies['last_8'] = {'layers': list(range(n_layers - 8, n_layers))}

    results = {}
    for name, config in strategies.items():
        layer_indices = config['layers']
        dim_per_layer = cal_all_layers[0][0].shape[0]

        # Concatenate selected layers
        cal_feats = []
        for sample in cal_all_layers:
            feat = np.concatenate([sample[l] for l in layer_indices])
            cal_feats.append(feat)

        id_feats = []
        for sample in id_all_layers:
            feat = np.concatenate([sample[l] for l in layer_indices])
            id_feats.append(feat)

        ood_feats = []
        for sample in ood_all_layers:
            feat = np.concatenate([sample[l] for l in layer_indices])
            ood_feats.append(feat)

        res = evaluate_features(cal_feats, id_feats, ood_feats, ood_labels)
        res['layers'] = layer_indices
        res['total_dim'] = len(cal_feats[0])
        results[name] = res
        print(f"  {name:<15}: AUROC={res['auroc']:.3f}, d={res['cohens_d']:.2f}, dim={res['total_dim']}", flush=True)

    # 7. PCA reduction of every-4th concatenated features
    print("\n  Testing PCA reductions of every-4th features...", flush=True)
    layer_indices = list(range(0, n_layers, 4))
    cal_concat = [np.concatenate([s[l] for l in layer_indices]) for s in cal_all_layers]
    id_concat = [np.concatenate([s[l] for l in layer_indices]) for s in id_all_layers]
    ood_concat = [np.concatenate([s[l] for l in layer_indices]) for s in ood_all_layers]

    all_concat = np.array(cal_concat + id_concat + ood_concat)
    n_cal = len(cal_concat)
    n_id = len(id_concat)

    pca_results = {}
    for n_comp in [4, 8, 16, 32]:
        if n_comp > min(all_concat.shape[0], all_concat.shape[1]):
            continue
        pca = PCA(n_components=n_comp, random_state=42)
        all_pca = pca.fit_transform(all_concat)
        cal_pca = all_pca[:n_cal]
        id_pca = all_pca[n_cal:n_cal+n_id]
        ood_pca = all_pca[n_cal+n_id:]

        res = evaluate_features(cal_pca.tolist(), id_pca.tolist(), ood_pca.tolist(), ood_labels)
        res['n_components'] = n_comp
        res['explained_var'] = float(sum(pca.explained_variance_ratio_))
        pca_results[f'pca_{n_comp}'] = res
        print(f"    PCA-{n_comp}: AUROC={res['auroc']:.3f}, d={res['cohens_d']:.2f}, var={res['explained_var']:.3f}", flush=True)

    # 8. Weighted average (weight by layer number — later layers get more weight)
    print("\n  Testing weighted average...", flush=True)
    weights = np.linspace(0.1, 1.0, n_layers)
    weights = weights / weights.sum()

    cal_wavg = []
    for sample in cal_all_layers:
        feat = sum(w * sample[l] for l, w in enumerate(weights))
        cal_wavg.append(feat)

    id_wavg = []
    for sample in id_all_layers:
        feat = sum(w * sample[l] for l, w in enumerate(weights))
        id_wavg.append(feat)

    ood_wavg = []
    for sample in ood_all_layers:
        feat = sum(w * sample[l] for l, w in enumerate(weights))
        ood_wavg.append(feat)

    res = evaluate_features(cal_wavg, id_wavg, ood_wavg, ood_labels)
    res['method'] = 'weighted_avg_linear'
    results['weighted_avg'] = res
    print(f"  weighted_avg   : AUROC={res['auroc']:.3f}, d={res['cohens_d']:.2f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'multi_layer_fusion',
        'experiment_number': 92,
        'timestamp': timestamp,
        'n_cal': len(cal_all_layers),
        'n_id': len(id_all_layers),
        'n_ood': len(ood_all_layers),
        'n_layers': n_layers,
        'strategies': results,
        'pca_results': pca_results,
    }
    output_path = os.path.join(RESULTS_DIR, f"multi_layer_fusion_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
