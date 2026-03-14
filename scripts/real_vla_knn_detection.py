"""
KNN-Based OOD Detection.

Tests k-nearest-neighbor distance as an alternative to centroid-based detection:
instead of comparing to the centroid, compare to the k closest calibration
embeddings. KNN captures non-convex ID manifold shapes that a single centroid
cannot.

Experiment 103 in the CalibDrive series.
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


def extract_hidden(model, processor, image, prompt):
    """Extract last hidden state."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    if not hasattr(fwd, 'hidden_states') or not fwd.hidden_states:
        return None
    return fwd.hidden_states[-1][0, -1, :].float().cpu().numpy()


def knn_score(test_embed, cal_embeds, k, metric='cosine'):
    """Compute k-nearest neighbor score."""
    if metric == 'cosine':
        # Cosine distance = 1 - cosine_similarity
        test_norm = test_embed / (np.linalg.norm(test_embed) + 1e-10)
        cal_norms = cal_embeds / (np.linalg.norm(cal_embeds, axis=1, keepdims=True) + 1e-10)
        sims = cal_norms @ test_norm
        dists = 1 - sims
    elif metric == 'euclidean':
        dists = np.linalg.norm(cal_embeds - test_embed, axis=1)
    elif metric == 'manhattan':
        dists = np.sum(np.abs(cal_embeds - test_embed), axis=1)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    sorted_dists = np.sort(dists)
    return float(np.mean(sorted_dists[:k]))


def main():
    print("=" * 70, flush=True)
    print("KNN-BASED OOD DETECTION", flush=True)
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

    # Phase 1: Collect all embeddings
    print("\n--- Collecting embeddings ---", flush=True)
    embeddings = {}
    for cat_name, (fn, group) in categories.items():
        print(f"  {cat_name} ({group})...", flush=True)
        embeds = []
        for i in range(15):
            h = extract_hidden(model, processor, Image.fromarray(fn(i + 600)), prompt)
            if h is not None:
                embeds.append(h)
        embeddings[cat_name] = {'embeds': np.array(embeds), 'group': group}
        print(f"    Collected {len(embeds)} embeddings", flush=True)

    # Phase 2: Build calibration set (first 10 from each ID category)
    cal_embeds = []
    for cat_name, data in embeddings.items():
        if data['group'] == 'ID':
            cal_embeds.append(data['embeds'][:10])
    cal_embeds = np.concatenate(cal_embeds, axis=0)
    print(f"\nCalibration set: {cal_embeds.shape[0]} embeddings", flush=True)

    # Test set: remaining from ID + all OOD
    test_embeds = []
    test_labels = []
    test_cats = []
    for cat_name, data in embeddings.items():
        if data['group'] == 'ID':
            for e in data['embeds'][10:]:
                test_embeds.append(e)
                test_labels.append(0)
                test_cats.append(cat_name)
        else:
            for e in data['embeds']:
                test_embeds.append(e)
                test_labels.append(1)
                test_cats.append(cat_name)

    test_embeds = np.array(test_embeds)
    test_labels = np.array(test_labels)

    print(f"Test set: {len(test_embeds)} embeddings ({sum(test_labels==0)} ID, {sum(test_labels==1)} OOD)", flush=True)

    # Phase 3: Test different k values and metrics
    print("\n--- KNN Detection ---", flush=True)
    k_values = [1, 3, 5, 10, 15, 20]
    metrics = ['cosine', 'euclidean', 'manhattan']

    knn_results = {}
    for metric in metrics:
        for k in k_values:
            if k > cal_embeds.shape[0]:
                continue
            scores = [knn_score(e, cal_embeds, k, metric) for e in test_embeds]
            auroc = roc_auc_score(test_labels, scores)
            key = f"{metric}_k{k}"
            knn_results[key] = {
                'metric': metric,
                'k': k,
                'auroc': float(auroc),
                'id_mean': float(np.mean([s for s, l in zip(scores, test_labels) if l == 0])),
                'ood_mean': float(np.mean([s for s, l in zip(scores, test_labels) if l == 0])),
                'id_std': float(np.std([s for s, l in zip(scores, test_labels) if l == 0])),
                'ood_std': float(np.std([s for s, l in zip(scores, test_labels) if l == 1])),
            }
            # Fix ood_mean
            knn_results[key]['ood_mean'] = float(np.mean([s for s, l in zip(scores, test_labels) if l == 1]))
            print(f"  {key}: AUROC={auroc:.4f}", flush=True)

    # Phase 4: Compare KNN vs centroid
    print("\n--- KNN vs Centroid Comparison ---", flush=True)
    centroid = np.mean(cal_embeds, axis=0)
    centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-10)

    centroid_scores = []
    for e in test_embeds:
        e_norm = e / (np.linalg.norm(e) + 1e-10)
        centroid_scores.append(float(1 - np.dot(e_norm, centroid_norm)))

    centroid_auroc = roc_auc_score(test_labels, centroid_scores)
    print(f"  Centroid cosine: AUROC={centroid_auroc:.4f}", flush=True)

    # Best KNN
    best_knn_key = max(knn_results, key=lambda k: knn_results[k]['auroc'])
    best_knn = knn_results[best_knn_key]
    print(f"  Best KNN ({best_knn_key}): AUROC={best_knn['auroc']:.4f}", flush=True)

    # Phase 5: Per-category KNN analysis with best k
    print("\n--- Per-Category Analysis (best KNN) ---", flush=True)
    best_metric = best_knn['metric']
    best_k = best_knn['k']

    per_cat_results = {}
    for cat_name in categories:
        cat_scores = [knn_score(e, cal_embeds, best_k, best_metric)
                      for e, c in zip(test_embeds, test_cats) if c == cat_name]
        per_cat_results[cat_name] = {
            'mean': float(np.mean(cat_scores)),
            'std': float(np.std(cat_scores)),
            'scores': [float(s) for s in cat_scores],
        }
        group = categories[cat_name][1]
        print(f"  {cat_name} ({group}): {np.mean(cat_scores):.4f}±{np.std(cat_scores):.4f}", flush=True)

    # Phase 6: Hybrid KNN+centroid ensemble
    print("\n--- KNN+Centroid Ensemble ---", flush=True)
    best_knn_scores = [knn_score(e, cal_embeds, best_k, best_metric) for e in test_embeds]

    # Normalize both score sets to [0, 1]
    knn_arr = np.array(best_knn_scores)
    cent_arr = np.array(centroid_scores)
    knn_norm = (knn_arr - knn_arr.min()) / (knn_arr.max() - knn_arr.min() + 1e-10)
    cent_norm = (cent_arr - cent_arr.min()) / (cent_arr.max() - cent_arr.min() + 1e-10)

    ensemble_results = {}
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        combined = alpha * knn_norm + (1 - alpha) * cent_norm
        auroc = roc_auc_score(test_labels, combined)
        ensemble_results[f"alpha_{alpha:.1f}"] = {
            'alpha_knn': float(alpha),
            'auroc': float(auroc),
        }
        print(f"  α_knn={alpha:.1f}: AUROC={auroc:.4f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'knn_detection',
        'experiment_number': 103,
        'timestamp': timestamp,
        'n_calibration': int(cal_embeds.shape[0]),
        'n_test_id': int(sum(test_labels == 0)),
        'n_test_ood': int(sum(test_labels == 1)),
        'knn_results': knn_results,
        'centroid_auroc': float(centroid_auroc),
        'best_knn': best_knn_key,
        'per_category': per_cat_results,
        'ensemble_results': ensemble_results,
    }
    output_path = os.path.join(RESULTS_DIR, f"knn_detection_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
