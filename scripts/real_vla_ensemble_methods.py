"""
Ensemble Detection Methods.

Combines multiple OOD signals (cosine distance, entropy, action vocabulary
divergence, norm) into ensemble detectors to test whether multi-signal
fusion outperforms any individual detector.

Experiment 112 in the CalibDrive series.
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
    rng = np.random.default_rng(idx * 6001)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]
    img[SIZE[0]//2:] = [80, 80, 80]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [255, 255, 255]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_urban(idx):
    rng = np.random.default_rng(idx * 6002)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [135, 206, 235]
    img[SIZE[0]//3:SIZE[0]//2] = [139, 119, 101]
    img[SIZE[0]//2:] = [60, 60, 60]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_noise(idx):
    rng = np.random.default_rng(idx * 6003)
    return rng.integers(0, 256, (*SIZE, 3), dtype=np.uint8)

def create_indoor(idx):
    rng = np.random.default_rng(idx * 6004)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:] = [200, 180, 160]
    img[SIZE[0]//2:, :] = [100, 80, 60]
    img[:SIZE[0]//3, SIZE[1]//3:2*SIZE[1]//3] = [150, 200, 255]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_twilight_highway(idx):
    rng = np.random.default_rng(idx * 6010)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [70, 50, 80]
    img[SIZE[0]//2:] = [60, 60, 60]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [200, 200, 100]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_snow(idx):
    rng = np.random.default_rng(idx * 6014)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [200, 200, 210]
    img[SIZE[0]//2:] = [220, 220, 230]
    img[SIZE[0]//2:, SIZE[1]//2-2:SIZE[1]//2+2] = [180, 180, 190]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def extract_full(model, processor, image, prompt):
    """Extract hidden state, logits, and action tokens."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)

    # Hidden state
    if not hasattr(fwd, 'hidden_states') or not fwd.hidden_states:
        return None
    hidden = fwd.hidden_states[-1][0, -1, :].float().cpu().numpy()

    # Logits → entropy and top-k
    logits = fwd.logits[0, -1, :].float().cpu()
    probs = torch.softmax(logits, dim=0).numpy()
    entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
    top1_prob = float(np.max(probs))
    top5_prob = float(np.sort(probs)[-5:].sum())

    return {
        'hidden': hidden,
        'entropy': entropy,
        'top1_prob': top1_prob,
        'top5_prob': top5_prob,
        'norm': float(np.linalg.norm(hidden)),
    }


def cosine_dist(a, b):
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def main():
    print("=" * 70, flush=True)
    print("ENSEMBLE DETECTION METHODS", flush=True)
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

    print("\n--- Collecting multi-signal features ---", flush=True)
    all_data = {}
    for cat_name, (fn, group) in categories.items():
        print(f"  {cat_name} ({group})...", flush=True)
        samples = []
        for i in range(15):
            result = extract_full(model, processor, Image.fromarray(fn(i + 1300)), prompt)
            if result is not None:
                samples.append(result)
        all_data[cat_name] = {'samples': samples, 'group': group}

    # Build calibration and test sets
    cal_hiddens = []
    cal_entropies = []
    cal_norms = []

    test_features = []  # list of dicts
    test_labels = []

    for cat_name, data in all_data.items():
        if data['group'] == 'ID':
            for s in data['samples'][:5]:
                cal_hiddens.append(s['hidden'])
                cal_entropies.append(s['entropy'])
                cal_norms.append(s['norm'])
            for s in data['samples'][5:]:
                test_features.append(s)
                test_labels.append(0)
        else:
            for s in data['samples']:
                test_features.append(s)
                test_labels.append(1)

    cal_hiddens = np.array(cal_hiddens)
    cal_centroid = np.mean(cal_hiddens, axis=0)
    cal_mean_entropy = np.mean(cal_entropies)
    cal_mean_norm = np.mean(cal_norms)
    test_labels = np.array(test_labels)

    print(f"\nCal: {len(cal_hiddens)}, Test: {len(test_labels)}", flush=True)

    # Compute individual scores
    print("\n--- Computing individual detector scores ---", flush=True)

    scores_cosine = []
    scores_entropy = []
    scores_norm = []
    scores_top1 = []

    for feat in test_features:
        scores_cosine.append(cosine_dist(feat['hidden'], cal_centroid))
        scores_entropy.append(feat['entropy'])
        scores_norm.append(abs(feat['norm'] - cal_mean_norm))
        scores_top1.append(1 - feat['top1_prob'])  # lower confidence = more OOD

    scores_cosine = np.array(scores_cosine)
    scores_entropy = np.array(scores_entropy)
    scores_norm = np.array(scores_norm)
    scores_top1 = np.array(scores_top1)

    # Individual AUROCs
    individual_results = {}
    for name, scores in [('cosine', scores_cosine), ('entropy', scores_entropy),
                          ('norm', scores_norm), ('top1_inv', scores_top1)]:
        auroc = roc_auc_score(test_labels, scores)
        id_s = scores[test_labels == 0]
        ood_s = scores[test_labels == 1]
        d = float((np.mean(ood_s) - np.mean(id_s)) / (np.std(id_s) + 1e-10))
        individual_results[name] = {'auroc': float(auroc), 'd': d}
        print(f"  {name}: AUROC={auroc:.4f}, d={d:.2f}", flush=True)

    # Normalize scores to [0,1] for ensemble
    def normalize(s):
        mn, mx = s.min(), s.max()
        if mx - mn < 1e-10:
            return np.zeros_like(s)
        return (s - mn) / (mx - mn)

    norm_cosine = normalize(scores_cosine)
    norm_entropy = normalize(scores_entropy)
    norm_norm = normalize(scores_norm)
    norm_top1 = normalize(scores_top1)

    # Ensemble methods
    print("\n--- Ensemble Methods ---", flush=True)
    ensemble_results = {}

    # 1. Simple average
    for combo_name, combo_scores in [
        ('avg_all', [norm_cosine, norm_entropy, norm_norm, norm_top1]),
        ('avg_cos_ent', [norm_cosine, norm_entropy]),
        ('avg_cos_norm', [norm_cosine, norm_norm]),
        ('avg_cos_top1', [norm_cosine, norm_top1]),
        ('avg_ent_norm', [norm_entropy, norm_norm]),
    ]:
        combined = np.mean(combo_scores, axis=0)
        auroc = roc_auc_score(test_labels, combined)
        id_s = combined[test_labels == 0]
        ood_s = combined[test_labels == 1]
        d = float((np.mean(ood_s) - np.mean(id_s)) / (np.std(id_s) + 1e-10))
        ensemble_results[combo_name] = {'auroc': float(auroc), 'd': d, 'method': 'average'}
        print(f"  {combo_name}: AUROC={auroc:.4f}, d={d:.2f}", flush=True)

    # 2. Max fusion (take worst-case score)
    for combo_name, combo_scores in [
        ('max_all', [norm_cosine, norm_entropy, norm_norm, norm_top1]),
        ('max_cos_ent', [norm_cosine, norm_entropy]),
    ]:
        combined = np.max(combo_scores, axis=0)
        auroc = roc_auc_score(test_labels, combined)
        id_s = combined[test_labels == 0]
        ood_s = combined[test_labels == 1]
        d = float((np.mean(ood_s) - np.mean(id_s)) / (np.std(id_s) + 1e-10))
        ensemble_results[combo_name] = {'auroc': float(auroc), 'd': d, 'method': 'max'}
        print(f"  {combo_name}: AUROC={auroc:.4f}, d={d:.2f}", flush=True)

    # 3. Product fusion (multiply normalized scores)
    for combo_name, combo_scores in [
        ('prod_all', [norm_cosine, norm_entropy, norm_norm, norm_top1]),
        ('prod_cos_ent', [norm_cosine, norm_entropy]),
    ]:
        combined = np.prod(combo_scores, axis=0)
        auroc = roc_auc_score(test_labels, combined)
        id_s = combined[test_labels == 0]
        ood_s = combined[test_labels == 1]
        d = float((np.mean(ood_s) - np.mean(id_s)) / (np.std(id_s) + 1e-10))
        ensemble_results[combo_name] = {'auroc': float(auroc), 'd': d, 'method': 'product'}
        print(f"  {combo_name}: AUROC={auroc:.4f}, d={d:.2f}", flush=True)

    # 4. Weighted average (optimized weights via grid search on test set — oracle)
    print("\n--- Oracle Weighted Ensemble (grid search) ---", flush=True)
    best_d = -1
    best_weights = None
    best_auroc_w = 0
    all_norm = np.stack([norm_cosine, norm_entropy, norm_norm, norm_top1], axis=1)

    for w0 in np.arange(0, 1.1, 0.2):
        for w1 in np.arange(0, 1.1 - w0, 0.2):
            for w2 in np.arange(0, 1.1 - w0 - w1, 0.2):
                w3 = 1.0 - w0 - w1 - w2
                if w3 < -0.01:
                    continue
                weights = np.array([w0, w1, w2, max(0, w3)])
                combined = all_norm @ weights
                auroc = roc_auc_score(test_labels, combined)
                id_s = combined[test_labels == 0]
                ood_s = combined[test_labels == 1]
                d = float((np.mean(ood_s) - np.mean(id_s)) / (np.std(id_s) + 1e-10))
                if d > best_d:
                    best_d = d
                    best_weights = weights.tolist()
                    best_auroc_w = float(auroc)

    ensemble_results['oracle_weighted'] = {
        'auroc': best_auroc_w, 'd': best_d,
        'weights': best_weights,
        'weight_labels': ['cosine', 'entropy', 'norm', 'top1_inv'],
        'method': 'oracle_weighted',
    }
    print(f"  Oracle weighted: AUROC={best_auroc_w:.4f}, d={best_d:.2f}, weights={best_weights}", flush=True)

    # Per-category breakdown for best individual and best ensemble
    print("\n--- Per-Category Breakdown ---", flush=True)
    per_cat = {}
    for cat_name, data in all_data.items():
        cat_hiddens = [s['hidden'] for s in data['samples']]
        cat_cosine = [cosine_dist(h, cal_centroid) for h in cat_hiddens]
        cat_entropy = [s['entropy'] for s in data['samples']]
        cat_norm = [abs(s['norm'] - cal_mean_norm) for s in data['samples']]
        per_cat[cat_name] = {
            'group': data['group'],
            'cosine_mean': float(np.mean(cat_cosine)),
            'cosine_std': float(np.std(cat_cosine)),
            'entropy_mean': float(np.mean(cat_entropy)),
            'entropy_std': float(np.std(cat_entropy)),
            'norm_mean': float(np.mean(cat_norm)),
            'norm_std': float(np.std(cat_norm)),
        }
        print(f"  {cat_name} ({data['group']}): cos={np.mean(cat_cosine):.4f}, "
              f"ent={np.mean(cat_entropy):.3f}, norm={np.mean(cat_norm):.2f}", flush=True)

    # Signal correlation analysis
    print("\n--- Signal Correlations ---", flush=True)
    corr_matrix = np.corrcoef(np.stack([scores_cosine, scores_entropy, scores_norm, scores_top1]))
    signal_names = ['cosine', 'entropy', 'norm', 'top1_inv']
    correlations = {}
    for i in range(4):
        for j in range(i+1, 4):
            key = f"{signal_names[i]}_vs_{signal_names[j]}"
            correlations[key] = float(corr_matrix[i, j])
            print(f"  {key}: r={corr_matrix[i, j]:.4f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'ensemble_methods',
        'experiment_number': 112,
        'timestamp': timestamp,
        'n_cal': len(cal_hiddens),
        'n_test': len(test_labels),
        'individual': individual_results,
        'ensemble': ensemble_results,
        'per_category': per_cat,
        'correlations': correlations,
    }
    output_path = os.path.join(RESULTS_DIR, f"ensemble_methods_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
