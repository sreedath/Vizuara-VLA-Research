"""
Calibration Curve: Fine-Grained Calibration Set Size Analysis.

Tests detection performance at every calibration size from 1 to 50,
measuring AUROC, Cohen's d, centroid stability, and false positive rates
at each size. This provides the definitive calibration curve for the
cosine distance detector.

Experiment 113 in the CalibDrive series.
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
    rng = np.random.default_rng(idx * 7001)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]
    img[SIZE[0]//2:] = [80, 80, 80]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [255, 255, 255]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_urban(idx):
    rng = np.random.default_rng(idx * 7002)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [135, 206, 235]
    img[SIZE[0]//3:SIZE[0]//2] = [139, 119, 101]
    img[SIZE[0]//2:] = [60, 60, 60]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_noise(idx):
    rng = np.random.default_rng(idx * 7003)
    return rng.integers(0, 256, (*SIZE, 3), dtype=np.uint8)

def create_indoor(idx):
    rng = np.random.default_rng(idx * 7004)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:] = [200, 180, 160]
    img[SIZE[0]//2:, :] = [100, 80, 60]
    img[:SIZE[0]//3, SIZE[1]//3:2*SIZE[1]//3] = [150, 200, 255]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_twilight_highway(idx):
    rng = np.random.default_rng(idx * 7010)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [70, 50, 80]
    img[SIZE[0]//2:] = [60, 60, 60]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [200, 200, 100]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_snow(idx):
    rng = np.random.default_rng(idx * 7014)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [200, 200, 210]
    img[SIZE[0]//2:] = [220, 220, 230]
    img[SIZE[0]//2:, SIZE[1]//2-2:SIZE[1]//2+2] = [180, 180, 190]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def extract_hidden(model, processor, image, prompt):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    if not hasattr(fwd, 'hidden_states') or not fwd.hidden_states:
        return None
    return fwd.hidden_states[-1][0, -1, :].float().cpu().numpy()


def cosine_dist(a, b):
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def main():
    print("=" * 70, flush=True)
    print("CALIBRATION CURVE — FINE-GRAINED SIZE ANALYSIS", flush=True)
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

    # Collect a large pool of ID embeddings (50 highway + 50 urban = 100 total)
    print("\n--- Collecting ID pool (100 samples) ---", flush=True)
    id_pool = []
    id_sources = []
    for i in range(50):
        h = extract_hidden(model, processor, Image.fromarray(create_highway(i + 1400)), prompt)
        if h is not None:
            id_pool.append(h)
            id_sources.append('highway')
        if (i + 1) % 10 == 0:
            print(f"  Highway {i+1}/50", flush=True)
    for i in range(50):
        h = extract_hidden(model, processor, Image.fromarray(create_urban(i + 1400)), prompt)
        if h is not None:
            id_pool.append(h)
            id_sources.append('urban')
        if (i + 1) % 10 == 0:
            print(f"  Urban {i+1}/50", flush=True)

    id_pool = np.array(id_pool)
    print(f"  ID pool: {len(id_pool)} samples", flush=True)

    # Collect OOD test set (15 per category = 60)
    print("\n--- Collecting OOD test set ---", flush=True)
    ood_embeds = {}
    for cat_name, fn in [('noise', create_noise), ('indoor', create_indoor),
                          ('twilight', create_twilight_highway), ('snow', create_snow)]:
        embeds = []
        for i in range(15):
            h = extract_hidden(model, processor, Image.fromarray(fn(i + 1400)), prompt)
            if h is not None:
                embeds.append(h)
        ood_embeds[cat_name] = np.array(embeds)
        print(f"  {cat_name}: {len(embeds)} samples", flush=True)

    all_ood = np.concatenate(list(ood_embeds.values()), axis=0)

    # Test every calibration size from 1 to 50
    print("\n--- Calibration curve ---", flush=True)
    # Use first N from pool as calibration, rest as ID test
    # Fixed test set: last 20 ID + all 60 OOD

    id_test = id_pool[-20:]  # fixed ID test set
    cal_pool = id_pool[:-20]  # up to 80 for calibration

    test_embeds = np.concatenate([id_test, all_ood], axis=0)
    test_labels = np.array([0] * len(id_test) + [1] * len(all_ood))

    results_by_size = {}
    full_centroid = np.mean(cal_pool, axis=0)

    for n_cal in range(1, min(51, len(cal_pool) + 1)):
        cal = cal_pool[:n_cal]
        centroid = np.mean(cal, axis=0)

        # Centroid stability
        centroid_sim = float(np.dot(centroid, full_centroid) /
                           (np.linalg.norm(centroid) * np.linalg.norm(full_centroid) + 1e-10))

        # Compute scores
        scores = np.array([cosine_dist(e, centroid) for e in test_embeds])
        id_scores = scores[test_labels == 0]
        ood_scores = scores[test_labels == 1]

        auroc = float(roc_auc_score(test_labels, scores))
        d = float((np.mean(ood_scores) - np.mean(id_scores)) / (np.std(id_scores) + 1e-10))

        # False positive rate at various thresholds
        fpr_at_95tpr = None
        sorted_ood = np.sort(ood_scores)
        thresh_95 = sorted_ood[max(0, int(0.05 * len(sorted_ood)))]
        fpr_at_95tpr = float(np.mean(id_scores >= thresh_95))

        results_by_size[str(n_cal)] = {
            'n_cal': n_cal,
            'auroc': auroc,
            'd': d,
            'centroid_cosine_sim': centroid_sim,
            'id_score_mean': float(np.mean(id_scores)),
            'id_score_std': float(np.std(id_scores)),
            'ood_score_mean': float(np.mean(ood_scores)),
            'ood_score_std': float(np.std(ood_scores)),
            'fpr_at_95tpr': fpr_at_95tpr,
        }

        if n_cal <= 10 or n_cal % 5 == 0:
            print(f"  n={n_cal}: AUROC={auroc:.4f}, d={d:.2f}, "
                  f"centroid_sim={centroid_sim:.6f}, FPR@95={fpr_at_95tpr:.4f}", flush=True)

    # Per-OOD-category breakdown at key sizes
    print("\n--- Per-category at key sizes ---", flush=True)
    per_cat_by_size = {}
    for n_cal in [1, 2, 3, 5, 10, 20, 50]:
        if n_cal > len(cal_pool):
            continue
        cal = cal_pool[:n_cal]
        centroid = np.mean(cal, axis=0)
        cat_results = {}
        for cat_name, cat_embeds in ood_embeds.items():
            cat_scores = [cosine_dist(e, centroid) for e in cat_embeds]
            cat_results[cat_name] = {
                'mean_score': float(np.mean(cat_scores)),
                'std_score': float(np.std(cat_scores)),
            }
        per_cat_by_size[str(n_cal)] = cat_results
        print(f"  n={n_cal}: " + ", ".join(f"{c}={r['mean_score']:.3f}" for c, r in cat_results.items()), flush=True)

    # Bootstrap confidence intervals at key sizes
    print("\n--- Bootstrap CIs ---", flush=True)
    bootstrap_results = {}
    for n_cal in [1, 3, 5, 10, 20]:
        if n_cal > len(cal_pool):
            continue
        aurocs_boot = []
        for boot in range(20):
            rng = np.random.default_rng(boot * 1000 + n_cal)
            indices = rng.choice(len(cal_pool), size=n_cal, replace=True)
            cal = cal_pool[indices]
            centroid = np.mean(cal, axis=0)
            scores = np.array([cosine_dist(e, centroid) for e in test_embeds])
            aurocs_boot.append(float(roc_auc_score(test_labels, scores)))
        bootstrap_results[str(n_cal)] = {
            'auroc_mean': float(np.mean(aurocs_boot)),
            'auroc_std': float(np.std(aurocs_boot)),
            'auroc_min': float(np.min(aurocs_boot)),
            'auroc_max': float(np.max(aurocs_boot)),
        }
        print(f"  n={n_cal}: AUROC={np.mean(aurocs_boot):.4f}+/-{np.std(aurocs_boot):.4f} "
              f"[{np.min(aurocs_boot):.4f}, {np.max(aurocs_boot):.4f}]", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'calibration_curve',
        'experiment_number': 113,
        'timestamp': timestamp,
        'id_pool_size': len(id_pool),
        'ood_test_size': len(all_ood),
        'id_test_size': len(id_test),
        'calibration_curve': results_by_size,
        'per_category_by_size': per_cat_by_size,
        'bootstrap': bootstrap_results,
    }
    output_path = os.path.join(RESULTS_DIR, f"calibration_curve_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
