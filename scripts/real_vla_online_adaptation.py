#!/usr/bin/env python3
"""Experiment 451: Online Calibration Adaptation

Tests whether the OOD detector can adapt its calibration online as it
encounters more clean data. In a real deployment the robot initially
calibrates on a few scenes, then progressively improves its calibration
as it observes more data. This experiment asks:
  1. How does AUROC grow with calibration set size (streaming update)?
  2. Running mean vs. EMA centroid strategies: which converges faster?
  3. How does the 3-sigma threshold stabilise as more scenes arrive?
  4. Can EMA track gradual distribution shift better than running mean?
  5. What detection quality can we expect from a cold start (1 scene)?
"""

import torch, json, os, sys, numpy as np
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor
from datetime import datetime


# ---------------------------------------------------------------------------
# Image & embedding helpers
# ---------------------------------------------------------------------------

def make_image(seed=42):
    rng = np.random.RandomState(seed)
    return Image.fromarray(rng.randint(0, 255, (224, 224, 3), dtype=np.uint8))


def apply_corruption(image, ctype, severity=1.0):
    arr = np.array(image).astype(np.float32) / 255.0
    if ctype == 'fog':
        arr = arr * (1 - 0.6 * severity) + 0.6 * severity
    elif ctype == 'night':
        arr = arr * max(0.01, 1.0 - 0.95 * severity)
    elif ctype == 'noise':
        arr = arr + np.random.RandomState(42).randn(*arr.shape) * 0.3 * severity
        arr = np.clip(arr, 0, 1)
    elif ctype == 'blur':
        return image.filter(ImageFilter.GaussianBlur(radius=10 * severity))
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))


def extract_hidden(model, processor, image, prompt, layer=3):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return fwd.hidden_states[layer][0, -1, :].float().cpu().numpy()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def auroc(scores, labels):
    """Hand-written AUROC. scores[i] is anomaly score, labels[i]=1 means OOD."""
    labels = np.array(labels)
    scores = np.array(scores)
    n_pos = int(labels.sum())
    n_neg = int((1 - labels).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    tp = np.cumsum(labels_sorted) / n_pos
    fp = np.cumsum(1 - labels_sorted) / n_neg
    fp = np.concatenate([[0.0], fp])
    tp = np.concatenate([[0.0], tp])
    return float(np.trapezoid(tp, fp)) if hasattr(np, 'trapezoid') else float(np.trapz(tp, fp))


def cosine_distance(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(1.0 - np.dot(a, b) / (na * nb))


def compute_auroc_from_sets(id_embs, ood_embs, centroid):
    """Given clean and OOD embeddings compute AUROC using cosine distance."""
    id_scores = [cosine_distance(e, centroid) for e in id_embs]
    ood_scores = [cosine_distance(e, centroid) for e in ood_embs]
    scores = id_scores + ood_scores
    labels = [0] * len(id_scores) + [1] * len(ood_scores)
    return auroc(scores, labels)


def sigma_threshold(cal_embs, centroid, sigma=3.0):
    """Return centroid + sigma * std of distances over calibration set."""
    dists = np.array([cosine_distance(e, centroid) for e in cal_embs])
    return float(dists.mean() + sigma * dists.std()) if len(dists) > 1 else float(dists[0] * 2.0)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _convert(obj):
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def recursive_convert(d):
    if isinstance(d, dict):
        return {k: recursive_convert(v) for k, v in d.items()}
    if isinstance(d, list):
        return [recursive_convert(x) for x in d]
    return _convert(d)


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------

CAL_SEEDS = [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000,
             6000, 7000, 8000, 9000, 10000, 11000]   # 15 calibration scenes
TEST_SEEDS = [50000, 51000, 52000, 53000, 54000]      # 5 held-out test scenes
CORRUPTION_TYPES = ['fog', 'night', 'noise', 'blur']


# ---------------------------------------------------------------------------
# Analysis 1 – Streaming Centroid Update
# ---------------------------------------------------------------------------

def analysis_streaming_centroid(cal_embs, test_clean_embs, test_ood_embs_by_type):
    """Evaluate AUROC as the calibration set grows from 1..15 scenes."""
    print("\n=== Analysis 1: Streaming Centroid Update ===")
    results = []
    for n in range(1, len(cal_embs) + 1):
        subset = cal_embs[:n]
        centroid = np.mean(subset, axis=0)
        per_type = {}
        for ctype, ood_embs in test_ood_embs_by_type.items():
            auc = compute_auroc_from_sets(test_clean_embs, ood_embs, centroid)
            per_type[ctype] = float(auc)
        mean_auc = float(np.mean(list(per_type.values())))
        results.append({
            'n_cal_scenes': n,
            'auroc_per_type': per_type,
            'mean_auroc': mean_auc,
        })
        print(f"  N={n:2d}: mean AUROC={mean_auc:.4f}  " +
              "  ".join(f"{ct}={per_type[ct]:.3f}" for ct in CORRUPTION_TYPES))
    return results


# ---------------------------------------------------------------------------
# Analysis 2 – Running Mean vs Exponential Moving Average
# ---------------------------------------------------------------------------

def analysis_ema_comparison(cal_embs, test_clean_embs, test_ood_embs_by_type):
    """Compare running mean vs several EMA alpha values across streaming steps."""
    print("\n=== Analysis 2: Running Mean vs EMA ===")
    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
    strategies = ['running_mean'] + [f'ema_{a}' for a in alphas]

    # State: {strategy: centroid}
    state = {'running_mean': None}
    for a in alphas:
        state[f'ema_{a}'] = None

    results = {s: [] for s in strategies}

    for n, emb in enumerate(cal_embs, start=1):
        # Update centroids
        if state['running_mean'] is None:
            state['running_mean'] = emb.copy()
        else:
            # Equivalent to running mean
            state['running_mean'] = state['running_mean'] + (emb - state['running_mean']) / n

        for a in alphas:
            key = f'ema_{a}'
            if state[key] is None:
                state[key] = emb.copy()
            else:
                state[key] = a * emb + (1.0 - a) * state[key]

        # Evaluate all strategies
        for s in strategies:
            centroid = state[s]
            per_type = {}
            for ctype, ood_embs in test_ood_embs_by_type.items():
                per_type[ctype] = float(compute_auroc_from_sets(
                    test_clean_embs, ood_embs, centroid))
            mean_auc = float(np.mean(list(per_type.values())))
            results[s].append({
                'n_cal_scenes': n,
                'auroc_per_type': per_type,
                'mean_auroc': mean_auc,
            })

        if n <= 3 or n % 5 == 0 or n == len(cal_embs):
            line = f"  N={n:2d}: " + "  ".join(
                f"{s}={results[s][-1]['mean_auroc']:.3f}" for s in strategies)
            print(line)

    return results


# ---------------------------------------------------------------------------
# Analysis 3 – Threshold Adaptation (3-sigma stabilisation)
# ---------------------------------------------------------------------------

def analysis_threshold_adaptation(cal_embs, test_clean_embs, test_ood_embs_by_type):
    """Track how 3-sigma threshold, FPR, and TPR evolve as cal set grows."""
    print("\n=== Analysis 3: Threshold Adaptation ===")
    results = []
    for n in range(1, len(cal_embs) + 1):
        subset = cal_embs[:n]
        centroid = np.mean(subset, axis=0)
        thr = sigma_threshold(subset, centroid, sigma=3.0)

        # FPR on held-out clean test set
        clean_scores = [cosine_distance(e, centroid) for e in test_clean_embs]
        fpr = float(np.mean([s > thr for s in clean_scores]))

        # TPR averaged over corruption types
        tpr_list = []
        for ood_embs in test_ood_embs_by_type.values():
            ood_scores = [cosine_distance(e, centroid) for e in ood_embs]
            tpr_list.append(float(np.mean([s > thr for s in ood_scores])))
        mean_tpr = float(np.mean(tpr_list))

        results.append({
            'n_cal_scenes': n,
            'threshold': float(thr),
            'fpr': fpr,
            'mean_tpr': mean_tpr,
        })
        print(f"  N={n:2d}: threshold={thr:.6f}  FPR={fpr:.3f}  TPR={mean_tpr:.3f}")

    return results


# ---------------------------------------------------------------------------
# Analysis 4 – Distribution Shift: Running Mean vs EMA
# ---------------------------------------------------------------------------

def analysis_distribution_shift(model, processor, prompt):
    """Simulate gradual distribution shift via seeds that produce increasingly
    dissimilar images. Drift seeds are spaced so consecutive images diverge.
    Compare running-mean vs EMA tracking quality under this drift.
    """
    print("\n=== Analysis 4: Distribution Shift Simulation ===")

    # Use seeds that produce visually different images to simulate drift
    drift_seeds = [42, 5000, 10000, 20000, 40000,
                   80000, 160000, 320000, 640000, 1280000]
    # Two fixed OOD images for evaluation at each step
    ood_eval_seeds = [50000, 51000]

    print("  Extracting drift-scenario embeddings...")
    drift_embs = []
    for seed in drift_seeds:
        img = make_image(seed)
        emb = extract_hidden(model, processor, img, prompt)
        drift_embs.append(emb)
        print(f"    drift seed {seed} done")

    ood_eval_embs = []
    for seed in ood_eval_seeds:
        img = make_image(seed)
        for ct in CORRUPTION_TYPES:
            emb = extract_hidden(model, processor, apply_corruption(img, ct, 1.0), prompt)
            ood_eval_embs.append(emb)

    alphas = [0.3, 0.7]
    state_rm = None           # running mean
    state_ema = {a: None for a in alphas}
    rm_results = []
    ema_results = {a: [] for a in alphas}

    for n, emb in enumerate(drift_embs, start=1):
        clean_so_far = drift_embs[:n]  # "clean" at this drift step

        # Running mean update
        if state_rm is None:
            state_rm = emb.copy()
        else:
            state_rm = state_rm + (emb - state_rm) / n

        # EMA updates
        for a in alphas:
            if state_ema[a] is None:
                state_ema[a] = emb.copy()
            else:
                state_ema[a] = a * emb + (1.0 - a) * state_ema[a]

        # Evaluate: use the current drift image as "clean reference"
        id_embs = [emb]   # only latest clean as reference for current distribution
        rm_auc = compute_auroc_from_sets(id_embs, ood_eval_embs, state_rm)
        rm_results.append({'n': n, 'mean_auroc': float(rm_auc)})

        for a in alphas:
            auc = compute_auroc_from_sets(id_embs, ood_eval_embs, state_ema[a])
            ema_results[a].append({'n': n, 'mean_auroc': float(auc)})

        print(f"  step {n:2d}: rm_auc={rm_auc:.3f}  " +
              "  ".join(f"ema_{a}={ema_results[a][-1]['mean_auroc']:.3f}" for a in alphas))

    return {
        'drift_seeds': drift_seeds,
        'running_mean': rm_results,
        **{f'ema_{a}': ema_results[a] for a in alphas},
    }


# ---------------------------------------------------------------------------
# Analysis 5 – Cold Start Performance Distribution
# ---------------------------------------------------------------------------

def analysis_cold_start(cal_embs, test_clean_embs, test_ood_embs_by_type):
    """With just 1 calibration scene, evaluate AUROC for each possible
    single-scene start (using the 15 calibration seeds)."""
    print("\n=== Analysis 5: Cold Start Performance ===")
    results = []
    for i, emb in enumerate(cal_embs):
        centroid = emb.copy()  # single-scene centroid = the embedding itself
        per_type = {}
        for ctype, ood_embs in test_ood_embs_by_type.items():
            per_type[ctype] = float(compute_auroc_from_sets(
                test_clean_embs, ood_embs, centroid))
        mean_auc = float(np.mean(list(per_type.values())))
        results.append({
            'cal_seed': CAL_SEEDS[i],
            'auroc_per_type': per_type,
            'mean_auroc': mean_auc,
        })
        print(f"  seed={CAL_SEEDS[i]:6d}: mean AUROC={mean_auc:.4f}  " +
              "  ".join(f"{ct}={per_type[ct]:.3f}" for ct in CORRUPTION_TYPES))

    mean_aucs = [r['mean_auroc'] for r in results]
    summary = {
        'mean': float(np.mean(mean_aucs)),
        'std': float(np.std(mean_aucs)),
        'min': float(np.min(mean_aucs)),
        'max': float(np.max(mean_aucs)),
        'median': float(np.median(mean_aucs)),
    }
    print(f"  Summary: mean={summary['mean']:.4f} std={summary['std']:.4f} "
          f"min={summary['min']:.4f} max={summary['max']:.4f}")
    return {'per_seed': results, 'summary': summary}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Experiment 451: Online Calibration Adaptation  [{timestamp}]")
    print("=" * 60)

    # ---- Load model ----
    print("\nLoading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()
    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    print("Model loaded.")

    # ---- Extract calibration embeddings ----
    print("\nExtracting calibration embeddings (15 scenes)...")
    cal_embs = []
    for idx, seed in enumerate(CAL_SEEDS):
        img = make_image(seed)
        emb = extract_hidden(model, processor, img, prompt)
        cal_embs.append(emb)
        print(f"  cal scene {idx+1:2d}/15  seed={seed}")

    # ---- Extract test embeddings (clean + all corruption types) ----
    print("\nExtracting test embeddings (5 held-out scenes × 5 variants each)...")
    test_clean_embs = []
    test_ood_embs_by_type = {ct: [] for ct in CORRUPTION_TYPES}

    for idx, seed in enumerate(TEST_SEEDS):
        img = make_image(seed)
        emb = extract_hidden(model, processor, img, prompt)
        test_clean_embs.append(emb)
        print(f"  test scene {idx+1}/5  seed={seed}  clean done")
        for ct in CORRUPTION_TYPES:
            emb_ood = extract_hidden(model, processor, apply_corruption(img, ct, 1.0), prompt)
            test_ood_embs_by_type[ct].append(emb_ood)
            print(f"    corruption={ct} done")

    # ---- Run analyses ----
    results = {
        'experiment': 'Experiment 451: Online Calibration Adaptation',
        'timestamp': timestamp,
        'cal_seeds': CAL_SEEDS,
        'test_seeds': TEST_SEEDS,
        'corruption_types': CORRUPTION_TYPES,
    }

    results['analysis_1_streaming_centroid'] = analysis_streaming_centroid(
        cal_embs, test_clean_embs, test_ood_embs_by_type)

    results['analysis_2_ema_comparison'] = analysis_ema_comparison(
        cal_embs, test_clean_embs, test_ood_embs_by_type)

    results['analysis_3_threshold_adaptation'] = analysis_threshold_adaptation(
        cal_embs, test_clean_embs, test_ood_embs_by_type)

    results['analysis_4_distribution_shift'] = analysis_distribution_shift(
        model, processor, prompt)

    results['analysis_5_cold_start'] = analysis_cold_start(
        cal_embs, test_clean_embs, test_ood_embs_by_type)

    # ---- Save ----
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"exp451_online_calibration_adaptation_{timestamp}.json")

    results_clean = recursive_convert(results)
    with open(out_path, 'w') as f:
        json.dump(results_clean, f, indent=2)

    print(f"\nResults saved to {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
