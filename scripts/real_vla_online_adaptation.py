#!/usr/bin/env python3
"""Experiment 387: Online Calibration Adaptation

Can the detector adapt its calibration during deployment?
1. Expanding calibration set: add new clean samples over time
2. Threshold decay: how quickly does adding OOD to calibration break it?
3. Contaminated calibration: what % of corrupt samples in calibration is tolerable?
4. Rolling centroid: moving average vs full history
5. Drift detection: can we detect when calibration is stale?
"""

import json, time, os, sys
import numpy as np
import torch
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor

def extract_hidden(model, processor, image, prompt, layer=3):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return fwd.hidden_states[layer][0, -1, :].float().cpu().numpy()

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

def cosine_dist(a, b):
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10: return 0.0
    return 1.0 - dot / (na * nb)

def compute_auroc(id_scores, ood_scores):
    id_s, ood_s = np.asarray(id_scores), np.asarray(ood_scores)
    n_id, n_ood = len(id_s), len(ood_s)
    if n_id == 0 or n_ood == 0: return 0.5
    count = sum(float(np.sum(o > id_s) + 0.5 * np.sum(o == id_s)) for o in ood_s)
    return count / (n_id * n_ood)

def main():
    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    results = {}
    ctypes = ['fog', 'night', 'noise', 'blur']

    print("Generating images...")
    seeds = list(range(0, 3000, 100))[:30]
    images = {}
    clean_embs = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        px = rng.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        images[seed] = Image.fromarray(px)
        clean_embs[seed] = extract_hidden(model, processor, images[seed], prompt)

    # Split: first 10 for calibration testing, next 10 for validation, last 10 for OOD testing
    cal_seeds = seeds[:10]
    val_seeds = seeds[10:20]
    test_seeds = seeds[20:30]

    # ========== 1. Expanding Calibration Set ==========
    print("\n=== Expanding Calibration Set ===")

    expanding = {}
    for n_cal in [1, 2, 3, 5, 7, 10]:
        cal_embs = [clean_embs[cal_seeds[i]] for i in range(n_cal)]
        centroid = np.mean(cal_embs, axis=0)
        cal_dists = [cosine_dist(centroid, e) for e in cal_embs]
        threshold = max(cal_dists) if len(cal_dists) > 1 else cosine_dist(centroid, cal_embs[0]) * 2

        # Test on validation set
        val_dists = [cosine_dist(centroid, clean_embs[s]) for s in val_seeds]
        fpr = sum(1 for d in val_dists if d > threshold) / len(val_dists)

        # Test detection
        ct_aurocs = {}
        for ct in ctypes:
            ood_dists = []
            for s in test_seeds[:5]:
                emb = extract_hidden(model, processor, apply_corruption(images[s], ct, 0.5), prompt)
                ood_dists.append(cosine_dist(emb, centroid))
            ct_aurocs[ct] = compute_auroc(val_dists, ood_dists)

        expanding[str(n_cal)] = {
            'fpr': float(fpr),
            'threshold': float(threshold),
            'aurocs': ct_aurocs,
        }
        print(f"  N={n_cal}: FPR={fpr:.2f}, aurocs=" +
              ", ".join(f"{ct}={ct_aurocs[ct]:.2f}" for ct in ctypes))

    results['expanding_calibration'] = expanding

    # ========== 2. Contaminated Calibration ==========
    print("\n=== Contaminated Calibration ===")

    contaminated = {}
    for contamination_pct in [0, 10, 20, 30, 50, 70]:
        n_total = 10
        n_corrupt = int(n_total * contamination_pct / 100)
        n_clean = n_total - n_corrupt

        cal_embs = [clean_embs[cal_seeds[i]] for i in range(n_clean)]
        # Add corrupt embeddings
        for i in range(n_corrupt):
            corrupt_img = apply_corruption(images[cal_seeds[n_clean + i]], 'fog', 0.5)
            cal_embs.append(extract_hidden(model, processor, corrupt_img, prompt))

        centroid = np.mean(cal_embs, axis=0)
        cal_dists = [cosine_dist(centroid, e) for e in cal_embs]
        threshold = max(cal_dists)

        # Test
        val_dists = [cosine_dist(centroid, clean_embs[s]) for s in val_seeds]
        fpr = sum(1 for d in val_dists if d > threshold) / len(val_dists)

        ct_aurocs = {}
        for ct in ctypes:
            ood_dists = []
            for s in test_seeds[:5]:
                emb = extract_hidden(model, processor, apply_corruption(images[s], ct, 0.5), prompt)
                ood_dists.append(cosine_dist(emb, centroid))
            ct_aurocs[ct] = compute_auroc(val_dists, ood_dists)

        contaminated[str(contamination_pct)] = {
            'fpr': float(fpr),
            'aurocs': ct_aurocs,
        }
        print(f"  {contamination_pct}% contaminated: FPR={fpr:.2f}, aurocs=" +
              ", ".join(f"{ct}={ct_aurocs[ct]:.2f}" for ct in ctypes))

    results['contaminated_calibration'] = contaminated

    # ========== 3. Rolling Centroid ==========
    print("\n=== Rolling Centroid ===")

    rolling = {}
    for window in [3, 5, 10, 'full']:
        all_embs = [clean_embs[s] for s in cal_seeds]

        if window == 'full':
            centroid = np.mean(all_embs, axis=0)
        else:
            centroid = np.mean(all_embs[-window:], axis=0)

        cal_dists = [cosine_dist(centroid, e) for e in all_embs]
        threshold = max(cal_dists)

        val_dists = [cosine_dist(centroid, clean_embs[s]) for s in val_seeds]
        fpr = sum(1 for d in val_dists if d > threshold) / len(val_dists)

        ct_aurocs = {}
        for ct in ctypes:
            ood_dists = []
            for s in test_seeds[:5]:
                emb = extract_hidden(model, processor, apply_corruption(images[s], ct, 0.5), prompt)
                ood_dists.append(cosine_dist(emb, centroid))
            ct_aurocs[ct] = compute_auroc(val_dists, ood_dists)

        rolling[str(window)] = {
            'fpr': float(fpr),
            'aurocs': ct_aurocs,
        }
        print(f"  window={window}: FPR={fpr:.2f}, aurocs=" +
              ", ".join(f"{ct}={ct_aurocs[ct]:.2f}" for ct in ctypes))

    results['rolling_centroid'] = rolling

    # ========== 4. Centroid Stability Over Time ==========
    print("\n=== Centroid Stability ===")

    stability = []
    running_embs = []
    for i, seed in enumerate(seeds[:20]):
        running_embs.append(clean_embs[seed])
        if len(running_embs) >= 3:
            curr_centroid = np.mean(running_embs, axis=0)
            prev_centroid = np.mean(running_embs[:-1], axis=0)
            drift = cosine_dist(curr_centroid, prev_centroid)
            stability.append({
                'n': len(running_embs),
                'drift': float(drift),
            })

    results['centroid_stability'] = stability
    drifts = [s['drift'] for s in stability]
    print(f"  Max drift: {max(drifts):.8f}, Mean: {np.mean(drifts):.8f}")
    print(f"  Converged (drift < 1e-6) at N={next((s['n'] for s in stability if s['drift'] < 1e-6), 'never')}")

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/online_adaptation_{ts}.json"
    def convert(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)): return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return obj
    def recursive_convert(d):
        if isinstance(d, dict): return {k: recursive_convert(v) for k, v in d.items()}
        if isinstance(d, list): return [recursive_convert(x) for x in d]
        return convert(d)
    results = recursive_convert(results)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
