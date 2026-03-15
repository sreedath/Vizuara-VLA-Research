#!/usr/bin/env python3
"""Experiment 344: Multi-Scene Deployment Protocol

Test practical deployment scenarios:
1. Scene transition detection and automatic recalibration
2. Calibration bank management (store centroids per scene)
3. Unknown scene detection (no matching calibration)
4. Calibration decay under gradual scene evolution
5. Minimum calibration bank size for coverage
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
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return 1.0 - dot / (na * nb)

def compute_auroc(id_scores, ood_scores):
    id_s = np.asarray(id_scores)
    ood_s = np.asarray(ood_scores)
    n_id, n_ood = len(id_s), len(ood_s)
    if n_id == 0 or n_ood == 0:
        return 0.5
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

    # Create 10 diverse scenes
    seeds = list(range(0, 1000, 100))
    scenes = {}
    cal_embs = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        px = rng.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        scenes[seed] = Image.fromarray(px)
        cal_embs[seed] = extract_hidden(model, processor, scenes[seed], prompt)
        print(f"  Scene {seed} calibrated")

    ctypes = ['fog', 'night', 'noise', 'blur']

    # ========== 1. Cross-scene distances ==========
    print("\n=== Cross-Scene Distances ===")

    cross_dists = {}
    for i, s1 in enumerate(seeds):
        for j, s2 in enumerate(seeds):
            if j <= i:
                continue
            d = cosine_dist(cal_embs[s1], cal_embs[s2])
            cross_dists[f"{s1}_{s2}"] = float(d)

    dists_list = list(cross_dists.values())
    results['cross_scene'] = {
        'pairwise': cross_dists,
        'mean': float(np.mean(dists_list)),
        'min': float(np.min(dists_list)),
        'max': float(np.max(dists_list)),
        'std': float(np.std(dists_list)),
    }
    print(f"  Cross-scene: mean={np.mean(dists_list):.6f}, "
          f"range=[{np.min(dists_list):.6f}, {np.max(dists_list):.6f}]")

    # ========== 2. Nearest-centroid detection ==========
    print("\n=== Nearest-Centroid Protocol ===")

    # For each scene, test if nearest centroid from bank works
    nc_results = {}
    for test_seed in seeds:
        bank_seeds = [s for s in seeds if s != test_seed]

        # Find nearest centroid
        dists_to_bank = [(s, cosine_dist(cal_embs[test_seed], cal_embs[s])) for s in bank_seeds]
        nearest_seed, nearest_dist = min(dists_to_bank, key=lambda x: x[1])

        # Test OOD detection using nearest centroid
        ood_dists = []
        for ct in ctypes:
            img = apply_corruption(scenes[test_seed], ct, 0.5)
            emb = extract_hidden(model, processor, img, prompt)
            d = cosine_dist(cal_embs[nearest_seed], emb)
            ood_dists.append(float(d))

        # Test with own centroid for comparison
        own_ood_dists = []
        for ct in ctypes:
            img = apply_corruption(scenes[test_seed], ct, 0.5)
            emb = extract_hidden(model, processor, img, prompt)
            d = cosine_dist(cal_embs[test_seed], emb)
            own_ood_dists.append(float(d))

        # Clean distance using nearest centroid
        clean_dist = float(cosine_dist(cal_embs[nearest_seed], cal_embs[test_seed]))

        auroc_nearest = compute_auroc([clean_dist], ood_dists)
        auroc_own = compute_auroc([0.0], own_ood_dists)

        nc_results[str(test_seed)] = {
            'nearest_seed': nearest_seed,
            'nearest_dist': float(nearest_dist),
            'auroc_nearest': float(auroc_nearest),
            'auroc_own': float(auroc_own),
            'clean_dist_nearest': clean_dist,
            'min_ood_nearest': float(min(ood_dists)),
        }
        print(f"  Scene {test_seed}: nearest={nearest_seed} (d={nearest_dist:.6f}), "
              f"AUROC_nearest={auroc_nearest:.3f}, AUROC_own={auroc_own:.3f}")

    results['nearest_centroid'] = nc_results

    # ========== 3. Calibration bank size sweep ==========
    print("\n=== Bank Size Sweep ===")

    bank_size_results = {}
    for bank_size in [1, 2, 3, 5, 7, 9]:
        aurocs = []
        for test_seed in seeds:
            # Build bank from random subset (excluding test)
            available = [s for s in seeds if s != test_seed]
            rng = np.random.RandomState(42 + test_seed)
            bank = rng.choice(available, min(bank_size, len(available)), replace=False)

            # Find nearest in bank
            dists_to_bank = [(s, cosine_dist(cal_embs[test_seed], cal_embs[s])) for s in bank]
            nearest_seed, nearest_dist = min(dists_to_bank, key=lambda x: x[1])

            # Test
            clean_dist = float(cosine_dist(cal_embs[nearest_seed], cal_embs[test_seed]))
            ood_dists = []
            for ct in ctypes:
                img = apply_corruption(scenes[test_seed], ct, 0.5)
                emb = extract_hidden(model, processor, img, prompt)
                d = cosine_dist(cal_embs[nearest_seed], emb)
                ood_dists.append(float(d))

            auroc = compute_auroc([clean_dist], ood_dists)
            aurocs.append(auroc)

        bank_size_results[str(bank_size)] = {
            'mean_auroc': float(np.mean(aurocs)),
            'min_auroc': float(np.min(aurocs)),
            'all_perfect': bool(all(a == 1.0 for a in aurocs)),
        }
        print(f"  Bank={bank_size}: mean_auroc={np.mean(aurocs):.3f}, min={np.min(aurocs):.3f}")

    results['bank_size'] = bank_size_results

    # ========== 4. Gradual scene evolution ==========
    print("\n=== Scene Evolution ===")

    evolution = {}
    base_seed = 0
    base_arr = np.array(scenes[0]).astype(np.float32)
    cal = cal_embs[0]

    for drift_pct in [1, 2, 5, 10, 15, 20, 30, 50]:
        rng = np.random.RandomState(99)
        drift = rng.randn(*base_arr.shape) * drift_pct / 100 * 255
        drifted = np.clip(base_arr + drift, 0, 255).astype(np.uint8)
        img = Image.fromarray(drifted)
        emb = extract_hidden(model, processor, img, prompt)
        drift_dist = cosine_dist(cal, emb)

        # Can we still detect corruption on the drifted scene?
        ood_dists = []
        for ct in ctypes:
            corrupt_img = apply_corruption(img, ct, 0.5)
            corrupt_emb = extract_hidden(model, processor, corrupt_img, prompt)
            d = cosine_dist(cal, corrupt_emb)
            ood_dists.append(float(d))

        auroc = compute_auroc([drift_dist], ood_dists)

        evolution[str(drift_pct)] = {
            'drift_distance': float(drift_dist),
            'min_ood_distance': float(min(ood_dists)),
            'auroc': float(auroc),
            'detection_gap': float(min(ood_dists) - drift_dist),
        }
        print(f"  Drift {drift_pct}%: drift_d={drift_dist:.6f}, AUROC={auroc:.3f}")

    results['evolution'] = evolution

    # ========== 5. Scene transition detection ==========
    print("\n=== Scene Transition Detection ===")

    # Simulate a sequence where scenes change
    transition = {}
    sequence = [0, 0, 0, 100, 100, 100, 200, 200, 200, 0, 0, 0]
    current_cal = cal_embs[0]

    for i, seed in enumerate(sequence):
        emb = extract_hidden(model, processor, scenes[seed], prompt)
        d = cosine_dist(current_cal, emb)

        # Is this a scene transition?
        is_transition = (i > 0 and sequence[i] != sequence[i-1])
        # Should we recalibrate?
        needs_recal = d > 0.001  # threshold for scene change

        transition[str(i)] = {
            'scene': seed,
            'dist_to_cal': float(d),
            'is_transition': is_transition,
            'detected_transition': bool(needs_recal),
            'correct': bool(is_transition == needs_recal),
        }

        if needs_recal:
            current_cal = emb  # recalibrate

    n_correct = sum(1 for v in transition.values() if v['correct'])
    results['transition'] = {
        'sequence': transition,
        'accuracy': float(n_correct / len(transition)),
        'n_transitions': sum(1 for v in transition.values() if v['is_transition']),
    }
    print(f"  Transition detection accuracy: {results['transition']['accuracy']:.3f}")

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/multi_scene_protocol_{ts}.json"
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
