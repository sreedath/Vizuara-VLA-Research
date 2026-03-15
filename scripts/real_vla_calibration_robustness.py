#!/usr/bin/env python3
"""Experiment 330: Calibration Robustness (Real OpenVLA-7B)

Tests detector robustness under various calibration conditions:
1. Stale calibration: what if scene changes after calibration?
2. Calibration with slightly corrupted image
3. Multi-scene centroid effectiveness
4. Random reference point
5. Per-scene vs global calibration
6. Calibration with gradual scene drift
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

    scenes = {}
    scene_embs = {}
    for seed in [0, 42, 99, 123, 255, 777, 1000, 2000, 5000, 9999]:
        rng = np.random.RandomState(seed)
        px = rng.randint(50, 200, (224, 224, 3)).astype(np.uint8)
        img = Image.fromarray(px)
        emb = extract_hidden(model, processor, img, prompt)
        scenes[seed] = img
        scene_embs[seed] = emb

    ctypes = ['fog', 'night', 'noise', 'blur']

    # ========== 1. Stale calibration ==========
    print("\n=== Stale Calibration ===")
    stale_results = {}
    cal_emb = scene_embs[42]
    for test_seed in [0, 99, 123, 777, 9999]:
        test_img = scenes[test_seed]
        test_emb = scene_embs[test_seed]
        clean_d = cosine_dist(cal_emb, test_emb)
        ood_dists = {}
        for ct in ctypes:
            img = apply_corruption(test_img, ct, 0.5)
            emb = extract_hidden(model, processor, img, prompt)
            ood_dists[ct] = float(cosine_dist(cal_emb, emb))
        separable = min(ood_dists.values()) > clean_d
        stale_results[f"cal42_test{test_seed}"] = {
            'clean_dist': float(clean_d), 'ood_dists': ood_dists,
            'separable': bool(separable), 'margin': float(min(ood_dists.values()) - clean_d),
        }
        print(f"  cal=42, test={test_seed}: clean_d={clean_d:.6f}, min_ood={min(ood_dists.values()):.6f}, sep={separable}")
    results['stale_calibration'] = stale_results

    # ========== 2. Corrupted calibration ==========
    print("\n=== Corrupted Calibration ===")
    corrupt_cal_results = {}
    base_img = scenes[42]
    true_clean = scene_embs[42]
    for cal_corruption in ['fog', 'noise', 'blur']:
        for cal_sev in [0.01, 0.05, 0.1, 0.2]:
            cal_img = apply_corruption(base_img, cal_corruption, cal_sev)
            cal_emb_c = extract_hidden(model, processor, cal_img, prompt)
            clean_d = cosine_dist(cal_emb_c, true_clean)
            ood_dists = []
            for ct in ctypes:
                img = apply_corruption(base_img, ct, 0.5)
                emb = extract_hidden(model, processor, img, prompt)
                ood_dists.append(float(cosine_dist(cal_emb_c, emb)))
            auroc = compute_auroc([clean_d], ood_dists)
            key = f"{cal_corruption}_{cal_sev}"
            corrupt_cal_results[key] = {
                'cal_clean_dist': float(clean_d), 'ood_dists': ood_dists, 'auroc': float(auroc),
            }
            print(f"  cal={cal_corruption}@{cal_sev}: AUROC={auroc:.3f}")
    results['corrupted_calibration'] = corrupt_cal_results

    # ========== 3. Multi-scene centroid ==========
    print("\n=== Multi-Scene Centroid ===")
    centroid_results = {}
    for n_cal in [1, 2, 3, 5, 8]:
        cal_seeds = list(scenes.keys())[:n_cal]
        centroid = np.mean([scene_embs[s] for s in cal_seeds], axis=0)
        test_seeds = [s for s in scenes.keys() if s not in cal_seeds]
        clean_dists = [cosine_dist(centroid, scene_embs[s]) for s in test_seeds]
        ood_dists = []
        for s in test_seeds[:3]:
            for ct in ctypes:
                img = apply_corruption(scenes[s], ct, 0.5)
                emb = extract_hidden(model, processor, img, prompt)
                ood_dists.append(float(cosine_dist(centroid, emb)))
        auroc = compute_auroc(clean_dists, ood_dists)
        centroid_results[f"n={n_cal}"] = {
            'auroc': float(auroc),
            'max_clean_dist': float(np.max(clean_dists)) if clean_dists else 0,
            'min_ood_dist': float(np.min(ood_dists)),
            'separable': bool(min(ood_dists) > max(clean_dists)) if clean_dists else True,
        }
        print(f"  n={n_cal}: AUROC={auroc:.3f}")
    results['multi_scene_centroid'] = centroid_results

    # ========== 4. Random reference ==========
    print("\n=== Random Reference ===")
    random_ref_results = {}
    for trial in range(5):
        rng = np.random.RandomState(trial)
        random_emb = rng.randn(4096).astype(np.float32)
        random_emb = random_emb / np.linalg.norm(random_emb) * np.linalg.norm(scene_embs[42])
        clean_d = cosine_dist(random_emb, scene_embs[42])
        ood_dists = []
        for ct in ctypes:
            img = apply_corruption(scenes[42], ct, 0.5)
            emb = extract_hidden(model, processor, img, prompt)
            ood_dists.append(float(cosine_dist(random_emb, emb)))
        auroc = compute_auroc([clean_d], ood_dists)
        random_ref_results[f"trial_{trial}"] = {
            'clean_dist': float(clean_d), 'auroc': float(auroc),
        }
        print(f"  Trial {trial}: AUROC={auroc:.3f}")
    results['random_reference'] = random_ref_results

    # ========== 5. Per-scene vs global ==========
    print("\n=== Per-Scene vs Global ===")
    global_centroid = np.mean(list(scene_embs.values()), axis=0)
    comparison_results = {}
    for test_seed in [42, 99, 777]:
        test_img = scenes[test_seed]
        global_clean = cosine_dist(global_centroid, scene_embs[test_seed])
        per_scene_ood = []
        global_ood = []
        for ct in ctypes:
            img = apply_corruption(test_img, ct, 0.5)
            emb = extract_hidden(model, processor, img, prompt)
            per_scene_ood.append(float(cosine_dist(scene_embs[test_seed], emb)))
            global_ood.append(float(cosine_dist(global_centroid, emb)))
        comparison_results[f"scene_{test_seed}"] = {
            'per_scene_clean': 0.0, 'global_clean': float(global_clean),
            'per_scene_min_ood': float(min(per_scene_ood)),
            'global_min_ood': float(min(global_ood)),
            'per_scene_sep': True, 'global_sep': bool(min(global_ood) > global_clean),
        }
        print(f"  Scene {test_seed}: global_sep={min(global_ood) > global_clean}")
    results['per_scene_vs_global'] = comparison_results

    # ========== 6. Scene drift ==========
    print("\n=== Scene Drift ===")
    drift_results = {}
    base_arr = np.array(scenes[42]).astype(float)
    cal_emb_d = scene_embs[42]
    for drift_pct in [0, 1, 2, 5, 10, 20, 50]:
        rng_d = np.random.RandomState(123)
        drift = rng_d.randn(*base_arr.shape) * drift_pct / 100 * 255
        drifted_arr = np.clip(base_arr + drift, 0, 255).astype(np.uint8)
        drifted_img = Image.fromarray(drifted_arr)
        drifted_emb = extract_hidden(model, processor, drifted_img, prompt)
        clean_d = cosine_dist(cal_emb_d, drifted_emb)
        ood_dists = []
        for ct in ['fog', 'blur']:
            img = apply_corruption(drifted_img, ct, 0.5)
            emb = extract_hidden(model, processor, img, prompt)
            ood_dists.append(float(cosine_dist(cal_emb_d, emb)))
        drift_results[f"drift_{drift_pct}pct"] = {
            'clean_dist': float(clean_d), 'ood_dists': ood_dists,
            'separable': bool(min(ood_dists) > clean_d),
            'margin': float(min(ood_dists) - clean_d),
        }
        print(f"  Drift {drift_pct}%: clean_d={clean_d:.6f}, margin={min(ood_dists)-clean_d:.6f}")
    results['scene_drift'] = drift_results

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/cal_robustness_{ts}.json"
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
