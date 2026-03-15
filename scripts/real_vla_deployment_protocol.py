#!/usr/bin/env python3
"""Experiment 316: Deployment Protocol Specification
Complete deployment procedure validation:
1. Minimum viable calibration (single image, timing)
2. Threshold selection strategies (fixed, adaptive, percentile)
3. Multi-environment calibration protocol
4. Graceful degradation under increasing corruption
5. Recovery protocol validation
6. Resource scaling (memory, compute vs accuracy)
"""

import torch
import numpy as np
import json
import time
from datetime import datetime
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor
from scipy.spatial.distance import cosine

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

def compute_auroc(id_scores, ood_scores):
    id_s = np.asarray(id_scores)
    ood_s = np.asarray(ood_scores)
    n_id, n_ood = len(id_s), len(ood_s)
    if n_id == 0 or n_ood == 0:
        return 0.5
    count = sum(float(np.sum(o > id_s) + 0.5 * np.sum(o == id_s)) for o in ood_s)
    return count / (n_id * n_ood)

def main():
    print("Loading OpenVLA-7B...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\nOut:"

    results = {
        "experiment": "deployment_protocol",
        "experiment_number": 316,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    corruptions = ['fog', 'night', 'blur', 'noise']

    # Part 1: Minimum Viable Calibration
    print("=== Part 1: Minimum Viable Calibration ===")
    # Test with 8 different scenes
    scenes = []
    for seed in [0, 13, 42, 77, 99, 123, 456, 777]:
        np.random.seed(seed)
        px = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        scenes.append((seed, Image.fromarray(px)))

    cal_results = {}
    for n_cal in [1, 2, 3, 5]:
        cal_scenes = scenes[:n_cal]

        # Calibrate: get centroid
        t0 = time.time()
        cal_embs = [extract_hidden(model, processor, img, prompt) for _, img in cal_scenes]
        centroid = np.mean(cal_embs, axis=0)
        cal_time = (time.time() - t0) * 1000

        # Test on ALL scenes (in-distribution) and corruptions (OOD)
        id_dists = []
        ood_dists = []

        for seed, img in scenes:
            emb = extract_hidden(model, processor, img, prompt)
            d = float(cosine(centroid, emb))
            id_dists.append(d)

            for c in corruptions:
                corrupted = apply_corruption(img, c, 0.5)
                emb_c = extract_hidden(model, processor, corrupted, prompt)
                d_c = float(cosine(centroid, emb_c))
                ood_dists.append(d_c)

        auroc = compute_auroc(id_dists, ood_dists)
        cal_results[n_cal] = {
            "n_scenes": n_cal,
            "cal_time_ms": float(cal_time),
            "id_max": float(max(id_dists)),
            "ood_min": float(min(ood_dists)),
            "auroc": auroc,
            "gap": float(min(ood_dists) - max(id_dists)),
        }
        print(f"  N={n_cal}: AUROC={auroc:.4f}, gap={min(ood_dists)-max(id_dists):.6f}, "
              f"cal_time={cal_time:.0f}ms")

    results["calibration"] = cal_results

    # Part 2: Per-Scene Calibration Protocol
    print("\n=== Part 2: Per-Scene Calibration ===")
    per_scene = {}

    for seed, img in scenes:
        emb = extract_hidden(model, processor, img, prompt)

        # Test clean stability
        emb2 = extract_hidden(model, processor, img, prompt)
        clean_d = float(cosine(emb, emb2))

        # Test OOD detection
        ood_dists_scene = []
        for c in corruptions:
            corrupted = apply_corruption(img, c, 0.5)
            emb_c = extract_hidden(model, processor, corrupted, prompt)
            d_c = float(cosine(emb, emb_c))
            ood_dists_scene.append(d_c)

        per_scene[seed] = {
            "clean_distance": clean_d,
            "ood_distances": dict(zip(corruptions, ood_dists_scene)),
            "min_ood": float(min(ood_dists_scene)),
            "perfect": clean_d == 0.0 and min(ood_dists_scene) > 0,
        }
        print(f"  Seed {seed}: clean_d={clean_d:.10f}, min_ood={min(ood_dists_scene):.6f}, "
              f"perfect={clean_d == 0.0 and min(ood_dists_scene) > 0}")

    results["per_scene"] = per_scene

    # Part 3: Threshold Strategy Comparison
    print("\n=== Part 3: Threshold Strategies ===")
    # Use seed 42 as primary, test across all scenes
    np.random.seed(42)
    px_main = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    main_img = Image.fromarray(px_main)
    main_emb = extract_hidden(model, processor, main_img, prompt)

    # Collect distances
    all_clean_dists = []
    all_ood_dists = []
    for seed, img in scenes:
        emb = extract_hidden(model, processor, img, prompt)
        all_clean_dists.append(float(cosine(main_emb, emb)))
        for c in corruptions:
            for sev in [0.1, 0.3, 0.5, 0.7, 1.0]:
                corrupted = apply_corruption(img, c, sev)
                emb_c = extract_hidden(model, processor, corrupted, prompt)
                all_ood_dists.append(float(cosine(main_emb, emb_c)))

    threshold_strategies = {}

    # Strategy 1: Fixed tau = 0
    tp = sum(1 for d in all_ood_dists if d > 0)
    tn = sum(1 for d in all_clean_dists if d == 0)
    fp = sum(1 for d in all_clean_dists if d > 0)
    fn = sum(1 for d in all_ood_dists if d == 0)
    threshold_strategies["tau=0"] = {
        "threshold": 0,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "sensitivity": tp/(tp+fn) if tp+fn>0 else 0,
        "specificity": tn/(tn+fp) if tn+fp>0 else 0,
    }

    # Strategy 2: tau = midpoint(clean_max, ood_min)
    clean_max = max(all_clean_dists)
    ood_min = min(d for d in all_ood_dists if d > 0) if any(d > 0 for d in all_ood_dists) else 0
    tau_mid = (clean_max + ood_min) / 2
    tp = sum(1 for d in all_ood_dists if d > tau_mid)
    tn = sum(1 for d in all_clean_dists if d <= tau_mid)
    fp = sum(1 for d in all_clean_dists if d > tau_mid)
    fn = sum(1 for d in all_ood_dists if d <= tau_mid)
    threshold_strategies["tau=midpoint"] = {
        "threshold": float(tau_mid),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "sensitivity": tp/(tp+fn) if tp+fn>0 else 0,
        "specificity": tn/(tn+fp) if tn+fp>0 else 0,
    }

    # Strategy 3: Per-scene calibration
    tp_ps = 0
    tn_ps = 0
    fp_ps = 0
    fn_ps = 0
    for seed, img in scenes:
        scene_emb = extract_hidden(model, processor, img, prompt)
        # Clean
        d_clean = float(cosine(scene_emb, extract_hidden(model, processor, img, prompt)))
        if d_clean == 0:
            tn_ps += 1
        else:
            fp_ps += 1
        # OOD
        for c in corruptions:
            for sev in [0.1, 0.3, 0.5, 0.7, 1.0]:
                corrupted = apply_corruption(img, c, sev)
                d_ood = float(cosine(scene_emb, extract_hidden(model, processor, corrupted, prompt)))
                if d_ood > 0:
                    tp_ps += 1
                else:
                    fn_ps += 1

    threshold_strategies["per_scene"] = {
        "threshold": 0,
        "tp": tp_ps, "tn": tn_ps, "fp": fp_ps, "fn": fn_ps,
        "sensitivity": tp_ps/(tp_ps+fn_ps) if tp_ps+fn_ps>0 else 0,
        "specificity": tn_ps/(tn_ps+fp_ps) if tn_ps+fp_ps>0 else 0,
    }

    results["threshold_strategies"] = threshold_strategies
    for name, s in threshold_strategies.items():
        print(f"  {name}: sens={s['sensitivity']:.3f}, spec={s['specificity']:.3f}, "
              f"TP={s['tp']}, FP={s['fp']}, FN={s['fn']}, TN={s['tn']}")

    # Part 4: Graceful Degradation Profile
    print("\n=== Part 4: Graceful Degradation ===")
    np.random.seed(42)
    base_img = Image.fromarray(np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8))
    base_emb = extract_hidden(model, processor, base_img, prompt)

    degradation = {}
    for c in corruptions:
        profile = []
        for sev in np.linspace(0, 1.0, 21):
            if sev == 0:
                d = 0.0
            else:
                corrupted = apply_corruption(base_img, c, float(sev))
                emb = extract_hidden(model, processor, corrupted, prompt)
                d = float(cosine(base_emb, emb))
            profile.append({"severity": float(sev), "distance": d})
        degradation[c] = profile

    results["degradation"] = degradation

    # Part 5: Recovery Protocol
    print("\n=== Part 5: Recovery Protocol ===")
    # Simulate: clean → corruption → recovery → clean
    recovery = []

    # 5 clean
    for i in range(5):
        d = float(cosine(base_emb, extract_hidden(model, processor, base_img, prompt)))
        recovery.append({"frame": i, "condition": "clean", "distance": d})

    # 5 increasing fog
    for i in range(5):
        sev = (i + 1) * 0.2
        corrupted = apply_corruption(base_img, 'fog', sev)
        d = float(cosine(base_emb, extract_hidden(model, processor, corrupted, prompt)))
        recovery.append({"frame": 5 + i, "condition": f"fog_{sev:.1f}", "distance": d})

    # 5 decreasing fog
    for i in range(5):
        sev = 1.0 - (i + 1) * 0.2
        if sev <= 0:
            d = float(cosine(base_emb, extract_hidden(model, processor, base_img, prompt)))
            recovery.append({"frame": 10 + i, "condition": "clean", "distance": d})
        else:
            corrupted = apply_corruption(base_img, 'fog', sev)
            d = float(cosine(base_emb, extract_hidden(model, processor, corrupted, prompt)))
            recovery.append({"frame": 10 + i, "condition": f"fog_{sev:.1f}", "distance": d})

    # 5 clean recovery
    for i in range(5):
        d = float(cosine(base_emb, extract_hidden(model, processor, base_img, prompt)))
        recovery.append({"frame": 15 + i, "condition": "clean", "distance": d})

    results["recovery"] = recovery
    for r in recovery:
        print(f"  Frame {r['frame']}: {r['condition']} d={r['distance']:.8f}")

    # Part 6: Resource Scaling
    print("\n=== Part 6: Resource Scaling ===")
    resource_scaling = {}

    # Different embedding dimensions (via truncation)
    for n_dims in [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
        clean_d = float(cosine(base_emb[:n_dims], base_emb[:n_dims]))
        ood_dists_trunc = []
        for c in corruptions:
            corrupted = apply_corruption(base_img, c, 0.5)
            emb_c = extract_hidden(model, processor, corrupted, prompt)
            d_c = float(cosine(base_emb[:n_dims], emb_c[:n_dims]))
            ood_dists_trunc.append(d_c)

        resource_scaling[n_dims] = {
            "dims": n_dims,
            "storage_bytes": n_dims * 4,
            "clean_distance": clean_d,
            "min_ood_distance": float(min(ood_dists_trunc)),
            "all_detected": all(d > 0 for d in ood_dists_trunc),
        }

    results["resource_scaling"] = resource_scaling
    for n, v in resource_scaling.items():
        print(f"  {n}D ({v['storage_bytes']}B): clean={v['clean_distance']:.10f}, "
              f"min_ood={v['min_ood_distance']:.6f}, all_detected={v['all_detected']}")

    # Save
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(v) for v in obj]
        return obj

    ts = results["timestamp"]
    out_path = f"experiments/deploy_{ts}.json"
    with open(out_path, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
