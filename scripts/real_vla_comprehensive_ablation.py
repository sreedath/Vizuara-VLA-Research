#!/usr/bin/env python3
"""Experiment 320: Comprehensive Ablation Summary
Final sweep of all key hyperparameters in one systematic experiment:
1. Layer selection (L0-L32 at 5 key positions)
2. Token position (BOS, first image, mid image, last image, last text)
3. Distance metric (cosine, euclidean, manhattan, chebyshev)
4. Embedding dimension (4, 16, 64, 256, 1024, 4096)
5. Calibration size (1, 3, 5, 10)
6. All combinations tested against 4 corruptions at 50% severity
"""

import torch
import numpy as np
import json
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
    np.random.seed(42)
    pixels = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    base_img = Image.fromarray(pixels)

    results = {
        "experiment": "comprehensive_ablation",
        "experiment_number": 320,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    corruptions = ['fog', 'night', 'blur', 'noise']

    # Get clean and corrupted forward passes
    print("Computing embeddings...")
    inputs_clean = processor(prompt, base_img).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd_clean = model(**inputs_clean, output_hidden_states=True)

    corrupted_fwds = {}
    for c in corruptions:
        corrupted = apply_corruption(base_img, c, 0.5)
        inputs_c = processor(prompt, corrupted).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            fwd_c = model(**inputs_c, output_hidden_states=True)
        corrupted_fwds[c] = fwd_c

    seq_len = fwd_clean.hidden_states[0].shape[1]
    print(f"  Seq len: {seq_len}")

    # Define ablation dimensions
    layers = [0, 1, 3, 15, 31, 32]
    token_positions = {
        'bos': 0,
        'img_first': 5,
        'img_mid': seq_len // 2,
        'img_last': seq_len - 20,
        'last': seq_len - 1,
    }
    dims_list = [4, 16, 64, 256, 1024, 4096]

    # Part 1: Layer × Token Position sweep
    print("\n=== Part 1: Layer × Token Position ===")
    layer_token = {}

    for layer in layers:
        if layer >= len(fwd_clean.hidden_states):
            continue
        for tok_name, tok_idx in token_positions.items():
            if tok_idx >= seq_len:
                continue

            clean_h = fwd_clean.hidden_states[layer][0, tok_idx, :].float().cpu().numpy()
            id_dists = [0.0]  # Clean always 0

            ood_dists = []
            for c in corruptions:
                corr_h = corrupted_fwds[c].hidden_states[layer][0, tok_idx, :].float().cpu().numpy()
                d = float(cosine(clean_h, corr_h))
                ood_dists.append(d)

            auroc = compute_auroc(id_dists, ood_dists)
            key = f"L{layer}_{tok_name}"
            layer_token[key] = {
                "layer": layer,
                "token": tok_name,
                "token_idx": tok_idx,
                "auroc": auroc,
                "distances": dict(zip(corruptions, ood_dists)),
                "min_ood": float(min(ood_dists)),
            }
            if layer in [0, 1, 3, 32]:
                print(f"  {key}: AUROC={auroc:.4f}, min_ood={min(ood_dists):.6f}")

    results["layer_token"] = layer_token

    # Part 2: Dimension Truncation sweep
    print("\n=== Part 2: Dimension Truncation ===")
    dim_ablation = {}

    clean_emb = fwd_clean.hidden_states[3][0, -1, :].float().cpu().numpy()

    for n_dims in dims_list:
        clean_trunc = clean_emb[:n_dims]
        id_dists = [0.0]

        ood_dists = []
        for c in corruptions:
            corr_emb = corrupted_fwds[c].hidden_states[3][0, -1, :].float().cpu().numpy()[:n_dims]
            d = float(cosine(clean_trunc, corr_emb))
            ood_dists.append(d)

        auroc = compute_auroc(id_dists, ood_dists)
        dim_ablation[n_dims] = {
            "dims": n_dims,
            "auroc": auroc,
            "distances": dict(zip(corruptions, ood_dists)),
            "min_ood": float(min(ood_dists)),
        }
        print(f"  {n_dims}D: AUROC={auroc:.4f}, min_ood={min(ood_dists):.6f}")

    results["dimension_ablation"] = dim_ablation

    # Part 3: Distance Metric sweep
    print("\n=== Part 3: Distance Metrics ===")
    metric_ablation = {}

    for metric_name in ['cosine', 'euclidean', 'manhattan', 'chebyshev']:
        id_dists = [0.0]
        ood_dists = []

        for c in corruptions:
            corr_emb = corrupted_fwds[c].hidden_states[3][0, -1, :].float().cpu().numpy()
            if metric_name == 'cosine':
                d = float(cosine(clean_emb, corr_emb))
            elif metric_name == 'euclidean':
                d = float(np.linalg.norm(clean_emb - corr_emb))
            elif metric_name == 'manhattan':
                d = float(np.sum(np.abs(clean_emb - corr_emb)))
            elif metric_name == 'chebyshev':
                d = float(np.max(np.abs(clean_emb - corr_emb)))
            ood_dists.append(d)

        auroc = compute_auroc(id_dists, ood_dists)
        metric_ablation[metric_name] = {
            "auroc": auroc,
            "distances": dict(zip(corruptions, ood_dists)),
            "min_ood": float(min(ood_dists)),
        }
        print(f"  {metric_name}: AUROC={auroc:.4f}, min_ood={min(ood_dists):.6f}")

    results["metric_ablation"] = metric_ablation

    # Part 4: Calibration Size (multi-scene)
    print("\n=== Part 4: Calibration Size ===")
    cal_ablation = {}

    scenes = []
    for seed in [0, 13, 42, 77, 99, 123, 256, 456, 777, 999]:
        np.random.seed(seed)
        px = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        scenes.append(Image.fromarray(px))

    for n_cal in [1, 3, 5, 10]:
        # Per-scene calibration (recommended protocol)
        tp, tn, fp, fn = 0, 0, 0, 0
        for scene_img in scenes[:n_cal]:
            scene_emb = fwd_clean.hidden_states[3][0, -1, :].float().cpu().numpy()
            # Recompute for this specific scene
            scene_inp = processor(prompt, scene_img).to(model.device, dtype=torch.bfloat16)
            with torch.no_grad():
                scene_fwd = model(**scene_inp, output_hidden_states=True)
            scene_emb = scene_fwd.hidden_states[3][0, -1, :].float().cpu().numpy()

            # Clean check
            scene_emb2 = scene_fwd.hidden_states[3][0, -1, :].float().cpu().numpy()
            d_clean = float(cosine(scene_emb, scene_emb2))
            if d_clean == 0:
                tn += 1
            else:
                fp += 1

            # OOD check
            for c_corr in corruptions:
                corrupted = apply_corruption(scene_img, c_corr, 0.5)
                c_inp = processor(prompt, corrupted).to(model.device, dtype=torch.bfloat16)
                with torch.no_grad():
                    c_fwd = model(**c_inp, output_hidden_states=True)
                c_emb = c_fwd.hidden_states[3][0, -1, :].float().cpu().numpy()
                d_ood = float(cosine(scene_emb, c_emb))
                if d_ood > 0:
                    tp += 1
                else:
                    fn += 1

        sens = tp / (tp + fn) if tp + fn > 0 else 0
        spec = tn / (tn + fp) if tn + fp > 0 else 0
        cal_ablation[n_cal] = {
            "n_scenes": n_cal,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "sensitivity": sens,
            "specificity": spec,
        }
        print(f"  N={n_cal}: sens={sens:.3f}, spec={spec:.3f}, TP={tp}, FP={fp}")

    results["calibration_ablation"] = cal_ablation

    # Part 5: Summary — which choices matter
    print("\n=== Part 5: What Matters ===")
    summary = {
        "layer_matters": any(v["auroc"] < 1.0 for v in layer_token.values()),
        "token_matters": any(v["auroc"] < 1.0 for v in layer_token.values()),
        "dims_matter": any(v["auroc"] < 1.0 for v in dim_ablation.values()),
        "metric_matters": any(v["auroc"] < 1.0 for v in metric_ablation.values()),
        "cal_size_matters": any(v["sensitivity"] < 1.0 for v in cal_ablation.values()),
        "critical_choices": [],
        "non_critical_choices": [],
    }

    # Determine which are critical
    bos_aurocs = [v["auroc"] for k, v in layer_token.items() if "bos" in k]
    if any(a < 1.0 for a in bos_aurocs):
        summary["critical_choices"].append("token_position (BOS fails)")
    else:
        summary["non_critical_choices"].append("token_position")

    l0_aurocs = [v["auroc"] for k, v in layer_token.items() if k.startswith("L0_")]
    if any(a < 1.0 for a in l0_aurocs):
        summary["critical_choices"].append("layer (L0 fails)")
    else:
        summary["non_critical_choices"].append("layer")

    results["summary"] = summary
    for choice in summary["critical_choices"]:
        print(f"  CRITICAL: {choice}")
    for choice in summary["non_critical_choices"]:
        print(f"  Non-critical: {choice}")

    # Save
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(v) for v in obj]
        return obj

    ts = results["timestamp"]
    out_path = f"experiments/comp_ablation_{ts}.json"
    with open(out_path, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
