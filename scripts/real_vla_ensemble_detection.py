#!/usr/bin/env python3
"""Experiment 308: Ensemble & Voting Detection Strategies
Tests whether combining multiple detection signals improves robustness:
1. Multi-layer voting (L1, L3, L7, L15, L31)
2. Multi-metric ensemble (cosine + euclidean + norm + correlation)
3. Multi-token ensemble (different sequence positions)
4. Weighted vs unweighted combination
5. Adversarial robustness of ensemble vs single detector
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
    if n_id == 0 or n_ood == 0: return 0.5
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
        "experiment": "ensemble_detection",
        "experiment_number": 308,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    corruptions = ['fog', 'night', 'blur', 'noise']
    target_layers = [1, 3, 7, 15, 31]

    # Get clean references at all layers
    print("Getting clean references...")
    inputs = processor(prompt, base_img).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    n_layers = len(fwd.hidden_states)
    seq_len = fwd.hidden_states[3].shape[1]

    clean_refs = {}
    for li in target_layers:
        if li < n_layers:
            clean_refs[li] = fwd.hidden_states[li][0, -1, :].float().cpu().numpy()

    # Token positions for multi-token ensemble
    token_positions = [-1, seq_len//4, seq_len//2, 3*seq_len//4]
    clean_token_refs = {}
    for pos in token_positions:
        clean_token_refs[pos] = fwd.hidden_states[3][0, pos, :].float().cpu().numpy()

    # Part 1: Multi-layer voting
    print("\n=== Part 1: Multi-Layer Voting ===")
    layer_voting = {}

    # Collect individual layer scores
    id_scores_by_layer = {li: [] for li in target_layers if li < n_layers}
    for _ in range(5):
        inputs = processor(prompt, base_img).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            f = model(**inputs, output_hidden_states=True)
        for li in target_layers:
            if li < n_layers:
                emb = f.hidden_states[li][0, -1, :].float().cpu().numpy()
                id_scores_by_layer[li].append(float(cosine(clean_refs[li], emb)))

    for c in corruptions:
        print(f"  {c}...")
        ood_scores_by_layer = {li: [] for li in target_layers if li < n_layers}
        for sev in [0.3, 0.5, 1.0]:
            corrupted = apply_corruption(base_img, c, sev)
            inputs = processor(prompt, corrupted).to(model.device, dtype=torch.bfloat16)
            with torch.no_grad():
                f = model(**inputs, output_hidden_states=True)
            for li in target_layers:
                if li < n_layers:
                    emb = f.hidden_states[li][0, -1, :].float().cpu().numpy()
                    ood_scores_by_layer[li].append(float(cosine(clean_refs[li], emb)))

        # Individual layer AUROCs
        individual = {}
        for li in target_layers:
            if li < n_layers:
                individual[f"L{li}"] = compute_auroc(id_scores_by_layer[li], ood_scores_by_layer[li])

        # Ensemble: mean of layer distances
        id_mean = [np.mean([id_scores_by_layer[li][i] for li in target_layers if li < n_layers])
                   for i in range(len(id_scores_by_layer[target_layers[0]]))]
        ood_mean = [np.mean([ood_scores_by_layer[li][i] for li in target_layers if li < n_layers])
                    for i in range(len(ood_scores_by_layer[target_layers[0]]))]
        mean_auroc = compute_auroc(id_mean, ood_mean)

        # Ensemble: max of layer distances
        id_max = [max(id_scores_by_layer[li][i] for li in target_layers if li < n_layers)
                  for i in range(len(id_scores_by_layer[target_layers[0]]))]
        ood_max = [max(ood_scores_by_layer[li][i] for li in target_layers if li < n_layers)
                   for i in range(len(ood_scores_by_layer[target_layers[0]]))]
        max_auroc = compute_auroc(id_max, ood_max)

        # Voting: majority vote
        thresholds = {}
        for li in target_layers:
            if li < n_layers:
                thresholds[li] = max(id_scores_by_layer[li]) * 1.5 if max(id_scores_by_layer[li]) > 0 else 1e-6

        id_votes = [sum(1 for li in target_layers if li < n_layers
                        and id_scores_by_layer[li][i] > thresholds[li])
                    for i in range(len(id_scores_by_layer[target_layers[0]]))]
        ood_votes = [sum(1 for li in target_layers if li < n_layers
                         and ood_scores_by_layer[li][i] > thresholds[li])
                     for i in range(len(ood_scores_by_layer[target_layers[0]]))]
        vote_auroc = compute_auroc(id_votes, ood_votes)

        layer_voting[c] = {
            "individual": individual,
            "mean_ensemble": mean_auroc,
            "max_ensemble": max_auroc,
            "vote_ensemble": vote_auroc,
        }
        print(f"    individual: {individual}")
        print(f"    mean={mean_auroc:.3f}, max={max_auroc:.3f}, vote={vote_auroc:.3f}")

    results["layer_voting"] = layer_voting

    # Part 2: Multi-metric ensemble
    print("\n=== Part 2: Multi-Metric Ensemble ===")
    metric_ensemble = {}
    clean_l3 = clean_refs[3]

    metric_fns = {
        "cosine": lambda a, b: float(cosine(a, b)),
        "euclidean": lambda a, b: float(np.linalg.norm(a - b)),
        "norm_diff": lambda a, b: abs(float(np.linalg.norm(a)) - float(np.linalg.norm(b))),
        "correlation": lambda a, b: float(1 - np.corrcoef(a, b)[0, 1]),
    }

    id_by_metric = {m: [] for m in metric_fns}
    for _ in range(5):
        inputs = processor(prompt, base_img).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            f = model(**inputs, output_hidden_states=True)
        emb = f.hidden_states[3][0, -1, :].float().cpu().numpy()
        for m, fn in metric_fns.items():
            id_by_metric[m].append(fn(clean_l3, emb))

    for c in corruptions:
        print(f"  {c}...")
        ood_by_metric = {m: [] for m in metric_fns}
        for sev in [0.3, 0.5, 1.0]:
            corrupted = apply_corruption(base_img, c, sev)
            inputs = processor(prompt, corrupted).to(model.device, dtype=torch.bfloat16)
            with torch.no_grad():
                f = model(**inputs, output_hidden_states=True)
            emb = f.hidden_states[3][0, -1, :].float().cpu().numpy()
            for m, fn in metric_fns.items():
                ood_by_metric[m].append(fn(clean_l3, emb))

        individual = {m: compute_auroc(id_by_metric[m], ood_by_metric[m]) for m in metric_fns}

        # Normalized ensemble
        id_ensemble = []
        for i in range(len(id_by_metric["cosine"])):
            scores = []
            for m in metric_fns:
                max_val = max(max(ood_by_metric[m]), max(id_by_metric[m])) + 1e-30
                scores.append(id_by_metric[m][i] / max_val)
            id_ensemble.append(np.mean(scores))

        ood_ensemble = []
        for i in range(len(ood_by_metric["cosine"])):
            scores = []
            for m in metric_fns:
                max_val = max(max(ood_by_metric[m]), max(id_by_metric[m])) + 1e-30
                scores.append(ood_by_metric[m][i] / max_val)
            ood_ensemble.append(np.mean(scores))

        ensemble_auroc = compute_auroc(id_ensemble, ood_ensemble)

        metric_ensemble[c] = {
            "individual": individual,
            "ensemble": ensemble_auroc,
        }
        print(f"    individual: {individual}, ensemble={ensemble_auroc:.3f}")

    results["metric_ensemble"] = metric_ensemble

    # Part 3: Multi-token position ensemble
    print("\n=== Part 3: Multi-Token Ensemble ===")
    token_ensemble = {}

    id_by_token = {pos: [] for pos in token_positions}
    for _ in range(5):
        inputs = processor(prompt, base_img).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            f = model(**inputs, output_hidden_states=True)
        for pos in token_positions:
            emb = f.hidden_states[3][0, pos, :].float().cpu().numpy()
            id_by_token[pos].append(float(cosine(clean_token_refs[pos], emb)))

    for c in corruptions:
        print(f"  {c}...")
        ood_by_token = {pos: [] for pos in token_positions}
        for sev in [0.3, 0.5, 1.0]:
            corrupted = apply_corruption(base_img, c, sev)
            inputs = processor(prompt, corrupted).to(model.device, dtype=torch.bfloat16)
            with torch.no_grad():
                f = model(**inputs, output_hidden_states=True)
            for pos in token_positions:
                emb = f.hidden_states[3][0, pos, :].float().cpu().numpy()
                ood_by_token[pos].append(float(cosine(clean_token_refs[pos], emb)))

        individual = {}
        for pos in token_positions:
            pos_name = f"pos{pos}" if pos >= 0 else "last"
            individual[pos_name] = compute_auroc(id_by_token[pos], ood_by_token[pos])

        id_mean = [np.mean([id_by_token[pos][i] for pos in token_positions])
                   for i in range(len(id_by_token[token_positions[0]]))]
        ood_mean = [np.mean([ood_by_token[pos][i] for pos in token_positions])
                    for i in range(len(ood_by_token[token_positions[0]]))]
        mean_auroc = compute_auroc(id_mean, ood_mean)

        token_ensemble[c] = {
            "individual": individual,
            "ensemble": mean_auroc,
        }
        print(f"    individual: {individual}, ensemble={mean_auroc:.3f}")

    results["token_ensemble"] = token_ensemble

    # Part 4: Full ensemble (layers × metrics)
    print("\n=== Part 4: Full Ensemble ===")
    full_ensemble = {}

    for c in corruptions:
        print(f"  {c}...")
        id_all = []
        ood_all = []

        for _ in range(5):
            inputs = processor(prompt, base_img).to(model.device, dtype=torch.bfloat16)
            with torch.no_grad():
                f = model(**inputs, output_hidden_states=True)
            scores = []
            for li in [3, 15, 31]:
                if li < n_layers:
                    emb = f.hidden_states[li][0, -1, :].float().cpu().numpy()
                    scores.append(float(cosine(clean_refs[li], emb)))
                    scores.append(float(np.linalg.norm(emb - clean_refs[li])))
            id_all.append(np.mean(scores))

        for sev in [0.3, 0.5, 1.0]:
            corrupted = apply_corruption(base_img, c, sev)
            inputs = processor(prompt, corrupted).to(model.device, dtype=torch.bfloat16)
            with torch.no_grad():
                f = model(**inputs, output_hidden_states=True)
            scores = []
            for li in [3, 15, 31]:
                if li < n_layers:
                    emb = f.hidden_states[li][0, -1, :].float().cpu().numpy()
                    scores.append(float(cosine(clean_refs[li], emb)))
                    scores.append(float(np.linalg.norm(emb - clean_refs[li])))
            ood_all.append(np.mean(scores))

        full_auroc = compute_auroc(id_all, ood_all)
        full_ensemble[c] = full_auroc
        print(f"    AUROC={full_auroc:.3f}")

    results["full_ensemble"] = full_ensemble

    # Part 5: Separation margins
    print("\n=== Part 5: Separation Margins ===")
    margins = {}

    for c in corruptions:
        corrupted = apply_corruption(base_img, c, 1.0)

        inputs_clean = processor(prompt, base_img).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            f_clean = model(**inputs_clean, output_hidden_states=True)
        clean_d = float(cosine(clean_refs[3], f_clean.hidden_states[3][0, -1, :].float().cpu().numpy()))

        inputs_corr = processor(prompt, corrupted).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            f_corr = model(**inputs_corr, output_hidden_states=True)
        corr_d = float(cosine(clean_refs[3], f_corr.hidden_states[3][0, -1, :].float().cpu().numpy()))

        single_margin = corr_d - clean_d

        clean_scores = []
        corr_scores = []
        for li in target_layers:
            if li < n_layers:
                c_emb = f_clean.hidden_states[li][0, -1, :].float().cpu().numpy()
                clean_scores.append(float(cosine(clean_refs[li], c_emb)))
                cr_emb = f_corr.hidden_states[li][0, -1, :].float().cpu().numpy()
                corr_scores.append(float(cosine(clean_refs[li], cr_emb)))

        ensemble_margin = np.mean(corr_scores) - np.mean(clean_scores)

        margins[c] = {
            "single_id": clean_d,
            "single_ood": corr_d,
            "single_margin": single_margin,
            "ensemble_id": float(np.mean(clean_scores)),
            "ensemble_ood": float(np.mean(corr_scores)),
            "ensemble_margin": float(ensemble_margin),
            "margin_ratio": float(ensemble_margin / (single_margin + 1e-30)) if single_margin != 0 else 0,
        }
        print(f"  {c}: single={single_margin:.6f}, ensemble={ensemble_margin:.6f}, "
              f"ratio={margins[c]['margin_ratio']:.2f}")

    results["separation_margins"] = margins

    # Save
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(v) for v in obj]
        return obj

    ts = results["timestamp"]
    out_path = f"experiments/ensemble_{ts}.json"
    with open(out_path, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
