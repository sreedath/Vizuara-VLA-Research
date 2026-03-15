#!/usr/bin/env python3
"""Experiment 305: Detection Pipeline Ablation Study
Systematically ablates each component:
1. Layer choice: which layers work, which don't
2. Token position: last, mean pooling, max pooling
3. Distance metric: cosine, euclidean, manhattan, chebyshev
4. Embedding dimension: full vs truncated
5. Calibration: one-shot vs multi-shot vs random centroid
6. Severity threshold: minimum detectable severity per corruption
"""

import torch
import numpy as np
import json
from datetime import datetime
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor
from scipy.spatial.distance import cosine, euclidean

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
        "experiment": "ablation_study",
        "experiment_number": 305,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    corruptions = ['fog', 'night', 'blur', 'noise']

    # Get full clean hidden states
    print("Getting clean embeddings...")
    inputs = processor(prompt, base_img).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    n_layers = len(fwd.hidden_states)

    # Ablation 1: Token position
    print("\n=== Ablation 1: Token Position ===")
    token_ablation = {}
    seq_len = fwd.hidden_states[3].shape[1]

    for tok_name, tok_fn in [
        ("last", lambda h: h[0, -1, :]),
        ("first", lambda h: h[0, 0, :]),
        ("mean", lambda h: h[0].mean(dim=0)),
        ("max", lambda h: h[0].max(dim=0).values),
        ("mid", lambda h: h[0, seq_len//2, :]),
    ]:
        clean_ref = tok_fn(fwd.hidden_states[3]).float().cpu().numpy()

        id_dists = []
        for _ in range(3):
            inputs = processor(prompt, base_img).to(model.device, dtype=torch.bfloat16)
            with torch.no_grad():
                f = model(**inputs, output_hidden_states=True)
            emb = tok_fn(f.hidden_states[3]).float().cpu().numpy()
            id_dists.append(float(cosine(clean_ref, emb)))

        per_c = {}
        for c in corruptions:
            ood_dists = []
            for sev in [0.3, 0.5, 1.0]:
                corrupted = apply_corruption(base_img, c, sev)
                inputs = processor(prompt, corrupted).to(model.device, dtype=torch.bfloat16)
                with torch.no_grad():
                    f = model(**inputs, output_hidden_states=True)
                emb = tok_fn(f.hidden_states[3]).float().cpu().numpy()
                ood_dists.append(float(cosine(clean_ref, emb)))
            per_c[c] = compute_auroc(id_dists, ood_dists)

        token_ablation[tok_name] = per_c
        mean_auroc = np.mean(list(per_c.values()))
        print(f"  {tok_name}: mean={mean_auroc:.3f}, " + ", ".join(f"{c}={v:.3f}" for c, v in per_c.items()))

    results["token_ablation"] = token_ablation

    # Ablation 2: Embedding dimension truncation
    print("\n=== Ablation 2: Dimension Truncation ===")
    dim_ablation = {}
    clean_full = fwd.hidden_states[3][0, -1, :].float().cpu().numpy()

    for n_dims in [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
        clean_trunc = clean_full[:n_dims]

        id_dists = []
        for _ in range(3):
            inputs = processor(prompt, base_img).to(model.device, dtype=torch.bfloat16)
            with torch.no_grad():
                f = model(**inputs, output_hidden_states=True)
            emb = f.hidden_states[3][0, -1, :].float().cpu().numpy()[:n_dims]
            id_dists.append(float(cosine(clean_trunc, emb)))

        per_c = {}
        for c in corruptions:
            ood_dists = []
            for sev in [0.3, 0.5, 1.0]:
                corrupted = apply_corruption(base_img, c, sev)
                inputs = processor(prompt, corrupted).to(model.device, dtype=torch.bfloat16)
                with torch.no_grad():
                    f = model(**inputs, output_hidden_states=True)
                emb = f.hidden_states[3][0, -1, :].float().cpu().numpy()[:n_dims]
                ood_dists.append(float(cosine(clean_trunc, emb)))
            per_c[c] = compute_auroc(id_dists, ood_dists)

        dim_ablation[n_dims] = per_c
        mean_auroc = np.mean(list(per_c.values()))
        print(f"  {n_dims}D: mean={mean_auroc:.3f}")

    results["dim_ablation"] = dim_ablation

    # Ablation 3: Distance metrics
    print("\n=== Ablation 3: Distance Metrics ===")
    metric_ablation = {}

    for metric_name, metric_fn in [
        ("cosine", lambda a, b: float(cosine(a, b))),
        ("euclidean", lambda a, b: float(np.linalg.norm(a - b))),
        ("manhattan", lambda a, b: float(np.sum(np.abs(a - b)))),
        ("chebyshev", lambda a, b: float(np.max(np.abs(a - b)))),
        ("correlation", lambda a, b: float(1 - np.corrcoef(a, b)[0, 1])),
    ]:
        id_dists = []
        for _ in range(3):
            inputs = processor(prompt, base_img).to(model.device, dtype=torch.bfloat16)
            with torch.no_grad():
                f = model(**inputs, output_hidden_states=True)
            emb = f.hidden_states[3][0, -1, :].float().cpu().numpy()
            id_dists.append(metric_fn(clean_full, emb))

        per_c = {}
        for c in corruptions:
            ood_dists = []
            for sev in [0.3, 0.5, 1.0]:
                corrupted = apply_corruption(base_img, c, sev)
                inputs = processor(prompt, corrupted).to(model.device, dtype=torch.bfloat16)
                with torch.no_grad():
                    f = model(**inputs, output_hidden_states=True)
                emb = f.hidden_states[3][0, -1, :].float().cpu().numpy()
                ood_dists.append(metric_fn(clean_full, emb))
            per_c[c] = compute_auroc(id_dists, ood_dists)

        metric_ablation[metric_name] = per_c
        mean_auroc = np.mean(list(per_c.values()))
        print(f"  {metric_name}: mean={mean_auroc:.3f}")

    results["metric_ablation"] = metric_ablation

    # Ablation 4: Random centroid (no calibration)
    print("\n=== Ablation 4: Random Centroid ===")
    random_ablation = {}

    for trial in range(5):
        rand_centroid = np.random.randn(4096).astype(np.float32)
        rand_centroid = rand_centroid / np.linalg.norm(rand_centroid) * np.linalg.norm(clean_full)

        id_dists = []
        for _ in range(3):
            inputs = processor(prompt, base_img).to(model.device, dtype=torch.bfloat16)
            with torch.no_grad():
                f = model(**inputs, output_hidden_states=True)
            emb = f.hidden_states[3][0, -1, :].float().cpu().numpy()
            id_dists.append(float(cosine(rand_centroid, emb)))

        per_c = {}
        for c in corruptions:
            ood_dists = []
            for sev in [0.3, 0.5, 1.0]:
                corrupted = apply_corruption(base_img, c, sev)
                inputs = processor(prompt, corrupted).to(model.device, dtype=torch.bfloat16)
                with torch.no_grad():
                    f = model(**inputs, output_hidden_states=True)
                emb = f.hidden_states[3][0, -1, :].float().cpu().numpy()
                ood_dists.append(float(cosine(rand_centroid, emb)))
            per_c[c] = compute_auroc(id_dists, ood_dists)

        random_ablation[f"trial_{trial}"] = per_c
        mean_auroc = np.mean(list(per_c.values()))
        print(f"  Trial {trial}: mean={mean_auroc:.3f}")

    results["random_centroid_ablation"] = random_ablation

    # Ablation 5: Minimum detectable severity
    print("\n=== Ablation 5: Minimum Detectable Severity ===")
    min_severity = {}
    for c in corruptions:
        for sev in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]:
            corrupted = apply_corruption(base_img, c, sev)
            inputs = processor(prompt, corrupted).to(model.device, dtype=torch.bfloat16)
            with torch.no_grad():
                f = model(**inputs, output_hidden_states=True)
            emb = f.hidden_states[3][0, -1, :].float().cpu().numpy()
            d = float(cosine(clean_full, emb))
            if c not in min_severity:
                min_severity[c] = []
            min_severity[c].append({"severity": sev, "distance": d, "detected": d > 0})
            if d > 0 and (len(min_severity[c]) == 1 or not min_severity[c][-2]["detected"]):
                print(f"  {c}: first detection at sev={sev}, d={d:.8f}")

    results["min_severity"] = min_severity

    # Save
    ts = results["timestamp"]
    out_path = f"experiments/ablation_{ts}.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
