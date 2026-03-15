#!/usr/bin/env python3
"""Experiment 299: Embedding Norm Decomposition
Decomposes embedding changes into parallel and orthogonal components
relative to the clean embedding and corruption direction vectors:
1. Parallel vs orthogonal decomposition of corruption shift
2. Layer-wise norm change profiles
3. Residual stream analysis (how much each layer adds)
4. Norm-based detection (L2 norm alone vs cosine distance)
5. Per-dimension contribution to distance
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
        "experiment": "norm_decomposition",
        "experiment_number": 299,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    corruptions = ['fog', 'night', 'blur', 'noise']
    target_layers = [0, 1, 2, 3, 5, 7, 11, 15, 23, 31, 32]

    # Get clean embeddings at all layers
    print("Getting clean embeddings...")
    inputs = processor(prompt, base_img).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)

    n_layers = len(fwd.hidden_states)
    clean_embs = {}
    clean_norms = {}
    for li in target_layers:
        if li < n_layers:
            emb = fwd.hidden_states[li][0, -1, :].float().cpu().numpy()
            clean_embs[li] = emb
            clean_norms[li] = float(np.linalg.norm(emb))

    results["clean_norms"] = {f"L{li}": clean_norms[li] for li in clean_norms}

    # Part 1: Parallel vs orthogonal decomposition
    print("\n=== Part 1: Parallel/Orthogonal Decomposition ===")
    decomposition = {}

    for c in corruptions:
        print(f"  {c}...")
        decomposition[c] = {}
        corrupted = apply_corruption(base_img, c, 1.0)
        inputs = processor(prompt, corrupted).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            fwd_c = model(**inputs, output_hidden_states=True)

        for li in target_layers:
            if li < n_layers:
                corr_emb = fwd_c.hidden_states[li][0, -1, :].float().cpu().numpy()
                clean = clean_embs[li]

                # Difference vector
                diff = corr_emb - clean
                diff_norm = float(np.linalg.norm(diff))

                # Project diff onto clean direction
                clean_unit = clean / (np.linalg.norm(clean) + 1e-30)
                parallel = float(np.dot(diff, clean_unit))
                parallel_vec = parallel * clean_unit
                orthogonal_vec = diff - parallel_vec
                orthogonal = float(np.linalg.norm(orthogonal_vec))

                # Norm change
                corr_norm = float(np.linalg.norm(corr_emb))
                norm_change = corr_norm - clean_norms[li]
                norm_pct_change = norm_change / clean_norms[li] * 100

                # Cosine distance
                cos_d = float(cosine(clean, corr_emb))

                decomposition[c][f"L{li}"] = {
                    "diff_norm": diff_norm,
                    "parallel_component": parallel,
                    "orthogonal_component": orthogonal,
                    "parallel_pct": abs(parallel) / (diff_norm + 1e-30) * 100,
                    "orthogonal_pct": orthogonal / (diff_norm + 1e-30) * 100,
                    "norm_change": norm_change,
                    "norm_pct_change": norm_pct_change,
                    "cosine_distance": cos_d,
                    "clean_norm": clean_norms[li],
                    "corrupted_norm": corr_norm
                }

        # Report L3
        l3 = decomposition[c].get("L3", {})
        print(f"    L3: diff={l3.get('diff_norm', 0):.4f}, "
              f"parallel={l3.get('parallel_component', 0):.4f} ({l3.get('parallel_pct', 0):.1f}%), "
              f"orthogonal={l3.get('orthogonal_component', 0):.4f} ({l3.get('orthogonal_pct', 0):.1f}%)")

    results["decomposition"] = decomposition

    # Part 2: Residual stream analysis
    print("\n=== Part 2: Residual Stream ===")
    residual = {}

    for c in corruptions:
        print(f"  {c}...")
        corrupted = apply_corruption(base_img, c, 1.0)
        inputs = processor(prompt, corrupted).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            fwd_c = model(**inputs, output_hidden_states=True)

        residual[c] = []
        prev_diff = None
        for li in range(min(n_layers, 33)):
            corr_emb = fwd_c.hidden_states[li][0, -1, :].float().cpu().numpy()
            clean = fwd.hidden_states[li][0, -1, :].float().cpu().numpy()
            diff = corr_emb - clean
            diff_norm = float(np.linalg.norm(diff))

            layer_contrib = 0.0
            if prev_diff is not None:
                # How much this layer changed the corruption signal
                delta = diff - prev_diff
                layer_contrib = float(np.linalg.norm(delta))

            residual[c].append({
                "layer": li,
                "total_shift": diff_norm,
                "layer_contribution": layer_contrib
            })
            prev_diff = diff

        # Top contributing layers
        contribs = [(r["layer_contribution"], r["layer"]) for r in residual[c] if r["layer"] > 0]
        contribs.sort(reverse=True)
        print(f"    Top 3 contributing layers: {[(f'L{l}', f'{v:.4f}') for v, l in contribs[:3]]}")

    results["residual_stream"] = residual

    # Part 3: Norm-based detection comparison
    print("\n=== Part 3: Norm-Based vs Cosine Detection ===")
    norm_detection = {}

    # ID norms
    id_norms = {li: [] for li in [3, 7, 15, 31] if li < n_layers}
    id_cosine = {li: [] for li in [3, 7, 15, 31] if li < n_layers}
    for _ in range(3):
        inputs = processor(prompt, base_img).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            fwd_clean = model(**inputs, output_hidden_states=True)
        for li in [3, 7, 15, 31]:
            if li < n_layers:
                emb = fwd_clean.hidden_states[li][0, -1, :].float().cpu().numpy()
                id_norms[li].append(float(np.linalg.norm(emb)))
                id_cosine[li].append(float(cosine(clean_embs[li], emb)))

    for c in corruptions:
        norm_detection[c] = {}
        ood_norms = {li: [] for li in [3, 7, 15, 31] if li < n_layers}
        ood_cosine = {li: [] for li in [3, 7, 15, 31] if li < n_layers}

        for sev in [0.3, 0.5, 1.0]:
            corrupted = apply_corruption(base_img, c, sev)
            inputs = processor(prompt, corrupted).to(model.device, dtype=torch.bfloat16)
            with torch.no_grad():
                fwd_c = model(**inputs, output_hidden_states=True)
            for li in [3, 7, 15, 31]:
                if li < n_layers:
                    emb = fwd_c.hidden_states[li][0, -1, :].float().cpu().numpy()
                    ood_norms[li].append(float(np.linalg.norm(emb)))
                    ood_cosine[li].append(float(cosine(clean_embs[li], emb)))

        for li in [3, 7, 15, 31]:
            if li < n_layers:
                # AUROC using absolute norm as score
                id_score = [abs(n - clean_norms[li]) for n in id_norms[li]]
                ood_score = [abs(n - clean_norms[li]) for n in ood_norms[li]]
                auroc_norm = compute_auroc(id_score, ood_score)
                auroc_cosine = compute_auroc(id_cosine[li], ood_cosine[li])

                norm_detection[c][f"L{li}"] = {
                    "auroc_norm": auroc_norm,
                    "auroc_cosine": auroc_cosine,
                    "id_mean_norm": float(np.mean(id_norms[li])),
                    "ood_mean_norm": float(np.mean(ood_norms[li])),
                    "norm_diff": float(abs(np.mean(ood_norms[li]) - np.mean(id_norms[li])))
                }

        print(f"  {c}: norm_AUROC@L3={norm_detection[c].get('L3', {}).get('auroc_norm', 'N/A'):.3f}, "
              f"cosine_AUROC@L3={norm_detection[c].get('L3', {}).get('auroc_cosine', 'N/A'):.3f}")

    results["norm_detection"] = norm_detection

    # Part 4: Per-dimension contribution to cosine distance
    print("\n=== Part 4: Per-Dimension Contribution ===")
    dim_contribution = {}

    for c in corruptions:
        corrupted = apply_corruption(base_img, c, 1.0)
        inputs = processor(prompt, corrupted).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            fwd_c = model(**inputs, output_hidden_states=True)
        emb = fwd_c.hidden_states[3][0, -1, :].float().cpu().numpy()
        clean = clean_embs[3]

        # Per-dimension squared difference
        sq_diff = (emb - clean) ** 2
        total = sq_diff.sum()

        # Top dimensions
        top_indices = np.argsort(sq_diff)[-20:][::-1]
        cumulative = np.cumsum(np.sort(sq_diff)[::-1]) / total

        dim_contribution[c] = {
            "top_20_dims": top_indices.tolist(),
            "top_20_contributions": sq_diff[top_indices].tolist(),
            "top_20_pct": float(sq_diff[top_indices].sum() / total * 100),
            "dims_for_50pct": int(np.searchsorted(cumulative, 0.50)) + 1,
            "dims_for_90pct": int(np.searchsorted(cumulative, 0.90)) + 1,
            "dims_for_99pct": int(np.searchsorted(cumulative, 0.99)) + 1,
            "gini_coefficient": float(1 - 2 * np.trapezoid(
                np.cumsum(np.sort(sq_diff)) / total,
                np.linspace(0, 1, len(sq_diff))
            ))
        }
        print(f"  {c}: top20={dim_contribution[c]['top_20_pct']:.1f}%, "
              f"50%@{dim_contribution[c]['dims_for_50pct']}D, "
              f"90%@{dim_contribution[c]['dims_for_90pct']}D, "
              f"Gini={dim_contribution[c]['gini_coefficient']:.3f}")

    results["dim_contribution"] = dim_contribution

    # Save
    ts = results["timestamp"]
    out_path = f"experiments/norm_decomp_{ts}.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
