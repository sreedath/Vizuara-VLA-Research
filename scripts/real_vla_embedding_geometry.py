#!/usr/bin/env python3
"""Experiment 383: Embedding Space Geometry Under Corruption

What is the geometric structure of the embedding manifold?
1. Principal angles between clean and corruption subspaces
2. Corruption-specific direction vectors (mean shift directions)
3. Orthogonality of corruption shift directions
4. Projection analysis: how much of corruption shift is shared vs unique
5. Residual analysis: what remains after removing the main corruption direction
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

def cosine_sim(a, b):
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return dot / (na * nb)

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
    seeds = list(range(0, 1500, 100))[:15]
    images = {}
    clean_embs = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        px = rng.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        images[seed] = Image.fromarray(px)
        clean_embs[seed] = extract_hidden(model, processor, images[seed], prompt)

    centroid = np.mean(list(clean_embs.values()), axis=0)

    # Extract corruption embeddings
    print("Extracting corruption embeddings...")
    corrupt_embs = {ct: {} for ct in ctypes}
    for ct in ctypes:
        for seed in seeds[:10]:
            corrupt_embs[ct][seed] = extract_hidden(
                model, processor, apply_corruption(images[seed], ct, 0.5), prompt)

    # ========== 1. Corruption Shift Directions ==========
    print("\n=== Corruption Shift Directions ===")

    shift_dirs = {}
    for ct in ctypes:
        shifts = [corrupt_embs[ct][s] - clean_embs[s] for s in seeds[:10]]
        mean_shift = np.mean(shifts, axis=0)
        mean_shift_norm = mean_shift / max(np.linalg.norm(mean_shift), 1e-10)
        shift_dirs[ct] = mean_shift_norm

        # How consistent are individual shifts with mean direction?
        consistencies = [cosine_sim(s, mean_shift) for s in shifts]
        results[f'{ct}_shift_consistency'] = {
            'mean': float(np.mean(consistencies)),
            'min': float(np.min(consistencies)),
            'max': float(np.max(consistencies)),
            'shift_magnitude': float(np.linalg.norm(mean_shift)),
        }
        print(f"  {ct}: consistency={np.mean(consistencies):.4f}, "
              f"magnitude={np.linalg.norm(mean_shift):.6f}")

    # ========== 2. Orthogonality of Shift Directions ==========
    print("\n=== Shift Direction Orthogonality ===")

    ortho = {}
    for i, ct1 in enumerate(ctypes):
        for j, ct2 in enumerate(ctypes):
            if j > i:
                sim = cosine_sim(shift_dirs[ct1], shift_dirs[ct2])
                key = f"{ct1}_vs_{ct2}"
                ortho[key] = float(sim)
                print(f"  {ct1} vs {ct2}: cos_sim={sim:.4f}")

    results['shift_orthogonality'] = ortho

    # ========== 3. Principal Component Analysis ==========
    print("\n=== PCA of Shift Space ===")

    all_shifts = []
    shift_labels = []
    for ct in ctypes:
        for seed in seeds[:10]:
            shift = corrupt_embs[ct][seed] - clean_embs[seed]
            all_shifts.append(shift)
            shift_labels.append(ct)

    all_shifts = np.array(all_shifts)
    # Center
    all_shifts_centered = all_shifts - np.mean(all_shifts, axis=0)

    # SVD
    U, S, Vt = np.linalg.svd(all_shifts_centered, full_matrices=False)
    total_var = np.sum(S**2)
    explained = (S**2) / max(total_var, 1e-10)
    cumulative = np.cumsum(explained)

    pca_results = {
        'explained_variance': explained[:20].tolist(),
        'cumulative_variance': cumulative[:20].tolist(),
        'n_components_90pct': int(np.searchsorted(cumulative, 0.9) + 1),
        'n_components_95pct': int(np.searchsorted(cumulative, 0.95) + 1),
        'n_components_99pct': int(np.searchsorted(cumulative, 0.99) + 1),
        'top1_variance': float(explained[0]),
        'top3_variance': float(cumulative[2]),
    }
    results['pca'] = pca_results
    print(f"  90% variance: {pca_results['n_components_90pct']} components")
    print(f"  95% variance: {pca_results['n_components_95pct']} components")
    print(f"  Top-1: {explained[0]:.4f}, Top-3: {cumulative[2]:.4f}")

    # ========== 4. Projection Analysis ==========
    print("\n=== Projection Analysis ===")

    # Project each corruption's shifts onto other corruption directions
    projection = {}
    for ct in ctypes:
        shifts = [corrupt_embs[ct][s] - clean_embs[s] for s in seeds[:10]]
        mean_shift = np.mean(shifts, axis=0)
        shift_norm = np.linalg.norm(mean_shift)

        proj = {}
        for ct2 in ctypes:
            # How much of ct's shift lies along ct2's direction?
            proj_amount = np.dot(mean_shift, shift_dirs[ct2])
            proj[ct2] = {
                'projection': float(proj_amount),
                'fraction': float(abs(proj_amount) / max(shift_norm, 1e-10)),
            }

        projection[ct] = proj

    results['projection'] = projection

    for ct in ctypes:
        self_frac = projection[ct][ct]['fraction']
        other_fracs = [projection[ct][ct2]['fraction'] for ct2 in ctypes if ct2 != ct]
        print(f"  {ct}: self={self_frac:.4f}, others={[f'{f:.4f}' for f in other_fracs]}")

    # ========== 5. Residual After Main Direction ==========
    print("\n=== Residual Analysis ===")

    residual = {}
    for ct in ctypes:
        shifts = [corrupt_embs[ct][s] - clean_embs[s] for s in seeds[:10]]
        mean_shift = np.mean(shifts, axis=0)

        # Remove projection onto own direction
        proj_on_self = np.dot(mean_shift, shift_dirs[ct]) * shift_dirs[ct]
        residual_vec = mean_shift - proj_on_self
        residual_mag = np.linalg.norm(residual_vec) / max(np.linalg.norm(mean_shift), 1e-10)

        # Remove projection onto ALL corruption directions
        remaining = mean_shift.copy()
        for ct2 in ctypes:
            proj = np.dot(remaining, shift_dirs[ct2]) * shift_dirs[ct2]
            remaining = remaining - proj

        unexplained = np.linalg.norm(remaining) / max(np.linalg.norm(mean_shift), 1e-10)

        residual[ct] = {
            'residual_after_self': float(residual_mag),
            'residual_after_all': float(unexplained),
        }
        print(f"  {ct}: after_self={residual_mag:.4f}, after_all={unexplained:.4f}")

    results['residual'] = residual

    # ========== 6. Severity Scaling of Directions ==========
    print("\n=== Severity Scaling ===")

    severity_scaling = {}
    for ct in ctypes:
        sevs = [0.1, 0.3, 0.5, 0.7, 1.0]
        dir_sims = []
        magnitudes = []
        for sev in sevs:
            shifts = []
            for seed in seeds[:5]:
                emb = extract_hidden(model, processor,
                    apply_corruption(images[seed], ct, sev), prompt)
                shifts.append(emb - clean_embs[seed])
            mean_s = np.mean(shifts, axis=0)
            sim = cosine_sim(mean_s, shift_dirs[ct])
            dir_sims.append(float(sim))
            magnitudes.append(float(np.linalg.norm(mean_s)))

        severity_scaling[ct] = {
            'severities': sevs,
            'direction_similarity': dir_sims,
            'magnitudes': magnitudes,
            'direction_stable': bool(min(dir_sims) > 0.9),
        }
        print(f"  {ct}: dir_sims={[f'{s:.3f}' for s in dir_sims]}, "
              f"stable={min(dir_sims) > 0.9}")

    results['severity_scaling'] = severity_scaling

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/embedding_geometry_{ts}.json"
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
