#!/usr/bin/env python3
"""Experiment 406: Embedding Space Geometry Deep Dive

Detailed analysis of the geometric structure of the embedding space,
including curvature, manifold properties, and corruption trajectories.

Tests:
1. Corruption trajectories in embedding space (interpolation)
2. Angular separation between corruption directions
3. Embedding norm analysis (clean vs corrupt)
4. Projection onto principal corruption subspace
5. Nearest-neighbor structure in high dimensions
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

def cosine_dist(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    return 1.0 - np.dot(a, b) / (na * nb)

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

def main():
    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    corruptions = ['fog', 'night', 'noise', 'blur']

    scenes = []
    for seed in [42, 123, 456, 789, 999]:
        scenes.append(Image.fromarray(
            np.random.RandomState(seed).randint(0, 255, (224, 224, 3), dtype=np.uint8)))

    # Collect embeddings at multiple severities
    print("Extracting embeddings...")
    severities = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    embeddings = {}  # (scene, corruption, severity) -> embedding

    for si, scene in enumerate(scenes):
        print(f"  Scene {si+1}/5")
        # Clean
        emb = extract_hidden(model, processor, scene, prompt)
        embeddings[(si, 'clean', 0.0)] = emb

        for c in corruptions:
            for sev in severities[1:]:  # skip 0.0, already have clean
                corrupted = apply_corruption(scene, c, sev)
                emb = extract_hidden(model, processor, corrupted, prompt)
                embeddings[(si, c, sev)] = emb

    results = {}

    # === Test 1: Corruption trajectories ===
    print("\n=== Corruption Trajectories ===")
    trajectories = {}
    for c in corruptions:
        traj_data = {"by_scene": {}}
        all_dists = {str(s): [] for s in severities}

        for si in range(len(scenes)):
            clean = embeddings[(si, 'clean', 0.0)]
            scene_dists = []
            for sev in severities:
                if sev == 0.0:
                    d = 0.0
                else:
                    d = cosine_dist(clean, embeddings[(si, c, sev)])
                scene_dists.append(d)
                all_dists[str(sev)].append(d)

            traj_data["by_scene"][str(si)] = [float(d) for d in scene_dists]

        # Mean trajectory
        mean_traj = [np.mean(all_dists[str(s)]) for s in severities]
        traj_data["mean_trajectory"] = [float(d) for d in mean_traj]

        # Linearity: R² of distance vs severity
        if len(mean_traj) > 2:
            x = np.array(severities)
            y = np.array(mean_traj)
            slope, intercept = np.polyfit(x, y, 1)
            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            traj_data["linearity_r2"] = float(r_squared)
        else:
            traj_data["linearity_r2"] = 0.0

        trajectories[c] = traj_data
        traj_str = " → ".join(f"{d:.6f}" for d in mean_traj[::2])
        print(f"  {c}: {traj_str} (R²={traj_data['linearity_r2']:.4f})")

    results["trajectories"] = trajectories

    # === Test 2: Angular separation between corruption directions ===
    print("\n=== Corruption Direction Angles ===")
    angles = {}
    for si in range(len(scenes)):
        clean = embeddings[(si, 'clean', 0.0)]
        # Direction vectors (corruption - clean)
        directions = {}
        for c in corruptions:
            corrupt = embeddings[(si, c, 1.0)]
            direction = corrupt - clean
            norm = np.linalg.norm(direction)
            if norm > 1e-12:
                directions[c] = direction / norm
            else:
                directions[c] = direction

        # Pairwise angles
        for i, c1 in enumerate(corruptions):
            for j, c2 in enumerate(corruptions):
                if i >= j:
                    continue
                cos_angle = np.dot(directions[c1], directions[c2])
                angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
                key = f"{c1}_vs_{c2}"
                if key not in angles:
                    angles[key] = []
                angles[key].append(float(angle_deg))

    angle_means = {k: float(np.mean(v)) for k, v in angles.items()}
    results["corruption_angles"] = angle_means
    for k, v in angle_means.items():
        print(f"  {k}: {v:.1f}°")

    # === Test 3: Embedding norm analysis ===
    print("\n=== Embedding Norm Analysis ===")
    norm_data = {}
    for condition in ['clean'] + corruptions:
        norms = []
        for si in range(len(scenes)):
            if condition == 'clean':
                emb = embeddings[(si, 'clean', 0.0)]
            else:
                emb = embeddings[(si, condition, 1.0)]
            norms.append(float(np.linalg.norm(emb)))

        norm_data[condition] = {
            "mean": float(np.mean(norms)),
            "std": float(np.std(norms)),
            "min": float(min(norms)),
            "max": float(max(norms))
        }
        print(f"  {condition}: mean={np.mean(norms):.2f}, std={np.std(norms):.4f}")

    results["norms"] = norm_data

    # === Test 4: Principal corruption subspace ===
    print("\n=== Principal Corruption Subspace ===")
    # Compute displacement vectors for all corruptions
    displacements = []
    for c in corruptions:
        for si in range(len(scenes)):
            clean = embeddings[(si, 'clean', 0.0)]
            corrupt = embeddings[(si, c, 1.0)]
            displacements.append(corrupt - clean)

    displacements = np.array(displacements)
    # PCA on displacements
    centered = displacements - displacements.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    total_var = eigvals.sum()
    cum_var = np.cumsum(eigvals) / total_var

    subspace_data = {
        "top_eigenvalues": [float(e) for e in eigvals[:10]],
        "variance_explained_1d": float(cum_var[0]),
        "variance_explained_2d": float(cum_var[1]),
        "variance_explained_3d": float(cum_var[2]),
        "variance_explained_5d": float(cum_var[4]) if len(cum_var) > 4 else 1.0,
    }
    results["corruption_subspace"] = subspace_data
    print(f"  1D: {cum_var[0]*100:.1f}%, 2D: {cum_var[1]*100:.1f}%, 3D: {cum_var[2]*100:.1f}%")

    # === Test 5: Trajectory curvature ===
    print("\n=== Trajectory Curvature ===")
    curvature_data = {}
    for c in corruptions:
        for si in range(len(scenes)):
            clean = embeddings[(si, 'clean', 0.0)]
            full_corrupt = embeddings[(si, c, 1.0)]

            # Expected positions along straight line
            actual_dists = []
            expected_dists = []
            deviations = []

            full_direction = full_corrupt - clean
            full_norm = np.linalg.norm(full_direction)

            for sev in severities[1:]:
                actual = embeddings[(si, c, sev)]
                expected = clean + sev * full_direction

                actual_dist = np.linalg.norm(actual - clean)
                expected_dist = sev * full_norm
                deviation = np.linalg.norm(actual - expected)

                actual_dists.append(actual_dist)
                expected_dists.append(expected_dist)
                deviations.append(deviation)

            mean_dev = np.mean(deviations)
            key = f"scene{si}"
            if c not in curvature_data:
                curvature_data[c] = {}
            curvature_data[c][key] = {
                "mean_deviation": float(mean_dev),
                "relative_deviation": float(mean_dev / max(full_norm, 1e-12))
            }

        mean_rel_dev = np.mean([curvature_data[c][f"scene{si}"]["relative_deviation"]
                                for si in range(len(scenes))])
        curvature_data[c]["mean_relative_deviation"] = float(mean_rel_dev)
        print(f"  {c}: mean relative deviation = {mean_rel_dev:.4f} "
              f"({'LINEAR' if mean_rel_dev < 0.1 else 'CURVED'})")

    results["curvature"] = curvature_data

    # Save
    out_path = "/workspace/Vizuara-VLA-Research/experiments/embedding_geometry_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
