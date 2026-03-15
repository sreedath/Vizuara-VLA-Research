#!/usr/bin/env python3
"""Experiment 390: Multi-Scene Robustness Validation

Tests detection performance across multiple distinct driving scenes to validate
that the detection method generalizes beyond a single scene. Uses different
random seeds to generate structurally different synthetic scenes.

Tests:
1. 5 distinct scenes with different visual properties
2. Per-scene calibration AUROC
3. Cross-scene transfer (calibrate on scene A, test on scene B)
4. Scene-averaged detection performance
5. Corruption shift consistency across scenes (>0.96 expected)
6. Detection threshold portability
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

def compute_auroc(id_scores, ood_scores):
    id_s = np.asarray(id_scores)
    ood_s = np.asarray(ood_scores)
    n_id, n_ood = len(id_s), len(ood_s)
    if n_id == 0 or n_ood == 0:
        return 0.5
    count = sum(float(np.sum(o > id_s) + 0.5 * np.sum(o == id_s)) for o in ood_s)
    return count / (n_id * n_ood)

def cosine_dist(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    return 1.0 - np.dot(a, b) / (na * nb)

def generate_scene(seed, size=224):
    """Generate a visually distinct synthetic scene."""
    rng = np.random.RandomState(seed)
    scene_type = seed % 5

    if scene_type == 0:
        # Urban: structured grid pattern with random colors
        img = np.zeros((size, size, 3), dtype=np.uint8)
        for i in range(0, size, 32):
            for j in range(0, size, 32):
                color = rng.randint(50, 250, 3)
                img[i:i+32, j:j+32] = color
        # Add "road" strip
        img[size//2-16:size//2+16, :] = [80, 80, 80]
    elif scene_type == 1:
        # Highway: gradient sky + road
        img = np.zeros((size, size, 3), dtype=np.uint8)
        for y in range(size):
            ratio = y / size
            img[y, :] = [int(135*(1-ratio)), int(206*(1-ratio)), int(235*(1-ratio))]
        img[size//2:, :] = rng.randint(40, 100, (size//2, size, 3))
    elif scene_type == 2:
        # Parking lot: regular pattern
        img = rng.randint(100, 180, (size, size, 3), dtype=np.uint8)
        for i in range(0, size, 40):
            img[i:i+2, :] = [255, 255, 255]
            img[:, i:i+2] = [255, 255, 255]
    elif scene_type == 3:
        # Rural: green-brown terrain
        img = np.zeros((size, size, 3), dtype=np.uint8)
        img[:size//3] = [100, 150, 220]  # sky
        img[size//3:2*size//3] = [50, 120, 30]  # trees
        img[2*size//3:] = [120, 100, 60]  # road
        img += rng.randint(-20, 20, img.shape).astype(np.int16).clip(0, 255).astype(np.uint8)
        img = np.clip(img, 0, 255).astype(np.uint8)
    else:
        # Random natural: pure random
        img = rng.randint(0, 255, (size, size, 3), dtype=np.uint8)

    return Image.fromarray(img)

def main():
    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    corruptions = ['fog', 'night', 'noise', 'blur']
    scene_seeds = [42, 123, 456, 789, 1024]
    n_scenes = len(scene_seeds)
    n_clean = 5
    n_ood = 5

    # Collect embeddings for each scene
    scene_data = {}
    for si, seed in enumerate(scene_seeds):
        scene_name = f"scene_{si}"
        print(f"\n=== {scene_name} (seed={seed}) ===")
        base_img = generate_scene(seed)

        # Clean embeddings
        clean_embs = []
        for i in range(n_clean):
            arr = np.array(base_img).astype(np.float32)
            arr += np.random.RandomState(seed * 100 + i).randn(*arr.shape) * 0.5
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            emb = extract_hidden(model, processor, Image.fromarray(arr), prompt)
            clean_embs.append(emb)
            print(f"  Clean {i+1}/{n_clean}")

        centroid = np.mean(clean_embs, axis=0)

        # Corrupted embeddings
        corrupt_embs = {}
        for c in corruptions:
            corrupt_embs[c] = []
            for i in range(n_ood):
                arr = np.array(base_img).astype(np.float32)
                arr += np.random.RandomState(seed * 100 + 50 + i).randn(*arr.shape) * 0.5
                arr = np.clip(arr, 0, 255).astype(np.uint8)
                corrupted = apply_corruption(Image.fromarray(arr), c)
                emb = extract_hidden(model, processor, corrupted, prompt)
                corrupt_embs[c].append(emb)
                print(f"  {c} {i+1}/{n_ood}")

        scene_data[scene_name] = {
            "clean_embs": clean_embs,
            "centroid": centroid,
            "corrupt_embs": corrupt_embs
        }

    results = {
        "n_scenes": n_scenes,
        "scene_seeds": scene_seeds,
        "n_clean": n_clean,
        "n_ood": n_ood
    }

    # 1. Per-scene calibration AUROC
    print("\n=== Per-Scene Calibration ===")
    per_scene_aurocs = {}
    for sname, sdata in scene_data.items():
        aurocs = {}
        for c in corruptions:
            id_scores = [cosine_dist(e, sdata["centroid"]) for e in sdata["clean_embs"]]
            ood_scores = [cosine_dist(e, sdata["centroid"]) for e in sdata["corrupt_embs"][c]]
            aurocs[c] = compute_auroc(id_scores, ood_scores)
        per_scene_aurocs[sname] = aurocs
        print(f"  {sname}: {aurocs}")
    results["per_scene_aurocs"] = per_scene_aurocs

    # 2. Cross-scene transfer
    print("\n=== Cross-Scene Transfer ===")
    cross_scene = {}
    for cal_name, cal_data in scene_data.items():
        cross_scene[cal_name] = {}
        for test_name, test_data in scene_data.items():
            if cal_name == test_name:
                continue
            aurocs = {}
            # Use cal_name's centroid, test_name's data
            centroid = cal_data["centroid"]
            for c in corruptions:
                id_scores = [cosine_dist(e, centroid) for e in test_data["clean_embs"]]
                ood_scores = [cosine_dist(e, centroid) for e in test_data["corrupt_embs"][c]]
                aurocs[c] = compute_auroc(id_scores, ood_scores)
            cross_scene[cal_name][test_name] = aurocs
        print(f"  Calibrate on {cal_name}: {cross_scene[cal_name]}")
    results["cross_scene_transfer"] = cross_scene

    # 3. Cross-scene transfer summary stats
    print("\n=== Transfer Summary ===")
    transfer_summary = {}
    for c in corruptions:
        vals = []
        for cal_name in cross_scene:
            for test_name in cross_scene[cal_name]:
                vals.append(cross_scene[cal_name][test_name][c])
        transfer_summary[c] = {
            "mean": float(np.mean(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "std": float(np.std(vals))
        }
        print(f"  {c}: mean={np.mean(vals):.4f}, min={np.min(vals):.4f}, max={np.max(vals):.4f}")
    results["transfer_summary"] = transfer_summary

    # 4. Corruption shift consistency across scenes
    print("\n=== Shift Consistency ===")
    shift_vectors = {}
    for sname, sdata in scene_data.items():
        shift_vectors[sname] = {}
        for c in corruptions:
            mean_corrupt = np.mean(sdata["corrupt_embs"][c], axis=0)
            shift = mean_corrupt - sdata["centroid"]
            shift_vectors[sname][c] = shift / (np.linalg.norm(shift) + 1e-12)

    # Pairwise cosine similarity of shift directions across scenes
    shift_consistency = {}
    for c in corruptions:
        sims = []
        scene_names = list(shift_vectors.keys())
        for i in range(len(scene_names)):
            for j in range(i+1, len(scene_names)):
                v1 = shift_vectors[scene_names[i]][c]
                v2 = shift_vectors[scene_names[j]][c]
                sim = np.dot(v1, v2)
                sims.append(float(sim))
        shift_consistency[c] = {
            "mean_sim": float(np.mean(sims)),
            "min_sim": float(np.min(sims)),
            "max_sim": float(np.max(sims)),
            "all_sims": sims
        }
        print(f"  {c}: mean_sim={np.mean(sims):.4f}, min={np.min(sims):.4f}")
    results["shift_consistency"] = shift_consistency

    # 5. Inter-scene centroid distances
    print("\n=== Inter-Scene Centroid Distances ===")
    centroids = {sname: sdata["centroid"] for sname, sdata in scene_data.items()}
    inter_dists = {}
    scene_names = list(centroids.keys())
    for i in range(len(scene_names)):
        for j in range(i+1, len(scene_names)):
            d = cosine_dist(centroids[scene_names[i]], centroids[scene_names[j]])
            key = f"{scene_names[i]}_to_{scene_names[j]}"
            inter_dists[key] = float(d)
            print(f"  {key}: {d:.6f}")
    results["inter_scene_centroid_dists"] = inter_dists

    # 6. Detection threshold portability
    print("\n=== Threshold Portability ===")
    threshold_port = {}
    for cal_name, cal_data in scene_data.items():
        # Compute threshold from cal scene (max clean distance)
        cal_dists = [cosine_dist(e, cal_data["centroid"]) for e in cal_data["clean_embs"]]
        threshold = max(cal_dists) * 1.1  # 10% margin

        test_results = {}
        for test_name, test_data in scene_data.items():
            # Apply threshold using cal centroid on test data
            test_clean_dists = [cosine_dist(e, cal_data["centroid"]) for e in test_data["clean_embs"]]
            fpr = float(np.mean([d > threshold for d in test_clean_dists]))

            tprs = {}
            for c in corruptions:
                test_ood_dists = [cosine_dist(e, cal_data["centroid"]) for e in test_data["corrupt_embs"][c]]
                tprs[c] = float(np.mean([d > threshold for d in test_ood_dists]))

            test_results[test_name] = {"fpr": fpr, "tprs": tprs}

        threshold_port[cal_name] = {
            "threshold": float(threshold),
            "results": test_results
        }
    results["threshold_portability"] = threshold_port
    print("  Done.")

    # Save
    out_path = "/workspace/Vizuara-VLA-Research/experiments/multiscene_robustness_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
