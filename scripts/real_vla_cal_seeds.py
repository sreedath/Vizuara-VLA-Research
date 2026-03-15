#!/usr/bin/env python3
"""Experiment 300: Calibration Seed Robustness
Tests how calibration image choice affects detection:
1. 20 different random seeds for calibration images
2. Cross-seed AUROC stability
3. Cross-image detection (calibrate on one image, test on different)
4. Multi-image calibration improvement curve
5. Distance distribution analysis
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

def extract_hidden(model, processor, image, prompt, layer=3):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return fwd.hidden_states[layer][0, -1, :].float().cpu().numpy()

def main():
    print("Loading OpenVLA-7B...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    corruptions = ['fog', 'night', 'blur', 'noise']

    results = {
        "experiment": "calibration_seed_robustness",
        "experiment_number": 300,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    # Part 1: Generate 20 calibration images with different seeds
    print("=== Part 1: Generating Calibration Images ===")
    seeds = list(range(20))
    cal_images = {}
    cal_embeddings = {}

    for seed in seeds:
        np.random.seed(seed)
        pixels = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(pixels)
        cal_images[seed] = img
        emb = extract_hidden(model, processor, img, prompt)
        cal_embeddings[seed] = emb
        print(f"  Seed {seed}: norm={np.linalg.norm(emb):.4f}")

    # Part 2: Cross-seed AUROC (same-image calibration)
    print("\n=== Part 2: Same-Image AUROC per Seed ===")
    same_image_aurocs = {}

    for cal_seed in seeds:
        cal_emb = cal_embeddings[cal_seed]
        cal_img = cal_images[cal_seed]

        # ID: same-seed clean passes
        id_dists = []
        for _ in range(3):
            emb = extract_hidden(model, processor, cal_img, prompt)
            id_dists.append(float(cosine(cal_emb, emb)))

        # OOD: corruptions of THIS calibration image
        per_corruption_auroc = {}
        for c in corruptions:
            ood_dists = []
            for sev in [0.3, 0.5, 1.0]:
                corrupted = apply_corruption(cal_img, c, sev)
                emb = extract_hidden(model, processor, corrupted, prompt)
                ood_dists.append(float(cosine(cal_emb, emb)))
            auroc = compute_auroc(id_dists, ood_dists)
            per_corruption_auroc[c] = auroc

        same_image_aurocs[cal_seed] = per_corruption_auroc
        print(f"  Seed {cal_seed}: " + ", ".join(f"{c}={a:.3f}" for c, a in per_corruption_auroc.items()))

    results["same_image_aurocs"] = same_image_aurocs

    # Part 3: Cross-image detection
    print("\n=== Part 3: Cross-Image Detection ===")
    cross_image = {}

    for cal_seed in [0, 5, 10, 15]:
        cal_emb = cal_embeddings[cal_seed]
        cross_image[cal_seed] = {}

        for test_seed in [0, 5, 10, 15]:
            if test_seed == cal_seed:
                continue
            test_img = cal_images[test_seed]

            # ID: clean test image
            id_dists = [float(cosine(cal_emb, extract_hidden(model, processor, test_img, prompt))) for _ in range(3)]

            # OOD: corrupted test image
            ood_dists = []
            for c in corruptions:
                for sev in [0.5, 1.0]:
                    corrupted = apply_corruption(test_img, c, sev)
                    ood_dists.append(float(cosine(cal_emb, extract_hidden(model, processor, corrupted, prompt))))

            auroc = compute_auroc(id_dists, ood_dists)
            cross_image[cal_seed][test_seed] = {
                "auroc": auroc,
                "id_mean": float(np.mean(id_dists)),
                "ood_mean": float(np.mean(ood_dists))
            }

        aurocs = [v["auroc"] for v in cross_image[cal_seed].values()]
        auroc_strs = [f"{a:.3f}" for a in aurocs]
        print(f"  Cal={cal_seed}: {auroc_strs}, mean={np.mean(aurocs):.3f}")

    results["cross_image"] = cross_image

    # Part 4: Multi-image calibration
    print("\n=== Part 4: Multi-Image Calibration ===")
    multi_cal = []

    # Fix test image at seed 42
    np.random.seed(42)
    pixels = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    test_img = Image.fromarray(pixels)

    for n_cal in [1, 2, 3, 5, 10, 20]:
        centroid = np.mean([cal_embeddings[s] for s in range(n_cal)], axis=0)

        id_dists = [float(cosine(centroid, extract_hidden(model, processor, test_img, prompt))) for _ in range(3)]

        all_ood = []
        per_corruption = {}
        for c in corruptions:
            ood_dists = []
            for sev in [0.3, 0.5, 1.0]:
                corrupted = apply_corruption(test_img, c, sev)
                ood_dists.append(float(cosine(centroid, extract_hidden(model, processor, corrupted, prompt))))
            auroc = compute_auroc(id_dists, ood_dists)
            per_corruption[c] = auroc
            all_ood.extend(ood_dists)

        overall = compute_auroc(id_dists, all_ood)
        multi_cal.append({
            "n_cal": n_cal,
            "overall_auroc": float(overall),
            "per_corruption": per_corruption,
            "id_mean": float(np.mean(id_dists)),
            "ood_mean": float(np.mean(all_ood))
        })
        print(f"  n={n_cal}: AUROC={overall:.3f}")

    results["multi_cal"] = multi_cal

    # Part 5: Embedding similarity between seeds
    print("\n=== Part 5: Cross-Seed Embedding Similarity ===")
    sim_matrix = {}
    for s1 in range(0, 20, 4):
        for s2 in range(s1+4, 20, 4):
            sim = float(1 - cosine(cal_embeddings[s1], cal_embeddings[s2]))
            sim_matrix[f"{s1}_vs_{s2}"] = sim
    results["cross_seed_similarity"] = sim_matrix

    cos_sims = list(sim_matrix.values())
    print(f"  Cross-seed cosine similarity: mean={np.mean(cos_sims):.6f}, min={min(cos_sims):.6f}, max={max(cos_sims):.6f}")

    # Save
    ts = results["timestamp"]
    out_path = f"experiments/cal_seeds_{ts}.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
