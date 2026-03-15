#!/usr/bin/env python3
"""Experiment 429: Detection Robustness Stress Test

Pushes the cosine distance detector to its limits with edge cases:
novel corruption types, extreme parameter ranges, adversarial
centroid perturbations, and contaminated calibration sets.

Tests:
1. Novel corruption types (not used in calibration)
2. Extreme severity ranges (0.01 to 5.0)
3. Centroid perturbation robustness
4. Contaminated calibration (corrupt samples in calibration set)
5. Minimal calibration set (1-2 images)
"""

import json, time, os, sys
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from transformers import AutoModelForVision2Seq, AutoProcessor

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
    elif ctype == 'contrast':
        return ImageEnhance.Contrast(image).enhance(max(0.01, 1 - severity))
    elif ctype == 'brightness':
        return ImageEnhance.Brightness(image).enhance(max(0.01, 1 - 0.9 * severity))
    elif ctype == 'saturation':
        return ImageEnhance.Saturation(image).enhance(max(0.01, 1 - severity))
    elif ctype == 'jpeg':
        import io
        quality = max(1, int(100 - 90 * severity))
        buf = io.BytesIO()
        image.save(buf, format='JPEG', quality=quality)
        buf.seek(0)
        return Image.open(buf).convert('RGB')
    elif ctype == 'invert':
        arr = 1.0 - arr * severity
        arr = np.clip(arr, 0, 1)
    elif ctype == 'pixelate':
        size = max(1, int(224 / (1 + 20 * severity)))
        small = image.resize((size, size), Image.NEAREST)
        return small.resize((224, 224), Image.NEAREST)
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))

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

def compute_auroc(id_scores, ood_scores):
    id_s = np.asarray(id_scores, dtype=np.float64)
    ood_s = np.asarray(ood_scores, dtype=np.float64)
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

    seeds = [42, 123, 456, 789, 999]
    scenes = [Image.fromarray(np.random.RandomState(s).randint(0, 255, (224, 224, 3), dtype=np.uint8)) for s in seeds]

    print("Extracting clean embeddings...")
    clean_embs = [extract_hidden(model, processor, s, prompt) for s in scenes]
    centroid = np.mean(clean_embs, axis=0)
    clean_dists = [cosine_dist(e, centroid) for e in clean_embs]

    results = {"n_scenes": len(scenes)}

    # === Test 1: Novel corruption types ===
    print("\n=== Novel Corruption Types ===")
    novel_corruptions = ['contrast', 'brightness', 'saturation', 'jpeg', 'invert', 'pixelate']
    novel_results = {}
    for c in novel_corruptions:
        ood_dists = []
        for s in scenes:
            try:
                emb = extract_hidden(model, processor, apply_corruption(s, c), prompt)
                ood_dists.append(float(cosine_dist(emb, centroid)))
            except Exception as e:
                print(f"    Error with {c}: {e}")
                ood_dists.append(0.0)
        auroc = float(compute_auroc(clean_dists, ood_dists))
        novel_results[c] = {
            "auroc": auroc,
            "mean_dist": float(np.mean(ood_dists)),
        }
        print(f"  {c}: AUROC={auroc:.4f}, dist={np.mean(ood_dists):.6f}")
    results["novel_corruptions"] = novel_results

    # === Test 2: Extreme severity ranges ===
    print("\n=== Extreme Severity Ranges ===")
    severities = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0]
    severity_results = {}
    for c in ['fog', 'night', 'noise']:
        per_sev = {}
        for sev in severities:
            ood_dists = []
            for s in scenes:
                emb = extract_hidden(model, processor, apply_corruption(s, c, sev), prompt)
                ood_dists.append(float(cosine_dist(emb, centroid)))
            auroc = float(compute_auroc(clean_dists, ood_dists))
            per_sev[str(sev)] = {"auroc": auroc, "mean_dist": float(np.mean(ood_dists))}
        severity_results[c] = per_sev
        print(f"  {c}: sev=0.01 AUROC={per_sev['0.01']['auroc']:.4f}, sev=5.0 AUROC={per_sev['5.0']['auroc']:.4f}")
    results["extreme_severity"] = severity_results

    # === Test 3: Centroid perturbation ===
    print("\n=== Centroid Perturbation Robustness ===")
    rng = np.random.RandomState(42)
    perturbation_results = {}
    for noise_level in [0.001, 0.01, 0.05, 0.1, 0.5]:
        perturbed_centroid = centroid + rng.randn(*centroid.shape) * noise_level
        perturbed_clean_dists = [cosine_dist(e, perturbed_centroid) for e in clean_embs]

        all_ood = []
        for c in ['fog', 'night', 'noise', 'blur']:
            for s in scenes:
                emb = extract_hidden(model, processor, apply_corruption(s, c), prompt)
                all_ood.append(float(cosine_dist(emb, perturbed_centroid)))

        auroc = float(compute_auroc(perturbed_clean_dists, all_ood))
        perturbation_results[str(noise_level)] = {"auroc": auroc}
        print(f"  noise={noise_level}: AUROC={auroc:.4f}")
    results["centroid_perturbation"] = perturbation_results

    # === Test 4: Contaminated calibration ===
    print("\n=== Contaminated Calibration ===")
    contamination_results = {}
    for n_contam in [0, 1, 2, 3]:
        # Add n_contam corrupted images to the "clean" calibration set
        calib_embs = list(clean_embs)
        for i in range(n_contam):
            corrupt_img = apply_corruption(scenes[i], 'fog')
            calib_embs.append(extract_hidden(model, processor, corrupt_img, prompt))

        contam_centroid = np.mean(calib_embs, axis=0)
        contam_clean_dists = [cosine_dist(e, contam_centroid) for e in clean_embs]

        all_ood = []
        for c in ['fog', 'night', 'noise', 'blur']:
            for s in scenes:
                emb = extract_hidden(model, processor, apply_corruption(s, c), prompt)
                all_ood.append(float(cosine_dist(emb, contam_centroid)))

        auroc = float(compute_auroc(contam_clean_dists, all_ood))
        contamination_results[str(n_contam)] = {
            "auroc": auroc,
            "n_total_calib": len(calib_embs),
            "contamination_rate": n_contam / len(calib_embs),
        }
        print(f"  {n_contam} contaminated ({n_contam}/{len(calib_embs)}): AUROC={auroc:.4f}")
    results["contaminated_calibration"] = contamination_results

    # === Test 5: Minimal calibration set ===
    print("\n=== Minimal Calibration Set ===")
    minimal_results = {}
    for n_calib in [1, 2, 3, 4, 5]:
        calib_subset = clean_embs[:n_calib]
        mini_centroid = np.mean(calib_subset, axis=0)
        # Test on ALL scenes (including those in calibration)
        mini_clean_dists = [cosine_dist(e, mini_centroid) for e in clean_embs]

        all_ood = []
        for c in ['fog', 'night', 'noise', 'blur']:
            for s in scenes:
                emb = extract_hidden(model, processor, apply_corruption(s, c), prompt)
                all_ood.append(float(cosine_dist(emb, mini_centroid)))

        auroc = float(compute_auroc(mini_clean_dists, all_ood))
        minimal_results[str(n_calib)] = {"auroc": auroc}
        print(f"  {n_calib} calibration images: AUROC={auroc:.4f}")
    results["minimal_calibration"] = minimal_results

    out_path = "/workspace/Vizuara-VLA-Research/experiments/detection_robustness_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
