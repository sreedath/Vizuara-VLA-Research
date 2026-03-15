#!/usr/bin/env python3
"""Experiment 310: Robustness Under Distribution Shift
Tests detection across diverse scene types to understand generalization:
1. Structured scenes (gradients, patterns, textures)
2. Color-biased scenes (red-heavy, blue-heavy, etc.)
3. Brightness-varied scenes (very dark, very bright)
4. Scene complexity (uniform, simple, complex)
5. Cross-scene calibration transfer matrix
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

def extract_hidden(model, processor, image, prompt, layer=3):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return fwd.hidden_states[layer][0, -1, :].float().cpu().numpy()

def compute_auroc(id_scores, ood_scores):
    id_s = np.asarray(id_scores)
    ood_s = np.asarray(ood_scores)
    n_id, n_ood = len(id_s), len(ood_s)
    if n_id == 0 or n_ood == 0: return 0.5
    count = sum(float(np.sum(o > id_s) + 0.5 * np.sum(o == id_s)) for o in ood_s)
    return count / (n_id * n_ood)

def make_scene(name, size=224):
    """Generate diverse scene types."""
    if name == 'random_42':
        np.random.seed(42)
        return np.random.randint(50, 200, (size, size, 3), dtype=np.uint8)
    elif name == 'gradient_h':
        arr = np.zeros((size, size, 3), dtype=np.uint8)
        for x in range(size):
            arr[:, x, :] = int(x / size * 255)
        return arr
    elif name == 'gradient_v':
        arr = np.zeros((size, size, 3), dtype=np.uint8)
        for y in range(size):
            arr[y, :, :] = int(y / size * 255)
        return arr
    elif name == 'checkerboard':
        arr = np.zeros((size, size, 3), dtype=np.uint8)
        for y in range(size):
            for x in range(size):
                if (x // 32 + y // 32) % 2 == 0:
                    arr[y, x, :] = 200
                else:
                    arr[y, x, :] = 50
        return arr
    elif name == 'red_heavy':
        np.random.seed(42)
        arr = np.random.randint(50, 200, (size, size, 3), dtype=np.uint8)
        arr[:, :, 0] = np.clip(arr[:, :, 0].astype(int) + 80, 0, 255).astype(np.uint8)
        arr[:, :, 1] = np.clip(arr[:, :, 1].astype(int) - 40, 0, 255).astype(np.uint8)
        return arr
    elif name == 'blue_heavy':
        np.random.seed(42)
        arr = np.random.randint(50, 200, (size, size, 3), dtype=np.uint8)
        arr[:, :, 2] = np.clip(arr[:, :, 2].astype(int) + 80, 0, 255).astype(np.uint8)
        arr[:, :, 0] = np.clip(arr[:, :, 0].astype(int) - 40, 0, 255).astype(np.uint8)
        return arr
    elif name == 'very_dark':
        np.random.seed(42)
        return np.random.randint(5, 40, (size, size, 3), dtype=np.uint8)
    elif name == 'very_bright':
        np.random.seed(42)
        return np.random.randint(200, 250, (size, size, 3), dtype=np.uint8)
    elif name == 'gray_uniform':
        return np.full((size, size, 3), 128, dtype=np.uint8)
    elif name == 'random_99':
        np.random.seed(99)
        return np.random.randint(50, 200, (size, size, 3), dtype=np.uint8)
    elif name == 'stripes':
        arr = np.zeros((size, size, 3), dtype=np.uint8)
        for y in range(size):
            if (y // 16) % 2 == 0:
                arr[y, :, :] = 180
            else:
                arr[y, :, :] = 60
        return arr
    elif name == 'circle':
        arr = np.full((size, size, 3), 100, dtype=np.uint8)
        cy, cx = size // 2, size // 2
        for y in range(size):
            for x in range(size):
                if (y - cy)**2 + (x - cx)**2 < (size//3)**2:
                    arr[y, x, :] = 200
        return arr
    return np.random.randint(50, 200, (size, size, 3), dtype=np.uint8)

def main():
    print("Loading OpenVLA-7B...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\nOut:"

    results = {
        "experiment": "distribution_shift",
        "experiment_number": 310,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    corruptions = ['fog', 'night', 'blur', 'noise']
    scene_names = ['random_42', 'gradient_h', 'gradient_v', 'checkerboard',
                   'red_heavy', 'blue_heavy', 'very_dark', 'very_bright',
                   'gray_uniform', 'random_99', 'stripes', 'circle']

    # Part 1: Per-scene same-image detection
    print("=== Part 1: Per-Scene Detection ===")
    per_scene = {}

    for scene_name in scene_names:
        print(f"  {scene_name}...")
        pixels = make_scene(scene_name)
        img = Image.fromarray(pixels)

        # Centroid from clean
        clean_emb = extract_hidden(model, processor, img, prompt)

        # ID scores
        id_dists = []
        for _ in range(3):
            emb = extract_hidden(model, processor, img, prompt)
            id_dists.append(float(cosine(clean_emb, emb)))

        # OOD scores per corruption
        per_c = {}
        for c in corruptions:
            ood_dists = []
            for sev in [0.3, 0.5, 1.0]:
                corrupted = apply_corruption(img, c, sev)
                emb = extract_hidden(model, processor, corrupted, prompt)
                ood_dists.append(float(cosine(clean_emb, emb)))
            auroc = compute_auroc(id_dists, ood_dists)
            per_c[c] = {
                "auroc": auroc,
                "mean_ood_dist": float(np.mean(ood_dists)),
            }

        mean_auroc = np.mean([per_c[c]["auroc"] for c in corruptions])
        per_scene[scene_name] = {
            "per_corruption": per_c,
            "mean_auroc": mean_auroc,
            "id_mean": float(np.mean(id_dists)),
            "clean_norm": float(np.linalg.norm(clean_emb)),
        }
        auroc_str = ", ".join(f"{c}={per_c[c]['auroc']:.3f}" for c in corruptions)
        print(f"    mean={mean_auroc:.3f}: {auroc_str}")

    results["per_scene"] = per_scene

    # Part 2: Cross-scene calibration transfer
    print("\n=== Part 2: Cross-Scene Transfer ===")
    cross_transfer = {}

    # Select 6 representative scenes
    test_scenes = ['random_42', 'gradient_h', 'checkerboard', 'very_dark', 'very_bright', 'random_99']

    for cal_scene in test_scenes:
        cal_pixels = make_scene(cal_scene)
        cal_img = Image.fromarray(cal_pixels)
        cal_emb = extract_hidden(model, processor, cal_img, prompt)

        cross_transfer[cal_scene] = {}
        for test_scene in test_scenes:
            if test_scene == cal_scene:
                continue

            test_pixels = make_scene(test_scene)
            test_img = Image.fromarray(test_pixels)

            # ID: clean test image distance from cal centroid
            id_dists = [float(cosine(cal_emb, extract_hidden(model, processor, test_img, prompt)))
                        for _ in range(3)]

            # OOD: corrupted test image
            ood_dists = []
            for c in corruptions:
                for sev in [0.5, 1.0]:
                    corrupted = apply_corruption(test_img, c, sev)
                    ood_dists.append(float(cosine(cal_emb, extract_hidden(model, processor, corrupted, prompt))))

            auroc = compute_auroc(id_dists, ood_dists)
            cross_transfer[cal_scene][test_scene] = {
                "auroc": auroc,
                "id_mean": float(np.mean(id_dists)),
                "ood_mean": float(np.mean(ood_dists)),
            }

        aurocs = [v["auroc"] for v in cross_transfer[cal_scene].values()]
        print(f"  Cal={cal_scene}: mean cross-AUROC={np.mean(aurocs):.3f}, "
              f"min={min(aurocs):.3f}, max={max(aurocs):.3f}")

    results["cross_transfer"] = cross_transfer

    # Part 3: Scene embedding similarity
    print("\n=== Part 3: Scene Embedding Similarity ===")
    scene_embs = {}
    for scene_name in scene_names:
        pixels = make_scene(scene_name)
        img = Image.fromarray(pixels)
        scene_embs[scene_name] = extract_hidden(model, processor, img, prompt)

    sim_matrix = {}
    for s1 in scene_names:
        for s2 in scene_names:
            if s1 < s2:
                sim = float(1 - cosine(scene_embs[s1], scene_embs[s2]))
                sim_matrix[f"{s1}_vs_{s2}"] = sim

    sims = list(sim_matrix.values())
    results["scene_similarity"] = {
        "matrix": sim_matrix,
        "mean": float(np.mean(sims)),
        "min": float(min(sims)),
        "max": float(max(sims)),
        "std": float(np.std(sims)),
    }
    print(f"  Cross-scene similarity: mean={np.mean(sims):.6f}, "
          f"min={min(sims):.6f}, max={max(sims):.6f}")

    # Part 4: Failure mode analysis
    print("\n=== Part 4: Failure Mode Analysis ===")
    failures = []
    for scene_name, data in per_scene.items():
        for c in corruptions:
            auroc = data["per_corruption"][c]["auroc"]
            if auroc < 1.0:
                failures.append({
                    "scene": scene_name,
                    "corruption": c,
                    "auroc": auroc,
                })
    results["failures"] = failures
    if failures:
        print(f"  {len(failures)} failure cases found:")
        for f in failures:
            print(f"    {f['scene']} + {f['corruption']}: AUROC={f['auroc']:.3f}")
    else:
        print("  No failures — all scenes × corruptions achieve AUROC=1.0")

    # Save
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(v) for v in obj]
        return obj

    ts = results["timestamp"]
    out_path = f"experiments/dist_shift_{ts}.json"
    with open(out_path, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
