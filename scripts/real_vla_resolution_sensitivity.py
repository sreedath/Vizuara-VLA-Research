#!/usr/bin/env python3
"""Experiment 378: Resolution and Image Preprocessing Sensitivity

How does image size, crop, and preprocessing affect detection?
1. Different input resolutions (resize before model's 224x224)
2. Random crop vs center crop
3. JPEG compression artifacts
4. Color space perturbations (brightness, contrast, saturation)
5. Image format/bit depth effects
"""

import json, time, os, sys, io
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageEnhance
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

def cosine_dist(a, b):
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return 1.0 - dot / (na * nb)

def compute_auroc(id_scores, ood_scores):
    id_s = np.asarray(id_scores)
    ood_s = np.asarray(ood_scores)
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
    results = {}
    ctypes = ['fog', 'night', 'noise', 'blur']

    # Generate base images at 224x224
    print("Generating images...")
    seeds = list(range(0, 1000, 100))[:10]
    images = {}
    clean_embs = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        px = rng.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        images[seed] = Image.fromarray(px)
        clean_embs[seed] = extract_hidden(model, processor, images[seed], prompt)

    centroid = np.mean(list(clean_embs.values()), axis=0)
    clean_dists = [cosine_dist(centroid, clean_embs[s]) for s in seeds]
    threshold = max(clean_dists)
    print(f"  Threshold: {threshold:.6f}")

    # ========== 1. Input Resolution Effects ==========
    print("\n=== Resolution Effects ===")

    resolutions = [56, 112, 224, 448, 672]
    resolution_results = {}

    for res in resolutions:
        res_dists = []
        for seed in seeds[:5]:
            rng = np.random.RandomState(seed)
            px = rng.randint(50, 200, (res, res, 3), dtype=np.uint8)
            img = Image.fromarray(px)
            emb = extract_hidden(model, processor, img, prompt)
            d = cosine_dist(emb, centroid)
            res_dists.append(d)

        resolution_results[str(res)] = {
            'mean_dist': float(np.mean(res_dists)),
            'max_dist': float(max(res_dists)),
            'all_below_threshold': all(d <= threshold for d in res_dists),
            'false_positive_rate': float(sum(1 for d in res_dists if d > threshold) / len(res_dists)),
        }
        print(f"  {res}x{res}: mean_dist={np.mean(res_dists):.6f}, "
              f"FPR={resolution_results[str(res)]['false_positive_rate']:.2f}")

    results['resolution'] = resolution_results

    # ========== 2. JPEG Compression ==========
    print("\n=== JPEG Compression ===")

    jpeg_results = {}
    for quality in [5, 10, 20, 50, 80, 95]:
        jpeg_dists = []
        for seed in seeds[:5]:
            buf = io.BytesIO()
            images[seed].save(buf, format='JPEG', quality=quality)
            buf.seek(0)
            jpeg_img = Image.open(buf).convert('RGB')
            emb = extract_hidden(model, processor, jpeg_img, prompt)
            d = cosine_dist(emb, centroid)
            jpeg_dists.append(d)

        jpeg_results[str(quality)] = {
            'mean_dist': float(np.mean(jpeg_dists)),
            'max_dist': float(max(jpeg_dists)),
            'false_positive_rate': float(sum(1 for d in jpeg_dists if d > threshold) / len(jpeg_dists)),
        }
        print(f"  Q={quality}: mean_dist={np.mean(jpeg_dists):.6f}, "
              f"FPR={jpeg_results[str(quality)]['false_positive_rate']:.2f}")

    results['jpeg'] = jpeg_results

    # ========== 3. Color Adjustments ==========
    print("\n=== Color Adjustments ===")

    color_results = {}
    adjustments = {
        'brightness_0.5': ('brightness', 0.5),
        'brightness_0.8': ('brightness', 0.8),
        'brightness_1.2': ('brightness', 1.2),
        'brightness_1.5': ('brightness', 1.5),
        'contrast_0.5': ('contrast', 0.5),
        'contrast_0.8': ('contrast', 0.8),
        'contrast_1.2': ('contrast', 1.2),
        'contrast_1.5': ('contrast', 1.5),
        'saturation_0.0': ('saturation', 0.0),
        'saturation_0.5': ('saturation', 0.5),
        'saturation_1.5': ('saturation', 1.5),
    }

    for name, (adj_type, factor) in adjustments.items():
        adj_dists = []
        for seed in seeds[:5]:
            img = images[seed].copy()
            if adj_type == 'brightness':
                img = ImageEnhance.Brightness(img).enhance(factor)
            elif adj_type == 'contrast':
                img = ImageEnhance.Contrast(img).enhance(factor)
            elif adj_type == 'saturation':
                img = ImageEnhance.Color(img).enhance(factor)
            emb = extract_hidden(model, processor, img, prompt)
            d = cosine_dist(emb, centroid)
            adj_dists.append(d)

        color_results[name] = {
            'mean_dist': float(np.mean(adj_dists)),
            'max_dist': float(max(adj_dists)),
            'false_positive_rate': float(sum(1 for d in adj_dists if d > threshold) / len(adj_dists)),
        }
        print(f"  {name}: mean_dist={np.mean(adj_dists):.6f}, "
              f"FPR={color_results[name]['false_positive_rate']:.2f}")

    results['color_adjustments'] = color_results

    # ========== 4. Crop Effects ==========
    print("\n=== Crop Effects ===")

    crop_results = {}
    for crop_frac in [0.7, 0.8, 0.9, 0.95, 1.0]:
        crop_dists = []
        for seed in seeds[:5]:
            rng = np.random.RandomState(seed)
            big_size = int(224 / crop_frac)
            px = rng.randint(50, 200, (big_size, big_size, 3), dtype=np.uint8)
            big_img = Image.fromarray(px)
            left = (big_size - 224) // 2
            top = (big_size - 224) // 2
            crop_img = big_img.crop((left, top, left + 224, top + 224))
            emb = extract_hidden(model, processor, crop_img, prompt)
            d = cosine_dist(emb, centroid)
            crop_dists.append(d)

        crop_results[str(crop_frac)] = {
            'mean_dist': float(np.mean(crop_dists)),
            'max_dist': float(max(crop_dists)),
            'false_positive_rate': float(sum(1 for d in crop_dists if d > threshold) / len(crop_dists)),
        }
        print(f"  crop={crop_frac}: mean_dist={np.mean(crop_dists):.6f}")

    results['crop'] = crop_results

    # ========== 5. Detection Under Preprocessing ==========
    print("\n=== Detection Under JPEG + Corruption ===")

    preprocess_det = {}
    for quality in [10, 50, 95]:
        for ct in ctypes:
            det_dists = []
            for seed in seeds[:5]:
                corrupt_img = apply_corruption(images[seed], ct, 0.5)
                buf = io.BytesIO()
                corrupt_img.save(buf, format='JPEG', quality=quality)
                buf.seek(0)
                jpeg_corrupt = Image.open(buf).convert('RGB')
                emb = extract_hidden(model, processor, jpeg_corrupt, prompt)
                d = cosine_dist(emb, centroid)
                det_dists.append(d)

            key = f"Q{quality}_{ct}"
            auroc = compute_auroc(clean_dists[:5], det_dists)
            preprocess_det[key] = {
                'mean_dist': float(np.mean(det_dists)),
                'detection_rate': float(sum(1 for d in det_dists if d > threshold) / len(det_dists)),
                'auroc': float(auroc),
            }
            print(f"  Q{quality}+{ct}: det={preprocess_det[key]['detection_rate']:.2f}, "
                  f"AUROC={auroc:.4f}")

    results['preprocess_detection'] = preprocess_det

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/resolution_sensitivity_{ts}.json"
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
