#!/usr/bin/env python3
"""Experiment 343: Real-World Camera Pipeline Effects

Test detection robustness against realistic camera artifacts:
1. Motion blur (camera/object movement)
2. Sensor noise (ISO-related)
3. Auto-exposure changes
4. White balance shifts
5. Lens distortion simulation
6. Combined pipeline effects
"""

import json, time, os, sys
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

def apply_motion_blur(image, kernel_size, angle=0):
    """Simulate directional motion blur."""
    arr = np.array(image).astype(np.float32)
    kernel = np.zeros((kernel_size, kernel_size))
    mid = kernel_size // 2
    if angle == 0:  # horizontal
        kernel[mid, :] = 1.0 / kernel_size
    elif angle == 90:  # vertical
        kernel[:, mid] = 1.0 / kernel_size
    else:  # diagonal
        for i in range(kernel_size):
            kernel[i, i] = 1.0 / kernel_size

    from PIL import ImageFilter
    # Use PIL box blur as approximation
    return image.filter(ImageFilter.BoxBlur(radius=kernel_size//2))

def apply_sensor_noise(image, iso_level):
    """Simulate sensor noise at different ISO levels."""
    arr = np.array(image).astype(np.float32) / 255.0
    rng = np.random.RandomState(42)
    # Shot noise + read noise
    shot_noise = rng.poisson(arr * 255 * iso_level / 100) / (255 * iso_level / 100) - arr
    read_noise = rng.randn(*arr.shape) * iso_level / 10000
    arr = np.clip(arr + shot_noise * 0.1 + read_noise, 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))

def apply_exposure(image, ev_shift):
    """Simulate exposure compensation (+/- EV stops)."""
    factor = 2.0 ** ev_shift
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def apply_white_balance(image, temp_shift):
    """Simulate white balance shift (warm/cool)."""
    arr = np.array(image).astype(np.float32)
    if temp_shift > 0:  # warmer
        arr[:, :, 0] = np.clip(arr[:, :, 0] * (1 + temp_shift * 0.1), 0, 255)  # more red
        arr[:, :, 2] = np.clip(arr[:, :, 2] * (1 - temp_shift * 0.05), 0, 255)  # less blue
    else:  # cooler
        arr[:, :, 0] = np.clip(arr[:, :, 0] * (1 + temp_shift * 0.05), 0, 255)  # less red
        arr[:, :, 2] = np.clip(arr[:, :, 2] * (1 - temp_shift * 0.1), 0, 255)  # more blue
    return Image.fromarray(arr.astype(np.uint8))

def apply_lens_distortion(image, k):
    """Simulate barrel/pincushion distortion."""
    arr = np.array(image)
    h, w = arr.shape[:2]
    cx, cy = w / 2, h / 2

    result = np.zeros_like(arr)
    for y in range(h):
        for x in range(w):
            dx = (x - cx) / cx
            dy = (y - cy) / cy
            r = np.sqrt(dx*dx + dy*dy)
            r_new = r * (1 + k * r * r)
            src_x = int(cx + dx * r_new / r * cx) if r > 0 else x
            src_y = int(cy + dy * r_new / r * cy) if r > 0 else y
            if 0 <= src_x < w and 0 <= src_y < h:
                result[y, x] = arr[src_y, src_x]
    return Image.fromarray(result)

def main():
    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    results = {}

    # Create scenes
    seeds = [0, 100, 200, 300, 400]
    scenes = {}
    cal_embs = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        px = rng.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        scenes[seed] = Image.fromarray(px)
        cal_embs[seed] = extract_hidden(model, processor, scenes[seed], prompt)
        print(f"  Scene {seed} calibrated")

    # ========== 1. Motion blur ==========
    print("\n=== Motion Blur ===")
    motion_results = {}
    for seed in seeds:
        for kernel in [3, 5, 7, 11, 15]:
            img = apply_motion_blur(scenes[seed], kernel)
            emb = extract_hidden(model, processor, img, prompt)
            d = cosine_dist(cal_embs[seed], emb)
            key = f"s{seed}_k{kernel}"
            motion_results[key] = float(d)
        print(f"  Scene {seed}: k3={motion_results[f's{seed}_k3']:.6f}, "
              f"k15={motion_results[f's{seed}_k15']:.6f}")

    # Aggregate by kernel size
    motion_summary = {}
    for kernel in [3, 5, 7, 11, 15]:
        dists = [motion_results[f's{seed}_k{kernel}'] for seed in seeds]
        motion_summary[str(kernel)] = {
            'mean': float(np.mean(dists)),
            'max': float(np.max(dists)),
            'min': float(np.min(dists)),
        }
    results['motion_blur'] = {'per_scene': motion_results, 'summary': motion_summary}

    # ========== 2. Sensor noise ==========
    print("\n=== Sensor Noise (ISO) ===")
    iso_results = {}
    for seed in seeds:
        for iso in [100, 200, 400, 800, 1600, 3200]:
            img = apply_sensor_noise(scenes[seed], iso)
            emb = extract_hidden(model, processor, img, prompt)
            d = cosine_dist(cal_embs[seed], emb)
            key = f"s{seed}_iso{iso}"
            iso_results[key] = float(d)
        print(f"  Scene {seed}: ISO100={iso_results[f's{seed}_iso100']:.6f}, "
              f"ISO3200={iso_results[f's{seed}_iso3200']:.6f}")

    iso_summary = {}
    for iso in [100, 200, 400, 800, 1600, 3200]:
        dists = [iso_results[f's{seed}_iso{iso}'] for seed in seeds]
        iso_summary[str(iso)] = {
            'mean': float(np.mean(dists)),
            'max': float(np.max(dists)),
        }
    results['sensor_noise'] = {'per_scene': iso_results, 'summary': iso_summary}

    # ========== 3. Exposure changes ==========
    print("\n=== Exposure Changes ===")
    exposure_results = {}
    for seed in seeds:
        for ev in [-2.0, -1.0, -0.5, -0.3, 0.3, 0.5, 1.0, 2.0]:
            img = apply_exposure(scenes[seed], ev)
            emb = extract_hidden(model, processor, img, prompt)
            d = cosine_dist(cal_embs[seed], emb)
            key = f"s{seed}_ev{ev}"
            exposure_results[key] = float(d)
        print(f"  Scene {seed}: EV-2={exposure_results[f's{seed}_ev-2.0']:.6f}, "
              f"EV+2={exposure_results[f's{seed}_ev2.0']:.6f}")

    ev_summary = {}
    for ev in [-2.0, -1.0, -0.5, -0.3, 0.3, 0.5, 1.0, 2.0]:
        dists = [exposure_results[f's{seed}_ev{ev}'] for seed in seeds]
        ev_summary[str(ev)] = {
            'mean': float(np.mean(dists)),
            'max': float(np.max(dists)),
        }
    results['exposure'] = {'per_scene': exposure_results, 'summary': ev_summary}

    # ========== 4. White balance ==========
    print("\n=== White Balance ===")
    wb_results = {}
    for seed in seeds:
        for temp in [-3, -2, -1, 1, 2, 3]:
            img = apply_white_balance(scenes[seed], temp)
            emb = extract_hidden(model, processor, img, prompt)
            d = cosine_dist(cal_embs[seed], emb)
            key = f"s{seed}_wb{temp}"
            wb_results[key] = float(d)
        print(f"  Scene {seed}: WB-3={wb_results[f's{seed}_wb-3']:.6f}, "
              f"WB+3={wb_results[f's{seed}_wb3']:.6f}")

    wb_summary = {}
    for temp in [-3, -2, -1, 1, 2, 3]:
        dists = [wb_results[f's{seed}_wb{temp}'] for seed in seeds]
        wb_summary[str(temp)] = {
            'mean': float(np.mean(dists)),
            'max': float(np.max(dists)),
        }
    results['white_balance'] = {'per_scene': wb_results, 'summary': wb_summary}

    # ========== 5. Comparison to real corruption ==========
    print("\n=== Pipeline vs Real Corruption ===")

    comparison = {}
    for seed in seeds[:3]:
        cal = cal_embs[seed]

        # Worst-case pipeline artifacts
        pipeline_dists = []
        for kernel in [5, 7]:
            img = apply_motion_blur(scenes[seed], kernel)
            emb = extract_hidden(model, processor, img, prompt)
            pipeline_dists.append(cosine_dist(cal, emb))
        for ev in [-0.5, 0.5]:
            img = apply_exposure(scenes[seed], ev)
            emb = extract_hidden(model, processor, img, prompt)
            pipeline_dists.append(cosine_dist(cal, emb))
        for temp in [-1, 1]:
            img = apply_white_balance(scenes[seed], temp)
            emb = extract_hidden(model, processor, img, prompt)
            pipeline_dists.append(cosine_dist(cal, emb))

        # Mild corruption
        corruption_dists = []
        for ct in ['fog', 'night', 'noise', 'blur']:
            img = apply_corruption(scenes[seed], ct, 0.3)
            emb = extract_hidden(model, processor, img, prompt)
            corruption_dists.append(cosine_dist(cal, emb))

        comparison[str(seed)] = {
            'max_pipeline': float(max(pipeline_dists)),
            'min_corruption': float(min(corruption_dists)),
            'gap_ratio': float(min(corruption_dists) / max(pipeline_dists)) if max(pipeline_dists) > 0 else float('inf'),
            'separable': bool(min(corruption_dists) > max(pipeline_dists)),
        }
        print(f"  Scene {seed}: max_pipeline={max(pipeline_dists):.6f}, "
              f"min_corrupt={min(corruption_dists):.6f}, "
              f"separable={comparison[str(seed)]['separable']}")

    results['comparison'] = comparison

    # ========== 6. Combined pipeline effects ==========
    print("\n=== Combined Pipeline ===")

    combined = {}
    for seed in seeds[:3]:
        cal = cal_embs[seed]

        # Realistic pipeline: slight motion + exposure + WB
        img = scenes[seed]
        img = apply_motion_blur(img, 3)
        img = apply_exposure(img, 0.3)
        img = apply_white_balance(img, 1)
        emb = extract_hidden(model, processor, img, prompt)
        d_pipeline = cosine_dist(cal, emb)

        # Same pipeline + corruption
        img_c = apply_corruption(scenes[seed], 'fog', 0.3)
        img_c = apply_motion_blur(img_c, 3)
        img_c = apply_exposure(img_c, 0.3)
        img_c = apply_white_balance(img_c, 1)
        emb_c = extract_hidden(model, processor, img_c, prompt)
        d_pipeline_corrupt = cosine_dist(cal, emb_c)

        combined[str(seed)] = {
            'pipeline_only': float(d_pipeline),
            'pipeline_plus_fog03': float(d_pipeline_corrupt),
            'corruption_signal': float(d_pipeline_corrupt - d_pipeline),
        }
        print(f"  Scene {seed}: pipeline={d_pipeline:.6f}, "
              f"pipeline+fog={d_pipeline_corrupt:.6f}")

    results['combined_pipeline'] = combined

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/camera_pipeline_{ts}.json"
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
