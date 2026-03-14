"""
Gradient-Free Sensitivity Analysis.

Tests how minimal pixel perturbations affect detection scores:
single-pixel changes, Gaussian noise at varying sigma, brightness
shifts, and progressive corruption. Establishes the minimum
detectable visual change.

Experiment 108 in the CalibDrive series.
"""
import os
import json
import datetime
import numpy as np
import torch
from PIL import Image

RESULTS_DIR = "/workspace/Vizuara-VLA-Research/experiments"
os.makedirs(RESULTS_DIR, exist_ok=True)
SIZE = (256, 256)


def create_highway(idx):
    rng = np.random.default_rng(idx * 5001)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]
    img[SIZE[0]//2:] = [80, 80, 80]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [255, 255, 255]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def extract_hidden(model, processor, image, prompt):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    if not hasattr(fwd, 'hidden_states') or not fwd.hidden_states:
        return None
    return fwd.hidden_states[-1][0, -1, :].float().cpu().numpy()


def cosine_dist(a, b):
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def main():
    print("=" * 70, flush=True)
    print("GRADIENT-FREE SENSITIVITY ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    from transformers import AutoModelForVision2Seq, AutoProcessor
    print("Loading OpenVLA-7B...", flush=True)
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        "openvla/openvla-7b", trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.", flush=True)

    prompt = "In: What action should the robot take to drive forward at 25 m/s safely?\nOut:"

    # Base image
    base_arr = create_highway(42)
    base_img = Image.fromarray(base_arr)
    base_embed = extract_hidden(model, processor, base_img, prompt)

    # Build calibration centroid from 10 different highways
    cal_embeds = []
    for i in range(10):
        h = extract_hidden(model, processor, Image.fromarray(create_highway(i + 1000)), prompt)
        if h is not None:
            cal_embeds.append(h)
    centroid = np.mean(cal_embeds, axis=0)
    base_score = cosine_dist(base_embed, centroid)
    print(f"Base score: {base_score:.6f}", flush=True)

    # Test 1: Single pixel change
    print("\n--- Test 1: Single Pixel Change ---", flush=True)
    single_pixel_results = []
    for pos in [(0,0), (128,128), (255,255), (128,0), (0,128)]:
        for delta in [1, 5, 10, 50, 127, 255]:
            perturbed = base_arr.copy()
            perturbed[pos[0], pos[1], 0] = min(255, perturbed[pos[0], pos[1], 0] + delta)
            h = extract_hidden(model, processor, Image.fromarray(perturbed), prompt)
            score = cosine_dist(h, centroid)
            dist = cosine_dist(h, base_embed)
            single_pixel_results.append({
                'pos': pos, 'delta': delta,
                'score': float(score), 'dist_from_base': float(dist),
                'score_change': float(score - base_score),
            })
            print(f"  pos={pos}, delta={delta}: score={score:.6f}, change={score-base_score:.6f}", flush=True)

    # Test 2: Gaussian noise at varying sigma
    print("\n--- Test 2: Gaussian Noise ---", flush=True)
    rng = np.random.default_rng(9999)
    gaussian_results = []
    for sigma in [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]:
        noise = rng.normal(0, sigma, base_arr.shape)
        perturbed = np.clip(base_arr.astype(np.float64) + noise, 0, 255).astype(np.uint8)
        n_changed = np.sum(perturbed != base_arr)
        h = extract_hidden(model, processor, Image.fromarray(perturbed), prompt)
        score = cosine_dist(h, centroid)
        dist = cosine_dist(h, base_embed)
        gaussian_results.append({
            'sigma': sigma, 'pixels_changed': int(n_changed),
            'pct_changed': float(n_changed / base_arr.size * 100),
            'score': float(score), 'dist_from_base': float(dist),
            'score_change': float(score - base_score),
        })
        print(f"  sigma={sigma}: score={score:.6f}, change={score-base_score:.6f}, "
              f"pixels_changed={n_changed} ({n_changed/base_arr.size*100:.1f}%)", flush=True)

    # Test 3: Brightness shift
    print("\n--- Test 3: Brightness Shift ---", flush=True)
    brightness_results = []
    for shift in [-100, -50, -20, -10, -5, -1, 1, 5, 10, 20, 50, 100]:
        perturbed = np.clip(base_arr.astype(np.int16) + shift, 0, 255).astype(np.uint8)
        h = extract_hidden(model, processor, Image.fromarray(perturbed), prompt)
        score = cosine_dist(h, centroid)
        dist = cosine_dist(h, base_embed)
        brightness_results.append({
            'shift': shift, 'score': float(score),
            'dist_from_base': float(dist),
            'score_change': float(score - base_score),
        })
        print(f"  shift={shift:+4d}: score={score:.6f}, change={score-base_score:.6f}", flush=True)

    # Test 4: Progressive corruption (blend with noise)
    print("\n--- Test 4: Progressive Corruption ---", flush=True)
    noise_arr = rng.integers(0, 256, base_arr.shape, dtype=np.uint8)
    corruption_results = []
    for alpha in [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        blended = np.clip(
            (1 - alpha) * base_arr.astype(np.float64) + alpha * noise_arr.astype(np.float64),
            0, 255
        ).astype(np.uint8)
        h = extract_hidden(model, processor, Image.fromarray(blended), prompt)
        score = cosine_dist(h, centroid)
        dist = cosine_dist(h, base_embed)
        corruption_results.append({
            'alpha': alpha, 'score': float(score),
            'dist_from_base': float(dist),
            'score_change': float(score - base_score),
        })
        print(f"  alpha={alpha:.2f}: score={score:.6f}, change={score-base_score:.6f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'sensitivity_analysis',
        'experiment_number': 108,
        'timestamp': timestamp,
        'base_score': float(base_score),
        'single_pixel': single_pixel_results,
        'gaussian_noise': gaussian_results,
        'brightness_shift': brightness_results,
        'progressive_corruption': corruption_results,
    }
    output_path = os.path.join(RESULTS_DIR, f"sensitivity_analysis_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
