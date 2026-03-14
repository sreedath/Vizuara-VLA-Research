"""
OOD Hardness Spectrum Analysis.

Creates a fine-grained hardness spectrum by generating OOD images
at varying "distances" from ID: gradually transitioning from ID-like
to fully OOD scenarios. Measures the detection boundary.

Experiment 88 in the CalibDrive series.
"""
import os
import json
import datetime
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

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

def create_urban(idx):
    rng = np.random.default_rng(idx * 5002)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [135, 206, 235]
    img[SIZE[0]//3:SIZE[0]//2] = [139, 119, 101]
    img[SIZE[0]//2:] = [60, 60, 60]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def create_interpolated(idx, alpha):
    """Interpolate between highway (ID) and indoor (OOD).
    alpha=0: pure highway, alpha=1: pure indoor.
    """
    rng = np.random.default_rng(idx * 7001)
    # Highway base
    hw = np.zeros((*SIZE, 3), dtype=np.float32)
    hw[:SIZE[0]//2] = [135, 206, 235]
    hw[SIZE[0]//2:] = [80, 80, 80]
    hw[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [255, 255, 255]

    # Indoor base
    indoor = np.zeros((*SIZE, 3), dtype=np.float32)
    indoor[:] = [200, 180, 160]
    indoor[SIZE[0]//2:, :] = [100, 80, 60]
    indoor[:SIZE[0]//3, SIZE[1]//3:2*SIZE[1]//3] = [150, 200, 255]

    # Interpolate
    img = (1 - alpha) * hw + alpha * indoor
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def create_color_shifted(idx, shift_amount):
    """Shift highway colors by given amount (hue rotation approximation)."""
    rng = np.random.default_rng(idx * 7002)
    img = np.zeros((*SIZE, 3), dtype=np.float32)
    # Base highway with shifted sky color
    sky_r = 135 + shift_amount * 120  # toward red
    sky_g = max(0, 206 - shift_amount * 150)
    sky_b = max(0, 235 - shift_amount * 200)
    img[:SIZE[0]//2] = [sky_r, sky_g, sky_b]
    img[SIZE[0]//2:] = [80, 80, 80]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [255, 255, 255]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def extract_hidden(model, processor, image, prompt):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    if hasattr(fwd, 'hidden_states') and fwd.hidden_states:
        return fwd.hidden_states[-1][0, -1, :].float().cpu().numpy()
    return None


def cosine_dist(a, b):
    return 1.0 - float(np.dot(a / (np.linalg.norm(a) + 1e-10),
                               b / (np.linalg.norm(b) + 1e-10)))


def main():
    print("=" * 70, flush=True)
    print("OOD HARDNESS SPECTRUM", flush=True)
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

    # Calibrate
    print("\nCalibrating...", flush=True)
    cal_hidden = []
    for fn in [create_highway, create_urban]:
        for i in range(15):
            h = extract_hidden(model, processor,
                               Image.fromarray(fn(i + 9000)), prompt)
            if h is not None:
                cal_hidden.append(h)
    centroid = np.mean(cal_hidden, axis=0)
    print(f"  {len(cal_hidden)} calibration samples", flush=True)

    # ID reference
    id_scores = []
    for fn in [create_highway, create_urban]:
        for i in range(8):
            h = extract_hidden(model, processor,
                               Image.fromarray(fn(i + 500)), prompt)
            if h is not None:
                id_scores.append(cosine_dist(h, centroid))

    # Test 1: Highway-Indoor interpolation
    print("\nInterpolation spectrum...", flush=True)
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    interp_results = {}
    for alpha in alphas:
        scores = []
        for i in range(6):
            h = extract_hidden(model, processor,
                               Image.fromarray(create_interpolated(i + 500, alpha)), prompt)
            if h is not None:
                scores.append(cosine_dist(h, centroid))

        labels = [0]*len(id_scores) + [1]*len(scores)
        all_scores = id_scores + scores
        auroc = roc_auc_score(labels, all_scores) if len(set(labels)) > 1 else 1.0

        interp_results[f"{alpha:.1f}"] = {
            'auroc': float(auroc),
            'mean_dist': float(np.mean(scores)),
            'std_dist': float(np.std(scores)),
        }
        print(f"  alpha={alpha:.1f}: dist={np.mean(scores):.4f}, AUROC={auroc:.3f}", flush=True)

    # Test 2: Color shift spectrum
    print("\nColor shift spectrum...", flush=True)
    shifts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    color_results = {}
    for shift in shifts:
        scores = []
        for i in range(6):
            h = extract_hidden(model, processor,
                               Image.fromarray(create_color_shifted(i + 500, shift)), prompt)
            if h is not None:
                scores.append(cosine_dist(h, centroid))

        labels = [0]*len(id_scores) + [1]*len(scores)
        all_scores = id_scores + scores
        auroc = roc_auc_score(labels, all_scores)

        color_results[f"{shift:.1f}"] = {
            'auroc': float(auroc),
            'mean_dist': float(np.mean(scores)),
            'std_dist': float(np.std(scores)),
        }
        print(f"  shift={shift:.1f}: dist={np.mean(scores):.4f}, AUROC={auroc:.3f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'ood_hardness',
        'experiment_number': 88,
        'timestamp': timestamp,
        'n_cal': len(cal_hidden),
        'n_id_ref': len(id_scores),
        'id_mean_dist': float(np.mean(id_scores)),
        'interpolation_results': interp_results,
        'color_shift_results': color_results,
    }
    output_path = os.path.join(RESULTS_DIR, f"ood_hardness_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
