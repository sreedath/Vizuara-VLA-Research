"""
Temporal Stability of Calibration.

Tests whether the calibration centroid remains effective when test
distributions shift gradually. Simulates temporal drift by applying
progressive brightness, contrast, and color temperature changes to
ID images, then measures detection stability.

Experiment 80 in the CalibDrive series.
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

def create_noise(idx):
    rng = np.random.default_rng(idx * 5003)
    return rng.integers(0, 256, (*SIZE, 3), dtype=np.uint8)

def create_indoor(idx):
    rng = np.random.default_rng(idx * 5004)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:] = [200, 180, 160]
    img[SIZE[0]//2:, :] = [100, 80, 60]
    img[:SIZE[0]//3, SIZE[1]//3:2*SIZE[1]//3] = [150, 200, 255]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_twilight_highway(idx):
    rng = np.random.default_rng(idx * 5010)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [70, 50, 80]
    img[SIZE[0]//2:] = [60, 60, 60]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [200, 200, 100]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def apply_drift(img_arr, drift_level):
    """Apply progressive drift to an image.

    drift_level: 0.0 = no change, 1.0 = maximum drift
    Combines brightness shift, contrast reduction, and warm color cast.
    """
    img = img_arr.astype(np.float32)

    # Brightness shift (darken progressively)
    brightness_factor = 1.0 - 0.4 * drift_level
    img = img * brightness_factor

    # Contrast reduction
    mean = img.mean()
    contrast_factor = 1.0 - 0.3 * drift_level
    img = mean + (img - mean) * contrast_factor

    # Warm color temperature shift (more red, less blue)
    img[:, :, 0] = img[:, :, 0] * (1.0 + 0.15 * drift_level)  # R up
    img[:, :, 2] = img[:, :, 2] * (1.0 - 0.15 * drift_level)  # B down

    return np.clip(img, 0, 255).astype(np.uint8)


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
    print("TEMPORAL STABILITY OF CALIBRATION", flush=True)
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

    # Calibrate on clean images (time=0)
    print("\nCalibrating on clean images...", flush=True)
    cal_hidden = []
    for fn in [create_highway, create_urban]:
        for i in range(15):
            h = extract_hidden(model, processor,
                               Image.fromarray(fn(i + 9000)), prompt)
            if h is not None:
                cal_hidden.append(h)
    centroid = np.mean(cal_hidden, axis=0)
    print(f"  {len(cal_hidden)} calibration samples", flush=True)

    # Drift levels simulate progression
    drift_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # OOD reference data (no drift applied to OOD)
    print("\nCollecting OOD reference...", flush=True)
    ood_hidden = []
    ood_fns = [create_noise, create_indoor, create_twilight_highway]
    for fn in ood_fns:
        for i in range(6):
            h = extract_hidden(model, processor,
                               Image.fromarray(fn(i + 500)), prompt)
            if h is not None:
                ood_hidden.append(h)
    print(f"  {len(ood_hidden)} OOD samples", flush=True)

    # Test at each drift level
    print("\nTesting drift levels...", flush=True)
    results = {}
    total = len(drift_levels)

    for di, drift in enumerate(drift_levels):
        # Generate drifted ID images
        id_drifted = []
        for fn in [create_highway, create_urban]:
            for i in range(8):
                base = fn(i + 500)
                drifted = apply_drift(base, drift)
                h = extract_hidden(model, processor,
                                   Image.fromarray(drifted), prompt)
                if h is not None:
                    id_drifted.append(h)

        # Compute scores
        id_scores = [cosine_dist(h, centroid) for h in id_drifted]
        ood_scores = [cosine_dist(h, centroid) for h in ood_hidden]

        id_mean = float(np.mean(id_scores))
        ood_mean = float(np.mean(ood_scores))

        labels = [0]*len(id_drifted) + [1]*len(ood_hidden)
        scores = id_scores + ood_scores
        auroc = roc_auc_score(labels, scores)

        # Cohen's d
        id_arr = np.array(id_scores)
        ood_arr = np.array(ood_scores)
        pooled_std = np.sqrt((id_arr.var() + ood_arr.var()) / 2)
        cohens_d = (ood_mean - id_mean) / (pooled_std + 1e-10)

        # Centroid distance: how far did drifted ID move from centroid?
        centroid_drift = float(np.mean([cosine_dist(h, centroid) for h in id_drifted]))

        results[f"{drift:.1f}"] = {
            'auroc': float(auroc),
            'cohens_d': float(cohens_d),
            'id_mean_dist': id_mean,
            'ood_mean_dist': ood_mean,
            'centroid_drift': centroid_drift,
            'n_id': len(id_drifted),
        }
        print(f"  [{di+1}/{total}] drift={drift:.1f}: AUROC={auroc:.3f}, d={cohens_d:.2f}, "
              f"ID_dist={id_mean:.4f}, centroid_drift={centroid_drift:.4f}", flush=True)

    # Find critical drift point (first AUROC < 0.95)
    critical_drift = None
    for drift_str, r in results.items():
        if r['auroc'] < 0.95:
            critical_drift = float(drift_str)
            break

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'temporal_stability',
        'experiment_number': 80,
        'timestamp': timestamp,
        'n_cal': len(cal_hidden),
        'n_ood': len(ood_hidden),
        'drift_levels': drift_levels,
        'critical_drift': critical_drift,
        'results': results,
    }
    output_path = os.path.join(RESULTS_DIR, f"temporal_stability_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)

    # Summary
    print("\n" + "=" * 70, flush=True)
    print("TEMPORAL STABILITY SUMMARY", flush=True)
    print("=" * 70, flush=True)
    for drift_str, r in results.items():
        status = "OK" if r['auroc'] >= 0.95 else "DEGRADED"
        print(f"  drift={drift_str}: AUROC={r['auroc']:.3f} [{status}]", flush=True)
    if critical_drift is not None:
        print(f"\n  Critical drift point: {critical_drift}", flush=True)
    else:
        print(f"\n  No critical drift point — stable across all levels!", flush=True)


if __name__ == "__main__":
    main()
