"""
Adversarial Perturbation Robustness.

Tests whether small adversarial-like perturbations (Gaussian noise,
salt-and-pepper, JPEG compression artifacts, Gaussian blur) can fool
the cosine distance OOD detector into misclassifying perturbed ID
images as OOD, or perturbed OOD images as ID.

Experiment 81 in the CalibDrive series.
"""
import os
import json
import datetime
import numpy as np
import torch
from PIL import Image, ImageFilter
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


def perturb_gaussian(img_arr, sigma):
    """Add Gaussian noise with given sigma."""
    rng = np.random.default_rng(42)
    noise = rng.normal(0, sigma, img_arr.shape)
    return np.clip(img_arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def perturb_salt_pepper(img_arr, prob):
    """Apply salt-and-pepper noise."""
    rng = np.random.default_rng(43)
    out = img_arr.copy()
    mask = rng.random(img_arr.shape[:2])
    out[mask < prob/2] = 0
    out[mask > 1 - prob/2] = 255
    return out

def perturb_jpeg(img_arr, quality):
    """Simulate JPEG compression artifacts."""
    import io
    pil = Image.fromarray(img_arr)
    buf = io.BytesIO()
    pil.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf))

def perturb_blur(img_arr, radius):
    """Apply Gaussian blur."""
    pil = Image.fromarray(img_arr)
    blurred = pil.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.array(blurred)


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
    print("ADVERSARIAL PERTURBATION ROBUSTNESS", flush=True)
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

    # Perturbation types and levels
    perturbations = {
        'gaussian_10': lambda img: perturb_gaussian(img, 10),
        'gaussian_25': lambda img: perturb_gaussian(img, 25),
        'gaussian_50': lambda img: perturb_gaussian(img, 50),
        'gaussian_100': lambda img: perturb_gaussian(img, 100),
        'salt_pepper_0.01': lambda img: perturb_salt_pepper(img, 0.01),
        'salt_pepper_0.05': lambda img: perturb_salt_pepper(img, 0.05),
        'salt_pepper_0.10': lambda img: perturb_salt_pepper(img, 0.10),
        'salt_pepper_0.20': lambda img: perturb_salt_pepper(img, 0.20),
        'jpeg_50': lambda img: perturb_jpeg(img, 50),
        'jpeg_20': lambda img: perturb_jpeg(img, 20),
        'jpeg_5': lambda img: perturb_jpeg(img, 5),
        'blur_1': lambda img: perturb_blur(img, 1),
        'blur_3': lambda img: perturb_blur(img, 3),
        'blur_5': lambda img: perturb_blur(img, 5),
    }

    # Clean OOD reference
    print("\nCollecting clean OOD reference...", flush=True)
    ood_hidden = []
    for fn in [create_noise, create_indoor, create_twilight_highway]:
        for i in range(6):
            h = extract_hidden(model, processor,
                               Image.fromarray(fn(i + 500)), prompt)
            if h is not None:
                ood_hidden.append(h)
    print(f"  {len(ood_hidden)} OOD samples", flush=True)
    ood_scores = [cosine_dist(h, centroid) for h in ood_hidden]

    # Clean ID reference
    print("\nCollecting clean ID reference...", flush=True)
    clean_id_hidden = []
    for fn in [create_highway, create_urban]:
        for i in range(8):
            h = extract_hidden(model, processor,
                               Image.fromarray(fn(i + 500)), prompt)
            if h is not None:
                clean_id_hidden.append(h)
    print(f"  {len(clean_id_hidden)} clean ID samples", flush=True)

    # Test each perturbation on ID images
    print("\nTesting perturbations on ID images...", flush=True)
    results = {}
    cnt = 0
    total = len(perturbations)

    for name, perturb_fn in perturbations.items():
        cnt += 1
        perturbed_hidden = []
        for fn in [create_highway, create_urban]:
            for i in range(6):
                base_img = fn(i + 600)
                perturbed_img = perturb_fn(base_img)
                h = extract_hidden(model, processor,
                                   Image.fromarray(perturbed_img), prompt)
                if h is not None:
                    perturbed_hidden.append(h)

        perturbed_scores = [cosine_dist(h, centroid) for h in perturbed_hidden]

        # AUROC: perturbed ID (label=0) vs clean OOD (label=1)
        labels = [0]*len(perturbed_hidden) + [1]*len(ood_hidden)
        scores = perturbed_scores + ood_scores
        auroc = roc_auc_score(labels, scores)

        # Mean distances
        pert_mean = float(np.mean(perturbed_scores))
        clean_mean = float(np.mean([cosine_dist(h, centroid) for h in clean_id_hidden]))
        ood_mean = float(np.mean(ood_scores))

        # Drift from clean ID
        drift = pert_mean - clean_mean

        results[name] = {
            'auroc': float(auroc),
            'perturbed_id_mean': pert_mean,
            'clean_id_mean': clean_mean,
            'ood_mean': ood_mean,
            'drift_from_clean': drift,
            'n_perturbed': len(perturbed_hidden),
        }
        print(f"  [{cnt}/{total}] {name:<20}: AUROC={auroc:.3f}, "
              f"pert_dist={pert_mean:.4f}, drift={drift:+.4f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'adversarial_robustness',
        'experiment_number': 81,
        'timestamp': timestamp,
        'n_cal': len(cal_hidden),
        'n_ood': len(ood_hidden),
        'n_clean_id': len(clean_id_hidden),
        'results': results,
    }
    output_path = os.path.join(RESULTS_DIR, f"adversarial_robustness_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)

    # Summary
    print("\n" + "=" * 70, flush=True)
    print("ADVERSARIAL ROBUSTNESS SUMMARY", flush=True)
    print("=" * 70, flush=True)
    for name, r in sorted(results.items(), key=lambda x: x[1]['auroc']):
        status = "ROBUST" if r['auroc'] >= 0.95 else "WEAK" if r['auroc'] >= 0.8 else "BROKEN"
        print(f"  {name:<20}: AUROC={r['auroc']:.3f} [{status}]", flush=True)


if __name__ == "__main__":
    main()
