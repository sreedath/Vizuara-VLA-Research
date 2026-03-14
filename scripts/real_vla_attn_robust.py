"""
Attention-Based Detection Robustness Under Perturbation.

Tests whether the calibration-free attention OOD detection is robust
to image perturbations (blur, brightness, JPEG, Gaussian noise).
Compares attention-based detection against cosine distance under
the same perturbations.

Experiment 65 in the CalibDrive series.
"""
import os
import json
import datetime
import numpy as np
import torch
from PIL import Image, ImageFilter
from io import BytesIO
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

def create_noise_img(idx):
    rng = np.random.default_rng(idx * 5003)
    return rng.integers(0, 256, (*SIZE, 3), dtype=np.uint8)

def create_indoor(idx):
    rng = np.random.default_rng(idx * 5004)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:] = [200, 180, 160]
    img[SIZE[0]//2:] = [139, 90, 43]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_blackout(idx):
    return np.zeros((*SIZE, 3), dtype=np.uint8)


def apply_blur(img_arr, radius):
    return np.array(Image.fromarray(img_arr).filter(ImageFilter.GaussianBlur(radius=radius)))

def apply_brightness(img_arr, factor):
    return np.clip(img_arr.astype(np.float32) * factor, 0, 255).astype(np.uint8)

def apply_jpeg(img_arr, quality):
    pil = Image.fromarray(img_arr)
    buf = BytesIO()
    pil.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf))

def apply_gaussian_noise(img_arr, sigma):
    rng = np.random.default_rng(42)
    noise = rng.normal(0, sigma, img_arr.shape)
    return np.clip(img_arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def extract_signals(model, processor, image, prompt):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)

    # Forward pass for attention
    with torch.no_grad():
        out = model(**inputs, output_attentions=True, output_hidden_states=True)

    result = {}

    # Attention from last layer
    if hasattr(out, 'attentions') and out.attentions:
        attn = out.attentions[-1][0].float().cpu().numpy()
        n_heads = attn.shape[0]
        last_attn = attn[:, -1, :]
        max_attns = [float(np.max(last_attn[h])) for h in range(n_heads)]
        entropies = [float(-np.sum((last_attn[h]+1e-10) * np.log(last_attn[h]+1e-10)))
                     for h in range(n_heads)]
        result['attn_max'] = float(np.mean(max_attns))
        result['attn_entropy'] = float(np.mean(entropies))

    # Hidden state
    if hasattr(out, 'hidden_states') and out.hidden_states:
        result['hidden'] = out.hidden_states[-1][0, -1, :].float().cpu().numpy()

    return result


def cosine_dist(a, b):
    return 1.0 - float(np.dot(a / (np.linalg.norm(a) + 1e-10),
                               b / (np.linalg.norm(b) + 1e-10)))


def main():
    print("=" * 70, flush=True)
    print("ATTENTION ROBUSTNESS UNDER PERTURBATION", flush=True)
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

    # Calibration for cosine baseline
    print("\nCalibrating (cosine baseline)...", flush=True)
    cal_hidden = []
    for i in range(10):
        sig = extract_signals(model, processor,
                              Image.fromarray(create_highway(i + 9000)), prompt)
        if 'hidden' in sig:
            cal_hidden.append(sig['hidden'])
    centroid = np.mean(cal_hidden, axis=0) if cal_hidden else None
    print(f"  Calibration: {len(cal_hidden)} samples", flush=True)

    # Perturbation experiments
    perturbations = {
        'none': lambda img: img,
        'blur_1': lambda img: apply_blur(img, 1),
        'blur_3': lambda img: apply_blur(img, 3),
        'blur_5': lambda img: apply_blur(img, 5),
        'bright_0.5': lambda img: apply_brightness(img, 0.5),
        'bright_1.5': lambda img: apply_brightness(img, 1.5),
        'bright_2.0': lambda img: apply_brightness(img, 2.0),
        'jpeg_10': lambda img: apply_jpeg(img, 10),
        'jpeg_50': lambda img: apply_jpeg(img, 50),
        'noise_25': lambda img: apply_gaussian_noise(img, 25),
        'noise_50': lambda img: apply_gaussian_noise(img, 50),
    }

    n_id = 6
    n_ood = 4  # Per OOD type
    ood_fns = {
        'noise': create_noise_img,
        'indoor': create_indoor,
        'blackout': create_blackout,
    }

    results = {}
    cnt = 0
    total = len(perturbations) * (n_id + len(ood_fns) * n_ood)

    for pert_name, pert_fn in perturbations.items():
        print(f"\n  Perturbation: {pert_name}", flush=True)
        data = []

        # ID samples (perturbed)
        for i in range(n_id):
            cnt += 1
            img_arr = pert_fn(create_highway(i + 400))
            sig = extract_signals(model, processor, Image.fromarray(img_arr), prompt)
            sig['is_ood'] = False
            data.append(sig)

        # OOD samples (unperturbed — OOD is OOD regardless)
        for ood_name, ood_fn in ood_fns.items():
            for i in range(n_ood):
                cnt += 1
                img_arr = ood_fn(i + 400)
                sig = extract_signals(model, processor, Image.fromarray(img_arr), prompt)
                sig['is_ood'] = True
                data.append(sig)

        if cnt % 20 == 0:
            print(f"    [{cnt}/{total}]", flush=True)

        labels = [1 if d['is_ood'] else 0 for d in data]

        # Attention-based detection (calibration-free)
        attn_max_auroc = None
        attn_ent_auroc = None
        if all(d.get('attn_max') is not None for d in data):
            attn_max_auroc = roc_auc_score(labels, [d['attn_max'] for d in data])
            attn_ent_auroc = roc_auc_score(labels, [-d['attn_entropy'] for d in data])

        # Cosine-based detection (calibrated)
        cos_auroc = None
        if centroid is not None and all('hidden' in d for d in data):
            cos_scores = [cosine_dist(d['hidden'], centroid) for d in data]
            cos_auroc = roc_auc_score(labels, cos_scores)

        results[pert_name] = {
            'attn_max': float(attn_max_auroc) if attn_max_auroc else None,
            'attn_entropy': float(attn_ent_auroc) if attn_ent_auroc else None,
            'cosine': float(cos_auroc) if cos_auroc else None,
        }

        print(f"    attn_max={attn_max_auroc:.3f}, attn_ent={attn_ent_auroc:.3f}, "
              f"cosine={cos_auroc:.3f}" if all(v is not None for v in [attn_max_auroc, attn_ent_auroc, cos_auroc])
              else f"    Some signals missing", flush=True)

    # Summary
    print("\n" + "=" * 70, flush=True)
    print("ROBUSTNESS SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"\n  {'Perturbation':<15} {'Attn Max':>10} {'Attn Ent':>10} {'Cosine':>10}", flush=True)
    print("  " + "-" * 47, flush=True)
    for pert_name, r in results.items():
        am = f"{r['attn_max']:.3f}" if r['attn_max'] else "N/A"
        ae = f"{r['attn_entropy']:.3f}" if r['attn_entropy'] else "N/A"
        co = f"{r['cosine']:.3f}" if r['cosine'] else "N/A"
        print(f"  {pert_name:<15} {am:>10} {ae:>10} {co:>10}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'attn_robust',
        'experiment_number': 65,
        'timestamp': timestamp,
        'n_id': n_id,
        'n_ood_per_type': n_ood,
        'n_ood_types': len(ood_fns),
        'total_inferences': cnt,
        'results': results,
    }
    output_path = os.path.join(RESULTS_DIR, f"attn_robust_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
