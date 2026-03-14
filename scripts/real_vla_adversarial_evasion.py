"""
Adversarial Evasion of OOD Detection.

Tests whether pixel-level perturbations to OOD images can shift their
embeddings toward the ID centroid, evading detection. Uses random
perturbation directions at increasing magnitudes (epsilon) since
true gradient-based attacks require differentiable image processing.

Also tests: Gaussian blur, brightness shift, contrast reduction as
"natural adversarial" transforms that might make OOD look like ID.

Experiment 124 in the CalibDrive series.
"""
import os
import json
import datetime
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageEnhance
from sklearn.metrics import roc_auc_score

RESULTS_DIR = "/workspace/Vizuara-VLA-Research/experiments"
os.makedirs(RESULTS_DIR, exist_ok=True)
SIZE = (256, 256)


def create_highway(idx):
    rng = np.random.default_rng(idx * 17001)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]
    img[SIZE[0]//2:] = [80, 80, 80]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [255, 255, 255]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_urban(idx):
    rng = np.random.default_rng(idx * 17002)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [135, 206, 235]
    img[SIZE[0]//3:SIZE[0]//2] = [139, 119, 101]
    img[SIZE[0]//2:] = [60, 60, 60]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_noise(idx):
    rng = np.random.default_rng(idx * 17003)
    return rng.integers(0, 256, (*SIZE, 3), dtype=np.uint8)

def create_indoor(idx):
    rng = np.random.default_rng(idx * 17004)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:] = [200, 180, 160]
    img[SIZE[0]//2:, :] = [100, 80, 60]
    img[:SIZE[0]//3, SIZE[1]//3:2*SIZE[1]//3] = [150, 200, 255]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_snow(idx):
    rng = np.random.default_rng(idx * 17014)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [200, 200, 210]
    img[SIZE[0]//2:] = [220, 220, 230]
    img[SIZE[0]//2:, SIZE[1]//2-2:SIZE[1]//2+2] = [180, 180, 190]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
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


def perturb_toward_id(ood_arr, id_arr, epsilon):
    """Blend OOD image toward ID image by epsilon (0-1)."""
    blended = (1 - epsilon) * ood_arr.astype(np.float32) + epsilon * id_arr.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def add_gaussian_noise(arr, sigma):
    """Add Gaussian noise."""
    rng = np.random.default_rng(42)
    noisy = arr.astype(np.float32) + rng.normal(0, sigma, arr.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def main():
    print("=" * 70, flush=True)
    print("ADVERSARIAL EVASION OF OOD DETECTION", flush=True)
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

    # Build calibration centroid
    print("\n--- Building ID centroid ---", flush=True)
    cal_embeds = []
    for fn in [create_highway, create_urban]:
        for i in range(8):
            h = extract_hidden(model, processor, Image.fromarray(fn(i + 2400)), prompt)
            if h is not None:
                cal_embeds.append(h)
    centroid = np.mean(cal_embeds, axis=0)
    print(f"  Centroid from {len(cal_embeds)} ID samples", flush=True)

    # ID baseline scores
    id_scores = []
    for fn in [create_highway, create_urban]:
        for i in range(5):
            h = extract_hidden(model, processor, Image.fromarray(fn(i + 2400 + 8)), prompt)
            if h is not None:
                id_scores.append(cosine_dist(h, centroid))
    id_mean = float(np.mean(id_scores))
    id_std = float(np.std(id_scores))
    print(f"  ID scores: {id_mean:.4f} +/- {id_std:.4f}", flush=True)

    # OOD categories to attack
    ood_fns = {
        'noise': create_noise,
        'indoor': create_indoor,
        'snow': create_snow,
    }

    # Reference ID image for blending
    ref_highway = create_highway(9999)

    # Test 1: Pixel blending toward ID
    print("\n--- Pixel Blending Toward ID ---", flush=True)
    epsilons = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    blend_results = {}

    for cat_name, fn in ood_fns.items():
        print(f"\n  {cat_name}:", flush=True)
        eps_scores = {}
        for eps in epsilons:
            scores = []
            for i in range(5):
                ood_arr = fn(i + 2400)
                blended = perturb_toward_id(ood_arr, ref_highway, eps)
                h = extract_hidden(model, processor, Image.fromarray(blended), prompt)
                if h is not None:
                    scores.append(cosine_dist(h, centroid))
            mean_s = float(np.mean(scores))
            print(f"    eps={eps:.1f}: score={mean_s:.4f}", flush=True)
            eps_scores[str(eps)] = {
                'mean': mean_s,
                'std': float(np.std(scores)),
                'scores': scores,
            }
        blend_results[cat_name] = eps_scores

    # Test 2: Natural transforms on OOD images
    print("\n--- Natural Adversarial Transforms ---", flush=True)
    transform_results = {}

    for cat_name, fn in ood_fns.items():
        print(f"\n  {cat_name}:", flush=True)
        cat_transforms = {}

        for i in range(5):
            ood_arr = fn(i + 2400)
            ood_img = Image.fromarray(ood_arr)

            # Original
            h = extract_hidden(model, processor, ood_img, prompt)
            if h is not None:
                cat_transforms.setdefault('original', []).append(cosine_dist(h, centroid))

            # Gaussian blur (r=2, 5, 10)
            for r in [2, 5, 10]:
                blurred = ood_img.filter(ImageFilter.GaussianBlur(radius=r))
                h = extract_hidden(model, processor, blurred, prompt)
                if h is not None:
                    cat_transforms.setdefault(f'blur_r{r}', []).append(cosine_dist(h, centroid))

            # Brightness (0.5=darker, 1.5=brighter)
            for b in [0.5, 1.5, 2.0]:
                bright = ImageEnhance.Brightness(ood_img).enhance(b)
                h = extract_hidden(model, processor, bright, prompt)
                if h is not None:
                    cat_transforms.setdefault(f'brightness_{b}', []).append(cosine_dist(h, centroid))

            # Contrast reduction
            for c in [0.3, 0.1]:
                contrast = ImageEnhance.Contrast(ood_img).enhance(c)
                h = extract_hidden(model, processor, contrast, prompt)
                if h is not None:
                    cat_transforms.setdefault(f'contrast_{c}', []).append(cosine_dist(h, centroid))

        # Print results
        for t_name, scores in cat_transforms.items():
            mean_s = float(np.mean(scores))
            print(f"    {t_name:20s}: score={mean_s:.4f}", flush=True)
        transform_results[cat_name] = {
            k: {'mean': float(np.mean(v)), 'std': float(np.std(v))}
            for k, v in cat_transforms.items()
        }

    # Find evasion threshold
    print("\n--- Evasion Analysis ---", flush=True)
    threshold = id_mean + 3 * id_std  # 3-sigma threshold
    print(f"Detection threshold (3-sigma): {threshold:.4f}", flush=True)

    for cat_name in ood_fns:
        print(f"\n  {cat_name}:", flush=True)
        # Find epsilon where blend score crosses threshold
        crossed = False
        for eps in epsilons:
            if blend_results[cat_name][str(eps)]['mean'] < threshold:
                print(f"    Crosses threshold at eps={eps:.1f} (score={blend_results[cat_name][str(eps)]['mean']:.4f})", flush=True)
                crossed = True
                break
        if not crossed:
            print(f"    Never crosses threshold (min score at eps=1.0: {blend_results[cat_name]['1.0']['mean']:.4f})", flush=True)

        # Check natural transforms
        min_transform = None
        min_score = float('inf')
        for t_name, t_data in transform_results[cat_name].items():
            if t_data['mean'] < min_score:
                min_score = t_data['mean']
                min_transform = t_name
        evaded = min_score < threshold
        status = "EVADES" if evaded else "does not evade"
        print(f"    Best natural transform: {min_transform} (score={min_score:.4f}) — {status}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'adversarial_evasion',
        'experiment_number': 124,
        'timestamp': timestamp,
        'id_baseline': {'mean': id_mean, 'std': id_std},
        'detection_threshold_3sigma': float(threshold),
        'blend_results': {
            cat: {eps: {'mean': v['mean'], 'std': v['std']} for eps, v in eps_data.items()}
            for cat, eps_data in blend_results.items()
        },
        'transform_results': transform_results,
    }
    output_path = os.path.join(RESULTS_DIR, f"adversarial_evasion_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
