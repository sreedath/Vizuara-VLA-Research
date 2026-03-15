"""
Embedding Drift Under Input Perturbation.

Measures how much the hidden-state embedding moves (cosine distance from 
original) under systematic image perturbations: brightness, contrast,
Gaussian noise, blur, color jitter, and occlusion.

Key question: How robust are embeddings to input corruption, and which
perturbations move embeddings toward OOD territory?

Experiment 134 in the CalibDrive series.
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
    rng = np.random.default_rng(idx * 27001)
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


def cosine_distance(a, b):
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def apply_brightness(img_pil, factor):
    return ImageEnhance.Brightness(img_pil).enhance(factor)

def apply_contrast(img_pil, factor):
    return ImageEnhance.Contrast(img_pil).enhance(factor)

def apply_gaussian_noise(img_arr, std):
    rng = np.random.default_rng(42)
    noise = rng.normal(0, std, img_arr.shape)
    return np.clip(img_arr.astype(float) + noise, 0, 255).astype(np.uint8)

def apply_blur(img_pil, radius):
    return img_pil.filter(ImageFilter.GaussianBlur(radius=radius))

def apply_occlusion(img_arr, frac):
    """Black out a fraction of the image (center patch)."""
    h, w = img_arr.shape[:2]
    side = int(np.sqrt(frac) * min(h, w))
    y0, x0 = (h - side) // 2, (w - side) // 2
    result = img_arr.copy()
    result[y0:y0+side, x0:x0+side] = 0
    return result

def apply_color_jitter(img_arr, strength):
    """Shift color channels randomly."""
    rng = np.random.default_rng(42)
    shifts = rng.integers(-int(strength), int(strength)+1, (3,))
    result = img_arr.astype(np.int16)
    for c in range(3):
        result[:,:,c] += shifts[c]
    return np.clip(result, 0, 255).astype(np.uint8)


def main():
    print("=" * 70, flush=True)
    print("EMBEDDING DRIFT UNDER INPUT PERTURBATION", flush=True)
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

    # Get reference embeddings (clean highway images)
    print("\n--- Reference embeddings ---", flush=True)
    ref_embeds = []
    for i in range(5):
        img_arr = create_highway(i + 3200)
        img_pil = Image.fromarray(img_arr)
        h = extract_hidden(model, processor, img_pil, prompt)
        if h is not None:
            ref_embeds.append(h)
    ref_centroid = np.mean(ref_embeds, axis=0)
    print(f"  Reference: {len(ref_embeds)} images", flush=True)

    # Also get OOD reference for comparison
    from PIL import Image as PILImage
    rng = np.random.default_rng(99999)
    noise_img = PILImage.fromarray(rng.integers(0, 256, (*SIZE, 3), dtype=np.uint8))
    ood_embed = extract_hidden(model, processor, noise_img, prompt)
    ood_dist = cosine_distance(ood_embed, ref_centroid)
    print(f"  OOD reference distance: {ood_dist:.6f}", flush=True)

    # Define perturbations
    perturbations = {
        'brightness': {
            'levels': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0],
            'fn': lambda arr, pil, level: np.array(apply_brightness(pil, level)),
            'label': 'Brightness factor',
        },
        'contrast': {
            'levels': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0],
            'fn': lambda arr, pil, level: np.array(apply_contrast(pil, level)),
            'label': 'Contrast factor',
        },
        'gaussian_noise': {
            'levels': [5, 10, 20, 30, 50, 75, 100, 128],
            'fn': lambda arr, pil, level: apply_gaussian_noise(arr, level),
            'label': 'Noise std',
        },
        'blur': {
            'levels': [1, 2, 3, 5, 8, 12, 20],
            'fn': lambda arr, pil, level: np.array(apply_blur(pil, level)),
            'label': 'Blur radius',
        },
        'occlusion': {
            'levels': [0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90],
            'fn': lambda arr, pil, level: apply_occlusion(arr, level),
            'label': 'Occlusion fraction',
        },
        'color_jitter': {
            'levels': [5, 10, 20, 40, 60, 80, 100, 128],
            'fn': lambda arr, pil, level: apply_color_jitter(arr, level),
            'label': 'Jitter strength',
        },
    }

    results = {}
    n_images = 3  # Test images per perturbation level

    for pert_name, pert_info in perturbations.items():
        print(f"\n--- {pert_name} ---", flush=True)
        pert_results = []
        for level in pert_info['levels']:
            dists = []
            for i in range(n_images):
                img_arr = create_highway(i + 3210)
                img_pil = Image.fromarray(img_arr)
                perturbed_arr = pert_info['fn'](img_arr, img_pil, level)
                perturbed_pil = Image.fromarray(perturbed_arr)
                h = extract_hidden(model, processor, perturbed_pil, prompt)
                if h is not None:
                    d = cosine_distance(h, ref_centroid)
                    dists.append(d)
            mean_dist = float(np.mean(dists)) if dists else 0
            std_dist = float(np.std(dists)) if dists else 0
            pert_results.append({
                'level': level,
                'mean_distance': mean_dist,
                'std_distance': std_dist,
                'fraction_of_ood': mean_dist / ood_dist if ood_dist > 0 else 0,
            })
            print(f"  level={level}: dist={mean_dist:.6f} ({mean_dist/ood_dist*100:.1f}% of OOD)", flush=True)

        results[pert_name] = {
            'label': pert_info['label'],
            'data': pert_results,
        }

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'embedding_drift',
        'experiment_number': 134,
        'timestamp': timestamp,
        'n_ref': len(ref_embeds),
        'n_images_per_level': n_images,
        'ood_reference_distance': ood_dist,
        'results': results,
    }
    output_path = os.path.join(RESULTS_DIR, f"embedding_drift_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
