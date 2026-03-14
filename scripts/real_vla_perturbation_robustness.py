"""
Input Perturbation Robustness Test.

Tests whether the OOD detection pipeline is robust to common image
perturbations that should NOT trigger OOD detection:
- Gaussian blur
- Brightness changes
- JPEG compression artifacts
- Gaussian noise

If these perturbations cause false OOD detections, it undermines
the pipeline's practical utility.

Also tests whether perturbations applied to OOD images make them
harder or easier to detect.

Experiment 53 in the CalibDrive series.
"""
import os
import json
import datetime
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageEnhance
from sklearn.metrics import roc_auc_score
import io

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
    img[SIZE[0]//2:] = [139, 90, 43]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


# Perturbation functions
def perturb_blur(img_arr, level):
    """Gaussian blur with radius = level."""
    img = Image.fromarray(img_arr)
    return np.array(img.filter(ImageFilter.GaussianBlur(radius=level)))

def perturb_brightness(img_arr, factor):
    """Brightness adjustment. factor=1.0 is original."""
    img = Image.fromarray(img_arr)
    enhancer = ImageEnhance.Brightness(img)
    return np.array(enhancer.enhance(factor))

def perturb_jpeg(img_arr, quality):
    """JPEG compression at given quality level."""
    img = Image.fromarray(img_arr)
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    return np.array(Image.open(buffer))

def perturb_gauss_noise(img_arr, std):
    """Add Gaussian noise with given std."""
    rng = np.random.default_rng(42)
    noise = rng.normal(0, std, img_arr.shape)
    return np.clip(img_arr.astype(float) + noise, 0, 255).astype(np.uint8)


def extract_signals(model, processor, image, prompt):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=7, do_sample=False,
            output_scores=True, output_hidden_states=True,
            return_dict_in_generate=True,
        )
    if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
        last_step = outputs.hidden_states[-1]
        if isinstance(last_step, tuple):
            hidden = last_step[-1][0, -1, :].float().cpu().numpy()
        else:
            hidden = last_step[0, -1, :].float().cpu().numpy()
    else:
        hidden = np.zeros(4096)

    vocab_size = outputs.scores[0].shape[-1]
    action_start = vocab_size - 256
    masses = []
    for score in outputs.scores[:7]:
        probs = torch.softmax(score[0].float(), dim=0)
        masses.append(float(probs[action_start:].sum()))

    return {
        'hidden': hidden,
        'action_mass': float(np.mean(masses)),
    }


def cosine_dist(a, b):
    return 1.0 - float(np.dot(a / (np.linalg.norm(a) + 1e-10),
                               b / (np.linalg.norm(b) + 1e-10)))


def main():
    print("=" * 70, flush=True)
    print("INPUT PERTURBATION ROBUSTNESS", flush=True)
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

    # Calibration (clean images only)
    print("\nCalibration (clean images)...", flush=True)
    cal_hidden = []
    for fn in [create_highway, create_urban]:
        for i in range(15):
            data = extract_signals(model, processor,
                                    Image.fromarray(fn(i + 9000)), prompt)
            cal_hidden.append(data['hidden'])
    centroid = np.mean(cal_hidden, axis=0)
    cal_cos = [cosine_dist(h, centroid) for h in cal_hidden]
    print(f"  Centroid computed from {len(cal_hidden)} samples", flush=True)
    print(f"  Cal cosine: mean={np.mean(cal_cos):.4f}, max={np.max(cal_cos):.4f}", flush=True)

    # Define perturbation levels
    perturbations = {
        'blur': {
            'fn': perturb_blur,
            'levels': [0, 1, 2, 3, 5],
            'level_names': ['None', 'r=1', 'r=2', 'r=3', 'r=5'],
        },
        'brightness': {
            'fn': perturb_brightness,
            'levels': [1.0, 0.5, 0.3, 1.5, 2.0],
            'level_names': ['1.0x', '0.5x', '0.3x', '1.5x', '2.0x'],
        },
        'jpeg': {
            'fn': perturb_jpeg,
            'levels': [95, 50, 20, 5, 1],
            'level_names': ['q95', 'q50', 'q20', 'q5', 'q1'],
        },
        'gauss_noise': {
            'fn': perturb_gauss_noise,
            'levels': [0, 10, 25, 50, 100],
            'level_names': ['σ=0', 'σ=10', 'σ=25', 'σ=50', 'σ=100'],
        },
    }

    # Test images
    id_images = []
    for fn in [create_highway, create_urban]:
        for i in range(5):
            id_images.append(fn(i + 200))

    ood_images = []
    for fn in [create_noise, create_indoor]:
        for i in range(5):
            ood_images.append(fn(i + 200))

    # Run all perturbation experiments
    print("\nRunning perturbation experiments...", flush=True)
    all_results = {}
    cnt = 0
    total = len(perturbations) * (len(id_images) + len(ood_images)) * 5  # approx

    for perturb_name, perturb_config in perturbations.items():
        print(f"\n  Perturbation: {perturb_name}", flush=True)
        perturb_fn = perturb_config['fn']
        levels = perturb_config['levels']
        level_names = perturb_config['level_names']

        level_results = {}
        for level, level_name in zip(levels, level_names):
            id_cos = []
            ood_cos = []
            id_mass = []
            ood_mass = []

            # ID images with perturbation
            for img_arr in id_images:
                cnt += 1
                if level == 0 and perturb_name in ['blur', 'gauss_noise']:
                    perturbed = img_arr
                elif level == 1.0 and perturb_name == 'brightness':
                    perturbed = img_arr
                else:
                    perturbed = perturb_fn(img_arr, level)
                data = extract_signals(model, processor,
                                        Image.fromarray(perturbed), prompt)
                cos = cosine_dist(data['hidden'], centroid)
                id_cos.append(cos)
                id_mass.append(data['action_mass'])

            # OOD images with perturbation
            for img_arr in ood_images:
                cnt += 1
                if level == 0 and perturb_name in ['blur', 'gauss_noise']:
                    perturbed = img_arr
                elif level == 1.0 and perturb_name == 'brightness':
                    perturbed = img_arr
                else:
                    perturbed = perturb_fn(img_arr, level)
                data = extract_signals(model, processor,
                                        Image.fromarray(perturbed), prompt)
                cos = cosine_dist(data['hidden'], centroid)
                ood_cos.append(cos)
                ood_mass.append(data['action_mass'])

            # Compute AUROC
            labels = [0]*len(id_cos) + [1]*len(ood_cos)
            scores = id_cos + ood_cos
            try:
                auroc = roc_auc_score(labels, scores)
            except ValueError:
                auroc = 0.5

            level_results[level_name] = {
                'auroc': auroc,
                'id_cos_mean': float(np.mean(id_cos)),
                'id_cos_std': float(np.std(id_cos)),
                'ood_cos_mean': float(np.mean(ood_cos)),
                'ood_cos_std': float(np.std(ood_cos)),
                'id_mass_mean': float(np.mean(id_mass)),
                'ood_mass_mean': float(np.mean(ood_mass)),
            }

            print(f"    {level_name}: AUROC={auroc:.3f}, "
                  f"ID cos={np.mean(id_cos):.4f}±{np.std(id_cos):.4f}, "
                  f"OOD cos={np.mean(ood_cos):.4f}±{np.std(ood_cos):.4f}", flush=True)

        all_results[perturb_name] = level_results

    # ===================================================================
    # Analysis
    # ===================================================================
    print("\n" + "=" * 70, flush=True)
    print("ROBUSTNESS SUMMARY", flush=True)
    print("=" * 70, flush=True)

    print(f"\n  {'Perturbation':<15}", end='', flush=True)
    for perturb_name in perturbations:
        levels = perturbations[perturb_name]['level_names']
        for ln in levels:
            print(f" {ln:>8}", end='', flush=True)
    print(flush=True)

    print(f"  {'':15}", end='', flush=True)
    for perturb_name in perturbations:
        for ln in perturbations[perturb_name]['level_names']:
            auroc = all_results[perturb_name][ln]['auroc']
            print(f" {auroc:>8.3f}", end='', flush=True)
    print(flush=True)

    # Simplified summary
    print("\n  Per-perturbation AUROC range:", flush=True)
    for perturb_name in perturbations:
        aurocs = [all_results[perturb_name][ln]['auroc']
                  for ln in perturbations[perturb_name]['level_names']]
        print(f"    {perturb_name:<15}: {min(aurocs):.3f} - {max(aurocs):.3f} "
              f"(range={max(aurocs)-min(aurocs):.3f})", flush=True)

    # ID false positive analysis
    print("\n  ID false positive (cosine distance > cal_max):", flush=True)
    cal_max = max(cal_cos)
    for perturb_name in perturbations:
        for ln in perturbations[perturb_name]['level_names']:
            id_mean = all_results[perturb_name][ln]['id_cos_mean']
            id_std = all_results[perturb_name][ln]['id_cos_std']
            above = id_mean + 2*id_std > cal_max
            if above:
                print(f"    WARNING: {perturb_name} {ln}: ID cos mean+2std = "
                      f"{id_mean + 2*id_std:.4f} > cal_max {cal_max:.4f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'perturbation_robustness',
        'experiment_number': 53,
        'timestamp': timestamp,
        'n_cal': len(cal_hidden),
        'n_id_test': len(id_images),
        'n_ood_test': len(ood_images),
        'cal_cos_max': float(cal_max),
        'results': all_results,
    }
    output_path = os.path.join(RESULTS_DIR, f"perturbation_robustness_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
