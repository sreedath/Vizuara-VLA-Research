"""
Action Mass Under Temperature Scaling on Real OpenVLA-7B.

Tests whether temperature scaling can amplify the action mass signal:
1. At low T, action mass should concentrate further on top bins
2. At high T, action mass may leak more to non-action tokens
3. Does temperature reveal a sweet spot for action mass AUROC?
4. Combine: temperature-scaled action mass + raw action mass

Also tests action mass stability under image augmentations:
- Horizontal flips, brightness shifts, crops

Experiment 20 in the CalibDrive series.
"""
import os
import json
import time
import datetime
import numpy as np
import torch
from PIL import Image, ImageEnhance

RESULTS_DIR = "/workspace/Vizuara-VLA-Research/experiments"
os.makedirs(RESULTS_DIR, exist_ok=True)

TEMPERATURES = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]

SCENARIOS = {
    'highway': {'n': 15, 'speed': '30', 'difficulty': 'easy'},
    'urban': {'n': 15, 'speed': '15', 'difficulty': 'easy'},
    'night': {'n': 10, 'speed': '25', 'difficulty': 'hard'},
    'rain': {'n': 10, 'speed': '20', 'difficulty': 'hard'},
    'ood_noise': {'n': 15, 'speed': '25', 'difficulty': 'ood'},
    'ood_blank': {'n': 15, 'speed': '25', 'difficulty': 'ood'},
}


def create_scene_image(scenario, idx, size=(256, 256)):
    np.random.seed(idx * 1500 + hash(scenario) % 15000)
    if scenario == 'highway':
        img = np.zeros((*size, 3), dtype=np.uint8)
        img[:size[0]//2] = [135, 206, 235]
        img[size[0]//2:] = [80, 80, 80]
    elif scenario == 'urban':
        img = np.zeros((*size, 3), dtype=np.uint8)
        img[:size[0]//3] = [135, 206, 235]
        img[size[0]//3:size[0]//2] = [139, 119, 101]
        img[size[0]//2:] = [80, 80, 80]
    elif scenario == 'night':
        img = np.full((*size, 3), 15, dtype=np.uint8)
        img[size[0]//2:] = [30, 30, 35]
    elif scenario == 'rain':
        img = np.zeros((*size, 3), dtype=np.uint8)
        img[:size[0]//2] = [100, 100, 110]
        img[size[0]//2:] = [60, 60, 65]
    elif scenario == 'ood_noise':
        img = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
    elif scenario == 'ood_blank':
        img = np.full((*size, 3), 128, dtype=np.uint8)
    else:
        img = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
    noise = np.random.randint(-5, 5, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def augment_image(image, aug_type):
    """Apply augmentation to PIL Image."""
    if aug_type == 'original':
        return image
    elif aug_type == 'flip_h':
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    elif aug_type == 'bright_up':
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(1.3)
    elif aug_type == 'bright_down':
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(0.7)
    elif aug_type == 'crop_center':
        w, h = image.size
        crop_w, crop_h = int(w * 0.8), int(h * 0.8)
        left = (w - crop_w) // 2
        top = (h - crop_h) // 2
        cropped = image.crop((left, top, left + crop_w, top + crop_h))
        return cropped.resize((w, h), Image.BILINEAR)
    return image


AUG_TYPES = ['original', 'flip_h', 'bright_up', 'bright_down', 'crop_center']


def forward_with_temperature(model, processor, image, prompt, temperatures):
    """Forward pass, then apply different temperatures to get action mass at each T."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=7,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

    vocab_size = outputs.scores[0].shape[-1]
    action_start = vocab_size - 256

    temp_results = {}
    for T in temperatures:
        dim_action_masses = []
        dim_entropies = []
        dim_confs = []

        for score in outputs.scores[:7]:
            full_logits = score[0].float() / T
            full_probs = torch.softmax(full_logits, dim=0).cpu().numpy()

            action_probs = full_probs[action_start:]
            action_mass = float(action_probs.sum())
            dim_action_masses.append(action_mass)

            action_norm = action_probs / (action_probs.sum() + 1e-10)
            entropy = float(-(action_norm * np.log(action_norm + 1e-10)).sum())
            conf = float(action_norm.max())
            dim_entropies.append(entropy)
            dim_confs.append(conf)

        temp_results[str(T)] = {
            'mean_action_mass': float(np.mean(dim_action_masses)),
            'min_action_mass': float(np.min(dim_action_masses)),
            'mean_entropy': float(np.mean(dim_entropies)),
            'geo_conf': float(np.exp(np.mean(np.log(np.array(dim_confs) + 1e-10)))),
            'dim_masses': dim_action_masses,
        }

    return temp_results


def compute_auroc(pos_scores, neg_scores):
    n_correct = sum(1 for p in pos_scores for n in neg_scores if p > n)
    n_ties = sum(0.5 for p in pos_scores for n in neg_scores if p == n)
    n_total = len(pos_scores) * len(neg_scores)
    return (n_correct + n_ties) / n_total if n_total > 0 else 0.5


def main():
    print("=" * 70, flush=True)
    print("ACTION MASS UNDER TEMPERATURE & AUGMENTATION ON REAL OpenVLA-7B", flush=True)
    print("=" * 70, flush=True)

    from transformers import AutoModelForVision2Seq, AutoProcessor
    print("Loading OpenVLA-7B...", flush=True)
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        "openvla/openvla-7b",
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.", flush=True)

    total = sum(s['n'] for s in SCENARIOS.values())
    # Phase 1: Temperature sweep (1 forward pass per sample, reuse logits)
    # Phase 2: Augmentation sweep (5 forward passes per sample)
    total_inferences = total * (1 + len(AUG_TYPES))
    print(f"Total samples: {total}, Total inferences: {total_inferences}", flush=True)
    print(f"Temperatures: {TEMPERATURES}", flush=True)
    print(f"Augmentations: {AUG_TYPES}", flush=True)
    print(flush=True)

    all_results = {}
    sample_idx = 0
    prompt_tmpl = "In: What action should the robot take to drive forward at {speed} m/s safely?\nOut:"

    for scenario, config in SCENARIOS.items():
        scenario_results = []

        for i in range(config['n']):
            sample_idx += 1
            image = create_scene_image(scenario, i)
            prompt = prompt_tmpl.format(speed=config['speed'])

            t0 = time.time()

            # Phase 1: Temperature sweep (single forward pass, reuse logits)
            temp_results = forward_with_temperature(model, processor, image, prompt, TEMPERATURES)

            # Phase 2: Augmentation sweep
            aug_results = {}
            for aug in AUG_TYPES:
                aug_img = augment_image(image, aug)
                aug_r = forward_with_temperature(model, processor, aug_img, prompt, [1.0])
                aug_results[aug] = aug_r['1.0']

            elapsed = time.time() - t0

            sample_result = {
                'scenario': scenario,
                'difficulty': config['difficulty'],
                'idx': i,
                'temperatures': temp_results,
                'augmentations': aug_results,
            }
            scenario_results.append(sample_result)

            if i % 5 == 0 or i == config['n'] - 1:
                masses_str = ", ".join(f"T={T}:{temp_results[str(T)]['mean_action_mass']:.3f}"
                                      for T in [0.5, 1.0, 3.0])
                print(f"  [{sample_idx}/{total}] {scenario}_{i}: {masses_str} ({elapsed:.1f}s)", flush=True)

        all_results[scenario] = scenario_results

    # ===================================================================
    # Analysis
    # ===================================================================
    print("\n" + "=" * 70, flush=True)
    print("TEMPERATURE & AUGMENTATION ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    # 1. Action mass vs temperature
    print("\n1. Action Mass vs Temperature (Per Scenario)", flush=True)
    print("-" * 100, flush=True)
    header = f"{'Scenario':<12}" + "".join(f"T={T:>4}" + " " * 4 for T in TEMPERATURES)
    print(header, flush=True)
    print("-" * 100, flush=True)

    for scenario in SCENARIOS:
        samples = all_results[scenario]
        row = f"{scenario:<12}"
        for T in TEMPERATURES:
            masses = [s['temperatures'][str(T)]['mean_action_mass'] for s in samples]
            row += f"{np.mean(masses):>8.4f}"
        print(row, flush=True)

    # 2. AUROC vs temperature
    print("\n2. AUROC (Easy vs OOD) vs Temperature", flush=True)
    print("-" * 60, flush=True)

    best_auroc = 0
    best_T = 1.0
    for T in TEMPERATURES:
        easy_mass = []
        ood_mass = []
        easy_ent = []
        ood_ent = []
        for scenario in SCENARIOS:
            for s in all_results[scenario]:
                mass = s['temperatures'][str(T)]['mean_action_mass']
                ent = s['temperatures'][str(T)]['mean_entropy']
                if s['difficulty'] == 'easy':
                    easy_mass.append(mass)
                    easy_ent.append(ent)
                elif s['difficulty'] == 'ood':
                    ood_mass.append(mass)
                    ood_ent.append(ent)

        auroc_mass = compute_auroc(easy_mass, ood_mass)
        auroc_ent = compute_auroc(ood_ent, easy_ent)

        if auroc_mass > best_auroc:
            best_auroc = auroc_mass
            best_T = T

        print(f"  T={T:<5}: Neg Action Mass AUROC = {auroc_mass:.3f}, "
              f"Entropy AUROC = {auroc_ent:.3f}", flush=True)

    print(f"\n  Best T for action mass: T={best_T} (AUROC={best_auroc:.3f})", flush=True)

    # 3. Augmentation robustness
    print("\n3. Action Mass Under Image Augmentations (T=1.0)", flush=True)
    print("-" * 70, flush=True)

    for aug in AUG_TYPES:
        easy_mass = []
        ood_mass = []
        for scenario in SCENARIOS:
            for s in all_results[scenario]:
                mass = s['augmentations'][aug]['mean_action_mass']
                if s['difficulty'] == 'easy':
                    easy_mass.append(mass)
                elif s['difficulty'] == 'ood':
                    ood_mass.append(mass)

        auroc = compute_auroc(easy_mass, ood_mass)
        print(f"  {aug:<15}: AUROC = {auroc:.3f} "
              f"(easy={np.mean(easy_mass):.4f}, ood={np.mean(ood_mass):.4f})", flush=True)

    # 4. Augmentation ensemble (average action mass across augmentations)
    print("\n4. Augmentation Ensemble (average across augmentations)", flush=True)
    print("-" * 60, flush=True)

    easy_ensemble = []
    ood_ensemble = []
    for scenario in SCENARIOS:
        for s in all_results[scenario]:
            avg_mass = np.mean([s['augmentations'][aug]['mean_action_mass'] for aug in AUG_TYPES])
            if s['difficulty'] == 'easy':
                easy_ensemble.append(avg_mass)
            elif s['difficulty'] == 'ood':
                ood_ensemble.append(avg_mass)

    auroc_ensemble = compute_auroc(easy_ensemble, ood_ensemble)
    print(f"  Augmentation ensemble AUROC = {auroc_ensemble:.3f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'timestamp': timestamp,
        'config': {
            'temperatures': TEMPERATURES,
            'augmentations': AUG_TYPES,
        },
        'per_sample': {
            scenario: [
                {k: v for k, v in s.items()}
                for s in samples
            ]
            for scenario, samples in all_results.items()
        },
    }

    output_path = os.path.join(RESULTS_DIR, f"action_mass_temp_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
