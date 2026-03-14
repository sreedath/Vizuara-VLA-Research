"""
Selective Prediction with Action Mass on Real OpenVLA-7B.

The definitive selective prediction experiment comparing all discovered signals:
1. Single-pass action mass (best new signal)
2. MC Dropout entropy (p=0.20, N=10)
3. Augmentation ensemble action mass (5 augmentations)
4. Temperature-scaled action mass (T=0.25)
5. Two-signal: action mass (T=0.25) + MC entropy

Measures coverage-safety trade-off at multiple coverage levels.

Experiment 21 in the CalibDrive series.
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

OPTIMAL_DROPOUT = 0.20
N_MC_PASSES = 10
COVERAGE_LEVELS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

SCENARIOS = {
    'highway': {'n': 20, 'speed': '30', 'difficulty': 'easy'},
    'urban': {'n': 20, 'speed': '15', 'difficulty': 'easy'},
    'night': {'n': 15, 'speed': '25', 'difficulty': 'hard'},
    'rain': {'n': 15, 'speed': '20', 'difficulty': 'hard'},
    'ood_noise': {'n': 20, 'speed': '25', 'difficulty': 'ood'},
    'ood_blank': {'n': 20, 'speed': '25', 'difficulty': 'ood'},
}

AUG_TYPES = ['original', 'flip_h', 'bright_up', 'bright_down', 'crop_center']


def create_scene_image(scenario, idx, size=(256, 256)):
    np.random.seed(idx * 1600 + hash(scenario) % 16000)
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
    if aug_type == 'original':
        return image
    elif aug_type == 'flip_h':
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    elif aug_type == 'bright_up':
        return ImageEnhance.Brightness(image).enhance(1.3)
    elif aug_type == 'bright_down':
        return ImageEnhance.Brightness(image).enhance(0.7)
    elif aug_type == 'crop_center':
        w, h = image.size
        cw, ch = int(w * 0.8), int(h * 0.8)
        l, t = (w - cw) // 2, (h - ch) // 2
        return image.crop((l, t, l + cw, t + ch)).resize((w, h), Image.BILINEAR)
    return image


def set_dropout_rate(model, rate):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = rate


def enable_mc_dropout(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


def forward_full_vocab(model, processor, image, prompt, temperature=1.0):
    """Forward pass returning action mass and entropy at given temperature."""
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

    dim_masses = []
    dim_entropies = []
    dim_confs = []

    for score in outputs.scores[:7]:
        full_logits = score[0].float() / temperature
        full_probs = torch.softmax(full_logits, dim=0).cpu().numpy()

        action_probs = full_probs[action_start:]
        dim_masses.append(float(action_probs.sum()))

        action_norm = action_probs / (action_probs.sum() + 1e-10)
        dim_entropies.append(float(-(action_norm * np.log(action_norm + 1e-10)).sum()))
        dim_confs.append(float(action_norm.max()))

    return {
        'action_mass': float(np.mean(dim_masses)),
        'entropy': float(np.mean(dim_entropies)),
        'conf': float(np.exp(np.mean(np.log(np.array(dim_confs) + 1e-10)))),
    }


def compute_auroc(pos_scores, neg_scores):
    n_correct = sum(1 for p in pos_scores for n in neg_scores if p > n)
    n_ties = sum(0.5 for p in pos_scores for n in neg_scores if p == n)
    n_total = len(pos_scores) * len(neg_scores)
    return (n_correct + n_ties) / n_total if n_total > 0 else 0.5


def selective_prediction_stats(scores, labels, coverage):
    """Given uncertainty scores (higher=more uncertain) and binary labels (1=bad),
    compute stats at given coverage level."""
    n = len(scores)
    n_keep = max(1, int(coverage * n))

    # Sort by uncertainty, keep lowest-uncertainty samples
    sorted_indices = np.argsort(scores)
    kept_indices = sorted_indices[:n_keep]
    rejected_indices = sorted_indices[n_keep:]

    kept_labels = [labels[i] for i in kept_indices]
    rejected_labels = [labels[i] for i in rejected_indices]

    ood_in_kept = sum(kept_labels)
    ood_in_rejected = sum(rejected_labels)
    total_ood = sum(labels)

    ood_rejection_rate = ood_in_rejected / total_ood if total_ood > 0 else 0
    ood_in_kept_rate = ood_in_kept / n_keep if n_keep > 0 else 0

    return {
        'coverage': coverage,
        'n_kept': n_keep,
        'ood_rejection_rate': float(ood_rejection_rate),
        'ood_in_kept_rate': float(ood_in_kept_rate),
    }


def main():
    print("=" * 70, flush=True)
    print("SELECTIVE PREDICTION WITH ACTION MASS ON REAL OpenVLA-7B", flush=True)
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
    # Per sample: 1 (single T=1.0) + 1 (single T=0.25) + 10 (MC) + 5 (augmentations)
    total_inferences = total * (1 + 1 + N_MC_PASSES + len(AUG_TYPES))
    print(f"Total samples: {total}, Total inferences: {total_inferences}", flush=True)
    print(flush=True)

    all_samples = []
    sample_idx = 0
    prompt = "In: What action should the robot take to drive forward at {speed} m/s safely?\nOut:"

    for scenario, config in SCENARIOS.items():
        for i in range(config['n']):
            sample_idx += 1
            image = create_scene_image(scenario, i)
            p = prompt.format(speed=config['speed'])

            t0 = time.time()

            # Signal 1: Single-pass action mass (T=1.0)
            set_dropout_rate(model, 0.0)
            model.eval()
            r1 = forward_full_vocab(model, processor, image, p, temperature=1.0)

            # Signal 2: Single-pass action mass (T=0.25)
            r2 = forward_full_vocab(model, processor, image, p, temperature=0.25)

            # Signal 3: MC Dropout entropy (p=0.20, N=10)
            set_dropout_rate(model, OPTIMAL_DROPOUT)
            enable_mc_dropout(model)
            mc_results = []
            for mc in range(N_MC_PASSES):
                r = forward_full_vocab(model, processor, image, p, temperature=1.0)
                mc_results.append(r)
            mc_entropy = float(np.mean([r['entropy'] for r in mc_results]))
            mc_action_mass = float(np.mean([r['action_mass'] for r in mc_results]))

            # Signal 4: Augmentation ensemble action mass
            set_dropout_rate(model, 0.0)
            model.eval()
            aug_masses = []
            for aug in AUG_TYPES:
                aug_img = augment_image(image, aug)
                r = forward_full_vocab(model, processor, aug_img, p, temperature=1.0)
                aug_masses.append(r['action_mass'])
            aug_ensemble_mass = float(np.mean(aug_masses))

            elapsed = time.time() - t0

            sample = {
                'scenario': scenario,
                'difficulty': config['difficulty'],
                'idx': i,
                'is_ood': 1 if config['difficulty'] == 'ood' else 0,
                'is_hard': 1 if config['difficulty'] == 'hard' else 0,
                'is_not_easy': 1 if config['difficulty'] != 'easy' else 0,
                # Signals (higher = more uncertain / more likely OOD)
                'neg_action_mass_T1': -r1['action_mass'],
                'neg_action_mass_T025': -r2['action_mass'],
                'entropy_T1': r1['entropy'],
                'mc_entropy': mc_entropy,
                'neg_aug_action_mass': -aug_ensemble_mass,
                # Combined: negative action mass (T=0.25) + MC entropy (normalized)
                'combined_mass_entropy': float(-r2['action_mass'] + mc_entropy / 5.0),
            }
            all_samples.append(sample)

            if i % 5 == 0 or i == config['n'] - 1:
                print(f"  [{sample_idx}/{total}] {scenario}_{i}: "
                      f"am={r1['action_mass']:.3f}, am_T025={r2['action_mass']:.3f}, "
                      f"mc_ent={mc_entropy:.3f}, aug_am={aug_ensemble_mass:.3f} "
                      f"({elapsed:.1f}s)", flush=True)

    # ===================================================================
    # Analysis
    # ===================================================================
    print("\n" + "=" * 70, flush=True)
    print("SELECTIVE PREDICTION ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    signal_names = [
        'neg_action_mass_T1',
        'neg_action_mass_T025',
        'entropy_T1',
        'mc_entropy',
        'neg_aug_action_mass',
        'combined_mass_entropy',
    ]
    display_names = {
        'neg_action_mass_T1': 'Action Mass (T=1.0)',
        'neg_action_mass_T025': 'Action Mass (T=0.25)',
        'entropy_T1': 'Entropy (T=1.0)',
        'mc_entropy': 'MC Entropy (p=0.20)',
        'neg_aug_action_mass': 'Aug Ensemble Mass',
        'combined_mass_entropy': 'Mass(T=0.25)+MCEnt',
    }

    # 1. AUROC for all signals and label types
    print("\n1. AUROC Comparison", flush=True)
    print("-" * 80, flush=True)
    print(f"{'Signal':<25} {'Easy vs OOD':>12} {'Easy vs Hard':>12} {'Easy vs Rest':>12}", flush=True)
    print("-" * 80, flush=True)

    for sig in signal_names:
        scores = [s[sig] for s in all_samples]

        # Easy vs OOD
        easy_s = [s[sig] for s in all_samples if s['difficulty'] == 'easy']
        ood_s = [s[sig] for s in all_samples if s['difficulty'] == 'ood']
        hard_s = [s[sig] for s in all_samples if s['difficulty'] == 'hard']
        rest_s = [s[sig] for s in all_samples if s['difficulty'] != 'easy']

        auroc_ood = compute_auroc(ood_s, easy_s)
        auroc_hard = compute_auroc(hard_s, easy_s)
        auroc_rest = compute_auroc(rest_s, easy_s)

        print(f"  {display_names[sig]:<25} {auroc_ood:>12.3f} {auroc_hard:>12.3f} {auroc_rest:>12.3f}", flush=True)

    # 2. Selective prediction curves
    print("\n2. OOD Rejection Rate at Various Coverage Levels", flush=True)
    print("-" * 100, flush=True)
    header = f"{'Coverage':>8}"
    for sig in signal_names:
        header += f" {display_names[sig][:12]:>13}"
    print(header, flush=True)
    print("-" * 100, flush=True)

    labels_ood = [s['is_ood'] for s in all_samples]

    for cov in COVERAGE_LEVELS:
        row = f"{cov:>8.0%}"
        for sig in signal_names:
            scores = [s[sig] for s in all_samples]
            stats = selective_prediction_stats(scores, labels_ood, cov)
            row += f" {stats['ood_rejection_rate']:>13.1%}"
        print(row, flush=True)

    # 3. Non-easy rejection rate
    print("\n3. Non-Easy Rejection Rate at Various Coverage Levels", flush=True)
    print("-" * 100, flush=True)
    print(header, flush=True)
    print("-" * 100, flush=True)

    labels_not_easy = [s['is_not_easy'] for s in all_samples]

    for cov in COVERAGE_LEVELS:
        row = f"{cov:>8.0%}"
        for sig in signal_names:
            scores = [s[sig] for s in all_samples]
            stats = selective_prediction_stats(scores, labels_not_easy, cov)
            row += f" {stats['ood_rejection_rate']:>13.1%}"
        print(row, flush=True)

    # 4. Compute cost per signal
    print("\n4. Computational Cost Per Signal", flush=True)
    print("-" * 50, flush=True)
    costs = {
        'neg_action_mass_T1': 1,
        'neg_action_mass_T025': 1,
        'entropy_T1': 1,
        'mc_entropy': N_MC_PASSES,
        'neg_aug_action_mass': len(AUG_TYPES),
        'combined_mass_entropy': 1 + N_MC_PASSES,
    }
    for sig in signal_names:
        easy_s = [s[sig] for s in all_samples if s['difficulty'] == 'easy']
        ood_s = [s[sig] for s in all_samples if s['difficulty'] == 'ood']
        auroc = compute_auroc(ood_s, easy_s)
        print(f"  {display_names[sig]:<25}: {costs[sig]:>2} passes, AUROC={auroc:.3f}, "
              f"AUROC/pass={auroc/costs[sig]:.3f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'timestamp': timestamp,
        'config': {
            'n_mc_passes': N_MC_PASSES,
            'dropout_rate': OPTIMAL_DROPOUT,
            'coverage_levels': COVERAGE_LEVELS,
        },
        'samples': all_samples,
    }

    output_path = os.path.join(RESULTS_DIR, f"selective_action_mass_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
