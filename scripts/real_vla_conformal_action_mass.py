"""
Conformal Prediction with Action Mass on Real OpenVLA-7B.

Applies conformal prediction to the action mass signal:
1. Use easy scenarios as calibration set
2. Compute nonconformity scores from action mass
3. Set thresholds at various alpha levels
4. Test whether conformal guarantees hold on OOD/hard test sets
5. Compare coverage-conditional vs marginal conformal prediction

This tests whether action mass satisfies exchangeability assumptions
required for valid conformal prediction guarantees.

Experiment 22 in the CalibDrive series.
"""
import os
import json
import time
import datetime
import numpy as np
import torch
from PIL import Image

RESULTS_DIR = "/workspace/Vizuara-VLA-Research/experiments"
os.makedirs(RESULTS_DIR, exist_ok=True)

ALPHA_LEVELS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

SCENARIOS = {
    'highway': {'n': 30, 'speed': '30', 'difficulty': 'easy'},
    'urban': {'n': 30, 'speed': '15', 'difficulty': 'easy'},
    'night': {'n': 20, 'speed': '25', 'difficulty': 'hard'},
    'rain': {'n': 20, 'speed': '20', 'difficulty': 'hard'},
    'ood_noise': {'n': 25, 'speed': '25', 'difficulty': 'ood'},
    'ood_blank': {'n': 25, 'speed': '25', 'difficulty': 'ood'},
}


def create_scene_image(scenario, idx, size=(256, 256)):
    np.random.seed(idx * 1700 + hash(scenario) % 17000)
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


def forward_full_vocab(model, processor, image, prompt):
    """Forward pass returning action mass and entropy."""
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
    dim_tokens = []

    for score in outputs.scores[:7]:
        full_logits = score[0].float()
        full_probs = torch.softmax(full_logits, dim=0).cpu().numpy()

        action_probs = full_probs[action_start:]
        dim_masses.append(float(action_probs.sum()))

        action_norm = action_probs / (action_probs.sum() + 1e-10)
        dim_entropies.append(float(-(action_norm * np.log(action_norm + 1e-10)).sum()))
        dim_confs.append(float(action_norm.max()))
        dim_tokens.append(int(action_norm.argmax()))

    return {
        'action_mass': float(np.mean(dim_masses)),
        'min_action_mass': float(np.min(dim_masses)),
        'dim_action_masses': dim_masses,
        'entropy': float(np.mean(dim_entropies)),
        'conf': float(np.exp(np.mean(np.log(np.array(dim_confs) + 1e-10)))),
        'tokens': dim_tokens,
    }


def compute_auroc(pos_scores, neg_scores):
    n_correct = sum(1 for p in pos_scores for n in neg_scores if p > n)
    n_ties = sum(0.5 for p in pos_scores for n in neg_scores if p == n)
    n_total = len(pos_scores) * len(neg_scores)
    return (n_correct + n_ties) / n_total if n_total > 0 else 0.5


def conformal_threshold(calibration_scores, alpha):
    """Compute conformal quantile threshold."""
    n = len(calibration_scores)
    level = np.ceil((1 - alpha) * (n + 1)) / n
    level = min(level, 1.0)
    return float(np.quantile(calibration_scores, level))


def main():
    print("=" * 70, flush=True)
    print("CONFORMAL PREDICTION WITH ACTION MASS ON REAL OpenVLA-7B", flush=True)
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
    print(f"Total samples: {total}", flush=True)
    print(flush=True)

    prompt_tmpl = "In: What action should the robot take to drive forward at {speed} m/s safely?\nOut:"

    all_samples = []
    sample_idx = 0

    for scenario, config in SCENARIOS.items():
        for i in range(config['n']):
            sample_idx += 1
            image = create_scene_image(scenario, i)
            prompt = prompt_tmpl.format(speed=config['speed'])

            t0 = time.time()
            r = forward_full_vocab(model, processor, image, prompt)
            elapsed = time.time() - t0

            sample = {
                'scenario': scenario,
                'difficulty': config['difficulty'],
                'idx': i,
                'action_mass': r['action_mass'],
                'min_action_mass': r['min_action_mass'],
                'dim_action_masses': r['dim_action_masses'],
                'entropy': r['entropy'],
                'conf': r['conf'],
                'tokens': r['tokens'],
                # Nonconformity: lower action mass = more nonconforming
                'nonconformity': 1.0 - r['action_mass'],
            }
            all_samples.append(sample)

            if i % 10 == 0 or i == config['n'] - 1:
                print(f"  [{sample_idx}/{total}] {scenario}_{i}: "
                      f"mass={r['action_mass']:.4f}, ent={r['entropy']:.3f} "
                      f"({elapsed:.1f}s)", flush=True)

    # ===================================================================
    # Conformal Prediction Analysis
    # ===================================================================
    print("\n" + "=" * 70, flush=True)
    print("CONFORMAL PREDICTION ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    # Split: use first half of easy as calibration, rest as test
    easy_samples = [s for s in all_samples if s['difficulty'] == 'easy']
    hard_samples = [s for s in all_samples if s['difficulty'] == 'hard']
    ood_samples = [s for s in all_samples if s['difficulty'] == 'ood']

    np.random.seed(42)
    np.random.shuffle(easy_samples)
    n_cal = len(easy_samples) // 2
    cal_samples = easy_samples[:n_cal]
    test_easy = easy_samples[n_cal:]

    print(f"\nCalibration set: {n_cal} easy samples", flush=True)
    print(f"Test sets: {len(test_easy)} easy, {len(hard_samples)} hard, {len(ood_samples)} OOD", flush=True)

    # Calibration nonconformity scores
    cal_scores = [s['nonconformity'] for s in cal_samples]

    # 1. Conformal thresholds and coverage
    print("\n1. Conformal Thresholds and Test Coverage", flush=True)
    print("-" * 90, flush=True)
    print(f"{'Alpha':>6} {'Threshold':>10} {'Easy Cov':>10} {'Hard Cov':>10} {'OOD Cov':>10} "
          f"{'OOD Flag':>10} {'Hard Flag':>10}", flush=True)
    print("-" * 90, flush=True)

    conformal_results = {}
    for alpha in ALPHA_LEVELS:
        threshold = conformal_threshold(cal_scores, alpha)

        # Coverage = fraction of test samples with nonconformity <= threshold
        easy_cov = np.mean([s['nonconformity'] <= threshold for s in test_easy])
        hard_cov = np.mean([s['nonconformity'] <= threshold for s in hard_samples])
        ood_cov = np.mean([s['nonconformity'] <= threshold for s in ood_samples])

        # Flag rate = fraction flagged as uncertain
        ood_flag = 1.0 - ood_cov
        hard_flag = 1.0 - hard_cov

        conformal_results[str(alpha)] = {
            'alpha': alpha,
            'threshold': threshold,
            'easy_coverage': float(easy_cov),
            'hard_coverage': float(hard_cov),
            'ood_coverage': float(ood_cov),
            'ood_flag_rate': float(ood_flag),
            'hard_flag_rate': float(hard_flag),
        }

        print(f"{alpha:>6.2f} {threshold:>10.4f} {easy_cov:>10.1%} {hard_cov:>10.1%} "
              f"{ood_cov:>10.1%} {ood_flag:>10.1%} {hard_flag:>10.1%}", flush=True)

    # 2. Per-dimension conformal
    print("\n2. Per-Dimension Conformal (alpha=0.10)", flush=True)
    print("-" * 60, flush=True)

    for d in range(7):
        cal_dim_scores = [1.0 - s['dim_action_masses'][d] for s in cal_samples]
        threshold = conformal_threshold(cal_dim_scores, 0.10)

        easy_flag = np.mean([1.0 - s['dim_action_masses'][d] > threshold for s in test_easy])
        hard_flag = np.mean([1.0 - s['dim_action_masses'][d] > threshold for s in hard_samples])
        ood_flag = np.mean([1.0 - s['dim_action_masses'][d] > threshold for s in ood_samples])

        print(f"  Dim {d}: threshold={threshold:.4f}, "
              f"easy_flag={easy_flag:.1%}, hard_flag={hard_flag:.1%}, "
              f"ood_flag={ood_flag:.1%}", flush=True)

    # 3. Comparison: conformal action mass vs raw AUROC
    print("\n3. Conformal vs Raw AUROC Comparison", flush=True)
    print("-" * 60, flush=True)

    # Raw AUROC on test set
    test_all = test_easy + hard_samples + ood_samples
    easy_mass = [-s['action_mass'] for s in test_easy]
    ood_mass = [-s['action_mass'] for s in ood_samples]
    hard_mass = [-s['action_mass'] for s in hard_samples]

    auroc_ood = compute_auroc(ood_mass, easy_mass)
    auroc_hard = compute_auroc(hard_mass, easy_mass)
    print(f"  Raw action mass AUROC (easy vs OOD): {auroc_ood:.3f}", flush=True)
    print(f"  Raw action mass AUROC (easy vs hard): {auroc_hard:.3f}", flush=True)

    easy_ent = [s['entropy'] for s in test_easy]
    ood_ent = [s['entropy'] for s in ood_samples]
    hard_ent = [s['entropy'] for s in hard_samples]
    print(f"  Raw entropy AUROC (easy vs OOD): {compute_auroc(ood_ent, easy_ent):.3f}", flush=True)
    print(f"  Raw entropy AUROC (easy vs hard): {compute_auroc(hard_ent, easy_ent):.3f}", flush=True)

    # 4. Token prediction sets
    print("\n4. Prediction Set Sizes (Conformal on Per-Dim Distributions)", flush=True)
    print("-" * 60, flush=True)

    # For each dimension, count how many bins are in the prediction set at alpha=0.10
    for difficulty in ['easy', 'hard', 'ood']:
        if difficulty == 'easy':
            samps = test_easy
        elif difficulty == 'hard':
            samps = hard_samples
        else:
            samps = ood_samples

        set_sizes = []
        for s in samps:
            # Average number of dims flagged as uncertain
            n_flagged = sum(1 for m in s['dim_action_masses'] if 1.0 - m > conformal_threshold(cal_scores, 0.10))
            set_sizes.append(n_flagged)
        print(f"  {difficulty:<6}: mean dims flagged = {np.mean(set_sizes):.2f} / 7 "
              f"(±{np.std(set_sizes):.2f})", flush=True)

    # 5. Action mass distribution statistics
    print("\n5. Action Mass Distribution by Difficulty", flush=True)
    print("-" * 60, flush=True)

    for label, samps in [('Cal Easy', cal_samples), ('Test Easy', test_easy),
                          ('Hard', hard_samples), ('OOD', ood_samples)]:
        masses = [s['action_mass'] for s in samps]
        print(f"  {label:<10}: mean={np.mean(masses):.4f} ± {np.std(masses):.4f}, "
              f"min={np.min(masses):.4f}, max={np.max(masses):.4f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'timestamp': timestamp,
        'config': {
            'alpha_levels': ALPHA_LEVELS,
            'n_calibration': n_cal,
        },
        'conformal_results': conformal_results,
        'samples': all_samples,
    }

    output_path = os.path.join(RESULTS_DIR, f"conformal_action_mass_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
