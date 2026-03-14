"""
Large-Scale Action Mass Validation on Real OpenVLA-7B.

Definitive validation of the action mass signal with larger sample sizes
and proper cross-validation for reliable AUROC estimates:
1. 200 samples across 6 scenarios
2. 5-fold cross-validated conformal prediction
3. Bootstrap confidence intervals for AUROC
4. Statistical tests comparing action mass vs entropy

Experiment 24 in the CalibDrive series.
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

SCENARIOS = {
    'highway': {'n': 40, 'speed': '30', 'difficulty': 'easy'},
    'urban': {'n': 40, 'speed': '15', 'difficulty': 'easy'},
    'night': {'n': 25, 'speed': '25', 'difficulty': 'hard'},
    'rain': {'n': 25, 'speed': '20', 'difficulty': 'hard'},
    'ood_noise': {'n': 35, 'speed': '25', 'difficulty': 'ood'},
    'ood_blank': {'n': 35, 'speed': '25', 'difficulty': 'ood'},
}


def create_scene_image(scenario, idx, size=(256, 256)):
    np.random.seed(idx * 1900 + hash(scenario) % 19000)
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
        full_logits = score[0].float()
        full_probs = torch.softmax(full_logits, dim=0).cpu().numpy()

        action_probs = full_probs[action_start:]
        dim_masses.append(float(action_probs.sum()))

        action_norm = action_probs / (action_probs.sum() + 1e-10)
        dim_entropies.append(float(-(action_norm * np.log(action_norm + 1e-10)).sum()))
        dim_confs.append(float(action_norm.max()))

    return {
        'action_mass': float(np.mean(dim_masses)),
        'min_action_mass': float(np.min(dim_masses)),
        'dim_masses': dim_masses,
        'entropy': float(np.mean(dim_entropies)),
        'conf': float(np.exp(np.mean(np.log(np.array(dim_confs) + 1e-10)))),
    }


def compute_auroc(pos_scores, neg_scores):
    n_correct = sum(1 for p in pos_scores for n in neg_scores if p > n)
    n_ties = sum(0.5 for p in pos_scores for n in neg_scores if p == n)
    n_total = len(pos_scores) * len(neg_scores)
    return (n_correct + n_ties) / n_total if n_total > 0 else 0.5


def bootstrap_auroc(pos_scores, neg_scores, n_bootstrap=1000, seed=42):
    """Compute AUROC with 95% CI via bootstrap."""
    rng = np.random.RandomState(seed)
    aurocs = []
    pos_arr = np.array(pos_scores)
    neg_arr = np.array(neg_scores)
    for _ in range(n_bootstrap):
        pos_boot = rng.choice(pos_arr, size=len(pos_arr), replace=True)
        neg_boot = rng.choice(neg_arr, size=len(neg_arr), replace=True)
        aurocs.append(compute_auroc(pos_boot.tolist(), neg_boot.tolist()))
    return {
        'mean': float(np.mean(aurocs)),
        'std': float(np.std(aurocs)),
        'ci_lower': float(np.percentile(aurocs, 2.5)),
        'ci_upper': float(np.percentile(aurocs, 97.5)),
    }


def main():
    print("=" * 70, flush=True)
    print("LARGE-SCALE ACTION MASS VALIDATION ON REAL OpenVLA-7B", flush=True)
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

    prompt = "In: What action should the robot take to drive forward at {speed} m/s safely?\nOut:"
    all_samples = []
    sample_idx = 0

    for scenario, config in SCENARIOS.items():
        for i in range(config['n']):
            sample_idx += 1
            image = create_scene_image(scenario, i)
            p = prompt.format(speed=config['speed'])

            t0 = time.time()
            r = forward_full_vocab(model, processor, image, p)
            elapsed = time.time() - t0

            sample = {
                'scenario': scenario,
                'difficulty': config['difficulty'],
                'idx': i,
                'action_mass': r['action_mass'],
                'min_action_mass': r['min_action_mass'],
                'dim_masses': r['dim_masses'],
                'entropy': r['entropy'],
                'conf': r['conf'],
            }
            all_samples.append(sample)

            if i % 10 == 0 or i == config['n'] - 1:
                print(f"  [{sample_idx}/{total}] {scenario}_{i}: "
                      f"mass={r['action_mass']:.4f}, ent={r['entropy']:.3f} "
                      f"({elapsed:.1f}s)", flush=True)

    # ===================================================================
    # Analysis
    # ===================================================================
    print("\n" + "=" * 70, flush=True)
    print("LARGE-SCALE VALIDATION ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    # 1. Per-scenario statistics
    print("\n1. Per-Scenario Statistics", flush=True)
    print("-" * 80, flush=True)
    print(f"{'Scenario':<12} {'N':>4} {'Mass':>8} {'±':>6} {'Entropy':>8} {'±':>6} {'Conf':>7}", flush=True)
    print("-" * 80, flush=True)

    for scenario in SCENARIOS:
        samples = [s for s in all_samples if s['scenario'] == scenario]
        masses = [s['action_mass'] for s in samples]
        ents = [s['entropy'] for s in samples]
        confs = [s['conf'] for s in samples]
        print(f"{scenario:<12} {len(samples):>4} {np.mean(masses):>8.4f} {np.std(masses):>6.4f} "
              f"{np.mean(ents):>8.3f} {np.std(ents):>6.3f} {np.mean(confs):>7.3f}", flush=True)

    # 2. AUROC with bootstrap CIs
    print("\n2. AUROC with 95% Confidence Intervals (1000 bootstraps)", flush=True)
    print("-" * 80, flush=True)

    easy_samples = [s for s in all_samples if s['difficulty'] == 'easy']
    hard_samples = [s for s in all_samples if s['difficulty'] == 'hard']
    ood_samples = [s for s in all_samples if s['difficulty'] == 'ood']

    signals = {
        'Neg Action Mass': (lambda s: -s['action_mass']),
        'Neg Min Action Mass': (lambda s: -s['min_action_mass']),
        'Entropy': (lambda s: s['entropy']),
        'Neg Confidence': (lambda s: -s['conf']),
    }

    for sig_name, sig_fn in signals.items():
        easy_vals = [sig_fn(s) for s in easy_samples]
        ood_vals = [sig_fn(s) for s in ood_samples]
        hard_vals = [sig_fn(s) for s in hard_samples]

        auroc_ood = bootstrap_auroc(ood_vals, easy_vals)
        auroc_hard = bootstrap_auroc(hard_vals, easy_vals)

        print(f"  {sig_name:<22}: OOD AUROC={auroc_ood['mean']:.3f} "
              f"[{auroc_ood['ci_lower']:.3f}, {auroc_ood['ci_upper']:.3f}], "
              f"Hard AUROC={auroc_hard['mean']:.3f} "
              f"[{auroc_hard['ci_lower']:.3f}, {auroc_hard['ci_upper']:.3f}]", flush=True)

    # 3. Statistical tests: action mass vs entropy
    print("\n3. Statistical Tests: Action Mass vs Entropy", flush=True)
    print("-" * 60, flush=True)

    from scipy.stats import mannwhitneyu, ttest_ind

    for label, group_a, group_b in [
        ("Easy vs OOD", easy_samples, ood_samples),
        ("Easy vs Hard", easy_samples, hard_samples),
    ]:
        mass_a = [s['action_mass'] for s in group_a]
        mass_b = [s['action_mass'] for s in group_b]
        ent_a = [s['entropy'] for s in group_a]
        ent_b = [s['entropy'] for s in group_b]

        t_mass, p_mass = ttest_ind(mass_a, mass_b)
        u_mass, p_u_mass = mannwhitneyu(mass_a, mass_b, alternative='two-sided')
        t_ent, p_ent = ttest_ind(ent_a, ent_b)

        print(f"\n  {label}:", flush=True)
        print(f"    Action mass: t={t_mass:.3f}, p={p_mass:.6f} (Welch's)", flush=True)
        print(f"    Action mass: U={u_mass:.0f}, p={p_u_mass:.6f} (Mann-Whitney)", flush=True)
        print(f"    Entropy:     t={t_ent:.3f}, p={p_ent:.6f} (Welch's)", flush=True)
        print(f"    Mass gap: {np.mean(mass_a)-np.mean(mass_b):.4f}", flush=True)
        print(f"    Entropy gap: {np.mean(ent_b)-np.mean(ent_a):.3f}", flush=True)
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(mass_a) + np.var(mass_b)) / 2)
        cohens_d = (np.mean(mass_a) - np.mean(mass_b)) / pooled_std if pooled_std > 0 else 0
        print(f"    Cohen's d (mass): {cohens_d:.3f}", flush=True)

    # 4. 5-Fold cross-validated conformal
    print("\n4. 5-Fold Cross-Validated Conformal Prediction (alpha=0.10)", flush=True)
    print("-" * 60, flush=True)

    np.random.seed(42)
    indices = np.arange(len(easy_samples))
    np.random.shuffle(indices)
    n_folds = 5
    fold_size = len(indices) // n_folds

    fold_results = []
    for fold in range(n_folds):
        test_idx = indices[fold * fold_size:(fold + 1) * fold_size]
        cal_idx = np.concatenate([indices[:fold * fold_size], indices[(fold + 1) * fold_size:]])

        cal_scores = [1.0 - easy_samples[i]['action_mass'] for i in cal_idx]
        threshold = float(np.quantile(cal_scores, np.ceil(0.90 * (len(cal_scores) + 1)) / len(cal_scores)))

        easy_cov = np.mean([1.0 - easy_samples[i]['action_mass'] <= threshold for i in test_idx])
        ood_flag = np.mean([1.0 - s['action_mass'] > threshold for s in ood_samples])
        hard_flag = np.mean([1.0 - s['action_mass'] > threshold for s in hard_samples])

        fold_results.append({
            'fold': fold,
            'threshold': threshold,
            'easy_coverage': float(easy_cov),
            'ood_flag_rate': float(ood_flag),
            'hard_flag_rate': float(hard_flag),
        })
        print(f"  Fold {fold+1}: threshold={threshold:.4f}, "
              f"easy_cov={easy_cov:.1%}, ood_flag={ood_flag:.1%}, "
              f"hard_flag={hard_flag:.1%}", flush=True)

    print(f"\n  Mean: easy_cov={np.mean([f['easy_coverage'] for f in fold_results]):.1%}, "
          f"ood_flag={np.mean([f['ood_flag_rate'] for f in fold_results]):.1%}, "
          f"hard_flag={np.mean([f['hard_flag_rate'] for f in fold_results]):.1%}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'timestamp': timestamp,
        'n_samples': total,
        'samples': all_samples,
        'fold_results': fold_results,
    }

    output_path = os.path.join(RESULTS_DIR, f"large_scale_action_mass_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
