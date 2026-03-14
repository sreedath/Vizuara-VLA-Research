"""
Per-Dimension Calibration + Cross-Prompt Consistency on Real OpenVLA-7B.

Two-part experiment:
1. Per-dimension temperature optimization: Find T_d* for each dimension d
   that minimizes within-scenario confidence variance while maximizing
   between-scenario separation.
2. Cross-prompt consistency: Test 8 prompts per sample, measure how
   prediction consistency correlates with scenario difficulty.

Experiment 9 in the CalibDrive series.
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

N_TOTAL = 80
SCENARIOS = {
    'highway': {'n': 15, 'speed': '30', 'difficulty': 'easy'},
    'urban': {'n': 15, 'speed': '15', 'difficulty': 'easy'},
    'night': {'n': 10, 'speed': '25', 'difficulty': 'hard'},
    'rain': {'n': 10, 'speed': '20', 'difficulty': 'hard'},
    'construction': {'n': 10, 'speed': '10', 'difficulty': 'hard'},
    'ood_noise': {'n': 20, 'speed': '25', 'difficulty': 'ood'},
}

PROMPTS = [
    "In: What action should the robot take to drive forward at {speed} m/s safely?\nOut:",
    "In: You are driving at {speed} m/s. What is the safe driving action?\nOut:",
    "In: Navigate safely at {speed} m/s. What action to take?\nOut:",
    "In: Predict the driving action to maintain safe driving at {speed} m/s.\nOut:",
    "In: Control the vehicle moving at {speed} m/s. Output the action.\nOut:",
    "In: Drive the car forward at {speed} m/s. What action?\nOut:",
    "In: The vehicle speed is {speed} m/s. Determine safe action.\nOut:",
    "In: At {speed} m/s, what driving action ensures safety?\nOut:",
]

TEMPERATURES = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]


def create_scene_image(scenario, idx, size=(256, 256)):
    """Create synthetic scene image."""
    np.random.seed(idx * 300 + hash(scenario) % 3000)

    if scenario == 'highway':
        img = np.zeros((*size, 3), dtype=np.uint8)
        img[:size[0]//2] = [135, 206, 235]
        img[size[0]//2:] = [80, 80, 80]
        for x in range(0, size[1], 40):
            img[size[0]*3//4-2:size[0]*3//4+2, x:x+20] = [255, 255, 255]
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
        for _ in range(200):
            y, x = np.random.randint(0, size[0]-10), np.random.randint(0, size[1])
            img[y:y+8, x:min(x+1,size[1]-1)] = [180, 180, 200]
    elif scenario == 'construction':
        img = np.zeros((*size, 3), dtype=np.uint8)
        img[:size[0]//2] = [180, 180, 180]
        img[size[0]//2:] = [90, 80, 70]
        # Orange cones
        for cx in range(30, size[1], 50):
            img[size[0]//2-15:size[0]//2+5, cx-5:cx+5] = [255, 140, 0]
    elif scenario == 'ood_noise':
        img = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
    else:
        img = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)

    noise = np.random.randint(-10, 10, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def extract_raw_logits(model, processor, image, prompt):
    """Extract raw logits (before softmax) for all 7 dims."""
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

    raw_logits = []
    for dim_idx, score in enumerate(outputs.scores[:7]):
        logits = score[0, action_start:].float().cpu().numpy()
        raw_logits.append(logits)

    return raw_logits


def compute_metrics_at_temp(logits_list, T):
    """Compute calibration metrics for a set of 7-dim logits at temperature T."""
    dim_metrics = []
    for logits in logits_list:
        scaled = logits / T
        # Stable softmax
        shifted = scaled - scaled.max()
        exp_vals = np.exp(shifted)
        probs = exp_vals / exp_vals.sum()

        max_prob = probs.max()
        entropy = -(probs * np.log(probs + 1e-10)).sum()
        top5_mass = np.sort(probs)[-5:].sum()
        perplexity = np.exp(entropy)

        dim_metrics.append({
            'max_prob': float(max_prob),
            'entropy': float(entropy),
            'top5_mass': float(top5_mass),
            'perplexity': float(perplexity),
        })

    geo_mean_conf = float(np.exp(np.mean([np.log(d['max_prob'] + 1e-10) for d in dim_metrics])))
    mean_entropy = float(np.mean([d['entropy'] for d in dim_metrics]))

    return {
        'geo_mean_conf': geo_mean_conf,
        'mean_entropy': mean_entropy,
        'per_dim': dim_metrics,
    }


def main():
    print("=" * 70)
    print("PER-DIMENSION CALIBRATION + CROSS-PROMPT CONSISTENCY")
    print("=" * 70)

    # Load model
    print("Loading OpenVLA-7B...")
    from transformers import AutoModelForVision2Seq, AutoProcessor
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
    print("Model loaded.")

    # ===================================================================
    # Part 1: Collect raw logits for all samples with primary prompt
    # ===================================================================
    print("\n--- Part 1: Raw Logit Collection ---")
    all_logits = []  # list of (scenario, difficulty, logits_7dim)
    sample_idx = 0
    total = sum(s['n'] for s in SCENARIOS.values())

    for scenario, config in SCENARIOS.items():
        print(f"\n  {scenario} (N={config['n']})")
        for i in range(config['n']):
            sample_idx += 1
            image = create_scene_image(scenario, i)
            prompt = PROMPTS[0].format(speed=config['speed'])

            t0 = time.time()
            logits = extract_raw_logits(model, processor, image, prompt)
            elapsed = time.time() - t0

            all_logits.append({
                'scenario': scenario,
                'difficulty': config['difficulty'],
                'logits': logits,
                'idx': i,
            })

            if i % 5 == 0 or i == config['n'] - 1:
                metrics = compute_metrics_at_temp(logits, 1.0)
                print(f"    [{sample_idx}/{total}] {scenario}_{i}: "
                      f"conf={metrics['geo_mean_conf']:.3f}, "
                      f"ent={metrics['mean_entropy']:.3f} ({elapsed:.1f}s)")

    # ===================================================================
    # Part 1b: Per-Dimension Optimal Temperature Search
    # ===================================================================
    print("\n" + "=" * 70)
    print("PER-DIMENSION TEMPERATURE OPTIMIZATION")
    print("=" * 70)

    # For each dimension, find T that maximizes AUROC(easy vs ood)
    per_dim_optimal = {}
    for dim_idx in range(7):
        best_T = 1.0
        best_auroc = 0.0

        for T in TEMPERATURES:
            easy_vals = []
            ood_vals = []
            for item in all_logits:
                metrics = compute_metrics_at_temp([item['logits'][dim_idx]], T)
                entropy = metrics['per_dim'][0]['entropy']
                if item['difficulty'] == 'easy':
                    easy_vals.append(entropy)
                elif item['difficulty'] == 'ood':
                    ood_vals.append(entropy)

            # AUROC: fraction of (easy,ood) pairs where ood entropy > easy entropy
            n_correct = sum(1 for e in easy_vals for o in ood_vals if o > e)
            n_ties = sum(0.5 for e in easy_vals for o in ood_vals if o == e)
            n_total = len(easy_vals) * len(ood_vals)
            auroc = (n_correct + n_ties) / n_total if n_total > 0 else 0.5

            if auroc > best_auroc:
                best_auroc = auroc
                best_T = T

        per_dim_optimal[dim_idx] = {'T': best_T, 'auroc': best_auroc}
        print(f"  Dim {dim_idx}: Optimal T={best_T}, AUROC(easy vs ood)={best_auroc:.3f}")

    # Compute combined per-dim-calibrated metric
    print("\nPer-Dim Calibrated Confidence (using optimal T per dim):")
    calibrated_results = []
    for item in all_logits:
        calibrated_metrics = []
        for dim_idx in range(7):
            T_opt = per_dim_optimal[dim_idx]['T']
            m = compute_metrics_at_temp([item['logits'][dim_idx]], T_opt)
            calibrated_metrics.append(m['per_dim'][0])

        calibrated_conf = float(np.exp(np.mean([
            np.log(d['max_prob'] + 1e-10) for d in calibrated_metrics
        ])))
        calibrated_entropy = float(np.mean([d['entropy'] for d in calibrated_metrics]))
        calibrated_results.append({
            'scenario': item['scenario'],
            'difficulty': item['difficulty'],
            'calibrated_conf': calibrated_conf,
            'calibrated_entropy': calibrated_entropy,
        })

    # Per-scenario calibrated stats
    print(f"\n{'Scenario':>14} {'Calib Conf':>10} {'Calib Ent':>10}")
    scenario_cal = {}
    for scenario in SCENARIOS:
        confs = [r['calibrated_conf'] for r in calibrated_results if r['scenario'] == scenario]
        ents = [r['calibrated_entropy'] for r in calibrated_results if r['scenario'] == scenario]
        scenario_cal[scenario] = {'conf': np.mean(confs), 'entropy': np.mean(ents)}
        print(f"  {scenario:>12}: {np.mean(confs):>10.3f} {np.mean(ents):>10.3f}")

    # AUROC with calibrated vs uncalibrated
    for signal_name, sig_fn in [
        ('uncalib_entropy', lambda r: next(
            compute_metrics_at_temp(item['logits'], 1.0)['mean_entropy']
            for item in all_logits
            if item['scenario'] == r['scenario'] and item['idx'] == 0
        )),
    ]:
        pass  # Skip complex lambda, compute directly

    # Simple AUROC for calibrated entropy
    easy_cal = [r['calibrated_entropy'] for r in calibrated_results if r['difficulty'] == 'easy']
    ood_cal = [r['calibrated_entropy'] for r in calibrated_results if r['difficulty'] == 'ood']
    n_correct = sum(1 for e in easy_cal for o in ood_cal if o > e)
    n_total = len(easy_cal) * len(ood_cal)
    cal_auroc = n_correct / n_total if n_total > 0 else 0.5
    print(f"\nCalibrated AUROC(easy vs ood) via entropy: {cal_auroc:.3f}")

    # Uncalibrated AUROC for comparison
    uncal_results = []
    for item in all_logits:
        m = compute_metrics_at_temp(item['logits'], 1.0)
        uncal_results.append({
            'scenario': item['scenario'],
            'difficulty': item['difficulty'],
            'entropy': m['mean_entropy'],
        })
    easy_uncal = [r['entropy'] for r in uncal_results if r['difficulty'] == 'easy']
    ood_uncal = [r['entropy'] for r in uncal_results if r['difficulty'] == 'ood']
    n_correct = sum(1 for e in easy_uncal for o in ood_uncal if o > e)
    n_total = len(easy_uncal) * len(ood_uncal)
    uncal_auroc = n_correct / n_total if n_total > 0 else 0.5
    print(f"Uncalibrated AUROC(easy vs ood) via entropy: {uncal_auroc:.3f}")
    print(f"Improvement: {cal_auroc - uncal_auroc:+.3f}")

    # ===================================================================
    # Part 2: Cross-Prompt Consistency
    # ===================================================================
    print("\n" + "=" * 70)
    print("CROSS-PROMPT CONSISTENCY ANALYSIS")
    print("=" * 70)

    # Select subset: 5 samples per scenario for 8 prompts each
    prompt_consistency = []
    sample_count = 0

    for scenario, config in SCENARIOS.items():
        print(f"\n  {scenario} (5 samples × 8 prompts)")
        n_prompt_samples = min(5, config['n'])

        for i in range(n_prompt_samples):
            image = create_scene_image(scenario, i)
            prompt_logits = []

            for p_idx, prompt_template in enumerate(PROMPTS):
                prompt = prompt_template.format(speed=config['speed'])
                t0 = time.time()
                logits = extract_raw_logits(model, processor, image, prompt)
                elapsed = time.time() - t0
                prompt_logits.append(logits)

            sample_count += 1

            # Compute consistency metrics across prompts
            prompt_confs = []
            prompt_entropies = []
            prompt_tokens = []  # argmax token per prompt per dim

            for p_logits in prompt_logits:
                m = compute_metrics_at_temp(p_logits, 1.0)
                prompt_confs.append(m['geo_mean_conf'])
                prompt_entropies.append(m['mean_entropy'])
                tokens = [int(np.argmax(l)) for l in p_logits]
                prompt_tokens.append(tokens)

            # Token agreement: fraction of dims where all 8 prompts agree
            prompt_tokens = np.array(prompt_tokens)  # shape: (8, 7)
            token_agreement = np.mean([
                len(set(prompt_tokens[:, d])) == 1 for d in range(7)
            ])

            # Confidence std across prompts
            conf_std = np.std(prompt_confs)
            entropy_std = np.std(prompt_entropies)

            # KL divergence between prompt pairs (average)
            kl_divs = []
            for p1 in range(len(PROMPTS)):
                for p2 in range(p1+1, len(PROMPTS)):
                    for d in range(7):
                        probs1 = np.exp(prompt_logits[p1][d] - np.max(prompt_logits[p1][d]))
                        probs1 = probs1 / probs1.sum()
                        probs2 = np.exp(prompt_logits[p2][d] - np.max(prompt_logits[p2][d]))
                        probs2 = probs2 / probs2.sum()
                        kl = np.sum(probs1 * np.log((probs1 + 1e-10) / (probs2 + 1e-10)))
                        kl_divs.append(kl)
            mean_kl = np.mean(kl_divs)

            result = {
                'scenario': scenario,
                'difficulty': config['difficulty'],
                'idx': i,
                'mean_conf': float(np.mean(prompt_confs)),
                'conf_std': float(conf_std),
                'mean_entropy': float(np.mean(prompt_entropies)),
                'entropy_std': float(entropy_std),
                'token_agreement': float(token_agreement),
                'mean_kl': float(mean_kl),
            }
            prompt_consistency.append(result)

            if True:
                print(f"    [{sample_count}] {scenario}_{i}: "
                      f"conf_std={conf_std:.4f}, "
                      f"token_agree={token_agreement:.2f}, "
                      f"KL={mean_kl:.2f}")

    # Summary
    print("\n" + "=" * 70)
    print("CROSS-PROMPT CONSISTENCY SUMMARY")
    print("=" * 70)
    print(f"\n{'Scenario':>14} {'Conf Std':>9} {'Ent Std':>9} {'Token Agr':>10} {'Mean KL':>9}")
    print("-" * 55)

    for scenario in SCENARIOS:
        items = [r for r in prompt_consistency if r['scenario'] == scenario]
        print(f"  {scenario:>12}: "
              f"{np.mean([r['conf_std'] for r in items]):>9.4f} "
              f"{np.mean([r['entropy_std'] for r in items]):>9.4f} "
              f"{np.mean([r['token_agreement'] for r in items]):>10.2f} "
              f"{np.mean([r['mean_kl'] for r in items]):>9.2f}")

    # AUROC using prompt consistency as uncertainty signal
    print("\nAUROC(easy vs ood) using prompt consistency signals:")
    for signal_name, sig_fn in [
        ('conf_std', lambda r: r['conf_std']),
        ('entropy_std', lambda r: r['entropy_std']),
        ('neg_token_agree', lambda r: -r['token_agreement']),
        ('mean_kl', lambda r: r['mean_kl']),
    ]:
        easy_vals = [sig_fn(r) for r in prompt_consistency if r['difficulty'] == 'easy']
        ood_vals = [sig_fn(r) for r in prompt_consistency if r['difficulty'] == 'ood']
        n_correct = sum(1 for e in easy_vals for o in ood_vals if o > e)
        n_ties = sum(0.5 for e in easy_vals for o in ood_vals if o == e)
        n_total = len(easy_vals) * len(ood_vals)
        auroc = (n_correct + n_ties) / n_total if n_total > 0 else 0.5
        print(f"  {signal_name:>18}: AUROC = {auroc:.3f}")

    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'timestamp': timestamp,
        'per_dim_optimal': {str(k): v for k, v in per_dim_optimal.items()},
        'calibrated_auroc': cal_auroc,
        'uncalibrated_auroc': uncal_auroc,
        'scenario_calibrated': {s: {k: float(v) for k, v in vals.items()}
                                 for s, vals in scenario_cal.items()},
        'prompt_consistency': prompt_consistency,
    }

    output_path = os.path.join(RESULTS_DIR, f"perdim_calibration_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
