"""
Combined Optimal UQ on Real OpenVLA-7B.

Uses the optimal dropout rate p=0.20 (from dropout sweep) combined with
per-dimension temperature calibration (from perdim experiment) and
prompt ensemble to create the best combined uncertainty signal.

Tests on 100 samples across 8 scenarios with:
1. MC Dropout (p=0.20, N=20 passes) — optimal rate
2. Per-dimension temperature calibration (T_d from perdim experiment)
3. Prompt ensemble (4 prompts)
4. Combined signal: weighted average of calibrated entropy + MC std + prompt disagreement

Experiment 11 in the CalibDrive series.
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

OPTIMAL_DROPOUT = 0.20
N_MC_PASSES = 20
N_PROMPTS = 4

# Per-dimension optimal temperatures from perdim experiment
# Dims 0-5: T=5.0, Dim 6 (gripper): T=0.5
PERDIM_TEMPS = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 0.5]

SCENARIOS = {
    'highway': {'n': 15, 'speed': '30', 'difficulty': 'easy'},
    'urban': {'n': 15, 'speed': '15', 'difficulty': 'easy'},
    'night': {'n': 10, 'speed': '25', 'difficulty': 'hard'},
    'rain': {'n': 10, 'speed': '20', 'difficulty': 'hard'},
    'fog': {'n': 10, 'speed': '20', 'difficulty': 'hard'},
    'construction': {'n': 10, 'speed': '10', 'difficulty': 'hard'},
    'ood_noise': {'n': 15, 'speed': '25', 'difficulty': 'ood'},
    'ood_blank': {'n': 15, 'speed': '25', 'difficulty': 'ood'},
}

PROMPT_TEMPLATES = [
    "In: What action should the robot take to drive forward at {speed} m/s safely?\nOut:",
    "In: You are driving at {speed} m/s. What is the safe driving action?\nOut:",
    "In: Navigate safely at {speed} m/s. What action to take?\nOut:",
    "In: Predict the driving action to maintain safe driving at {speed} m/s.\nOut:",
]


def create_scene_image(scenario, idx, size=(256, 256)):
    np.random.seed(idx * 500 + hash(scenario) % 5000)
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
    elif scenario == 'fog':
        img = np.full((*size, 3), 200, dtype=np.uint8)
        img[size[0]//2:] = [180, 180, 185]
    elif scenario == 'construction':
        img = np.zeros((*size, 3), dtype=np.uint8)
        img[:size[0]//2] = [135, 206, 235]
        img[size[0]//2:] = [80, 80, 80]
        img[size[0]//4:size[0]//2, ::4] = [255, 165, 0]
    elif scenario == 'ood_noise':
        img = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
    elif scenario == 'ood_blank':
        img = np.full((*size, 3), 128, dtype=np.uint8)
    else:
        img = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)

    noise = np.random.randint(-10, 10, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def set_dropout_rate(model, rate):
    count = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = rate
            count += 1
    return count


def enable_mc_dropout(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


def single_forward(model, processor, image, prompt, temperatures=None):
    """Run a single forward pass with optional per-dimension temperature scaling."""
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

    dim_confs = []
    dim_entropies = []
    dim_tokens = []
    dim_calibrated_confs = []
    dim_calibrated_entropies = []

    for d, score in enumerate(outputs.scores[:7]):
        logits = score[0, action_start:].float()

        # Raw (uncalibrated)
        probs = torch.softmax(logits, dim=0)
        dim_confs.append(probs.max().item())
        dim_entropies.append(-(probs * torch.log(probs + 1e-10)).sum().item())
        dim_tokens.append(probs.argmax().item())

        # Calibrated with per-dimension temperature
        if temperatures is not None:
            T = temperatures[d] if d < len(temperatures) else 1.0
            cal_probs = torch.softmax(logits / T, dim=0)
            dim_calibrated_confs.append(cal_probs.max().item())
            dim_calibrated_entropies.append(
                -(cal_probs * torch.log(cal_probs + 1e-10)).sum().item()
            )

    geo_conf = np.exp(np.mean(np.log(np.array(dim_confs) + 1e-10)))
    mean_ent = np.mean(dim_entropies)

    result = {
        'geo_conf': float(geo_conf),
        'mean_entropy': float(mean_ent),
        'dim_confs': [float(c) for c in dim_confs],
        'dim_entropies': [float(e) for e in dim_entropies],
        'dim_tokens': dim_tokens,
    }

    if temperatures is not None:
        cal_geo_conf = np.exp(np.mean(np.log(np.array(dim_calibrated_confs) + 1e-10)))
        result['cal_geo_conf'] = float(cal_geo_conf)
        result['cal_mean_entropy'] = float(np.mean(dim_calibrated_entropies))
        result['cal_dim_confs'] = [float(c) for c in dim_calibrated_confs]
        result['cal_dim_entropies'] = [float(e) for e in dim_calibrated_entropies]

    return result


def compute_auroc(scores_pos, scores_neg):
    """Compute AUROC where higher scores indicate positive class."""
    n_correct = sum(1 for p in scores_pos for n in scores_neg if p > n)
    n_ties = sum(0.5 for p in scores_pos for n in scores_neg if p == n)
    n_total = len(scores_pos) * len(scores_neg)
    return (n_correct + n_ties) / n_total if n_total > 0 else 0.5


def main():
    print("=" * 70, flush=True)
    print("COMBINED OPTIMAL UQ ON REAL OpenVLA-7B", flush=True)
    print("=" * 70, flush=True)
    print(f"Dropout rate: {OPTIMAL_DROPOUT}", flush=True)
    print(f"MC passes: {N_MC_PASSES}", flush=True)
    print(f"Prompts: {N_PROMPTS}", flush=True)
    print(f"Per-dim temps: {PERDIM_TEMPS}", flush=True)
    print(flush=True)

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

    n_dropout = set_dropout_rate(model, OPTIMAL_DROPOUT)
    enable_mc_dropout(model)
    print(f"Model loaded. {n_dropout} Dropout layers set to p={OPTIMAL_DROPOUT}.", flush=True)

    # Pre-generate images
    images = {}
    for scenario, config in SCENARIOS.items():
        for i in range(config['n']):
            images[f"{scenario}_{i}"] = create_scene_image(scenario, i)

    total_samples = sum(s['n'] for s in SCENARIOS.values())
    total_inferences = total_samples * (N_MC_PASSES + N_PROMPTS)  # MC passes + prompt ensemble
    print(f"Total samples: {total_samples}", flush=True)
    print(f"Total inferences: {total_inferences}", flush=True)
    print(flush=True)

    all_results = {}
    sample_idx = 0

    for scenario, config in SCENARIOS.items():
        scenario_results = []

        for i in range(config['n']):
            sample_idx += 1
            key = f"{scenario}_{i}"
            image = images[key]
            base_prompt = PROMPT_TEMPLATES[0].format(speed=config['speed'])

            t0 = time.time()

            # === Phase 1: MC Dropout (N=20 passes, optimal p=0.20) ===
            mc_results = []
            for mc in range(N_MC_PASSES):
                r = single_forward(model, processor, image, base_prompt, PERDIM_TEMPS)
                mc_results.append(r)

            mc_confs = [r['geo_conf'] for r in mc_results]
            mc_entropies = [r['mean_entropy'] for r in mc_results]
            mc_cal_entropies = [r['cal_mean_entropy'] for r in mc_results]
            mc_tokens = [r['dim_tokens'] for r in mc_results]

            mc_conf_mean = np.mean(mc_confs)
            mc_conf_std = np.std(mc_confs)
            mc_entropy_mean = np.mean(mc_entropies)
            mc_cal_entropy_mean = np.mean(mc_cal_entropies)

            # Token agreement across MC passes
            tokens_arr = np.array(mc_tokens)
            token_agree = np.mean([len(set(tokens_arr[:, d])) == 1 for d in range(7)])

            # Per-dim MC variance
            dim_confs_arr = np.array([r['dim_confs'] for r in mc_results])
            perdim_mc_std = np.std(dim_confs_arr, axis=0).tolist()

            # === Phase 2: Prompt Ensemble (4 prompts, single pass each) ===
            prompt_results = []
            for p_idx in range(N_PROMPTS):
                prompt = PROMPT_TEMPLATES[p_idx].format(speed=config['speed'])
                r = single_forward(model, processor, image, prompt, PERDIM_TEMPS)
                prompt_results.append(r)

            prompt_confs = [r['geo_conf'] for r in prompt_results]
            prompt_entropies = [r['mean_entropy'] for r in prompt_results]
            prompt_tokens = [r['dim_tokens'] for r in prompt_results]

            prompt_conf_std = np.std(prompt_confs)
            prompt_entropy_std = np.std(prompt_entropies)

            # Prompt token agreement
            ptok_arr = np.array(prompt_tokens)
            prompt_token_agree = np.mean([len(set(ptok_arr[:, d])) == 1 for d in range(7)])

            elapsed = time.time() - t0

            # === Combined uncertainty signals ===
            # Signal 1: Calibrated mean entropy (from MC average)
            sig_cal_entropy = mc_cal_entropy_mean
            # Signal 2: MC confidence std
            sig_mc_std = mc_conf_std
            # Signal 3: Prompt disagreement (entropy std across prompts)
            sig_prompt_disagree = prompt_entropy_std
            # Signal 4: Max per-dim MC std
            sig_max_perdim_std = max(perdim_mc_std)
            # Signal 5: Combined (RMS of normalized signals)
            # Normalize each to [0,1] range approximately
            sig_combined = np.sqrt(
                (sig_cal_entropy / 5.55) ** 2 +  # max entropy ~5.55
                (sig_mc_std / 0.15) ** 2 +  # rough max
                (sig_prompt_disagree / 0.5) ** 2 +  # rough max
                (sig_max_perdim_std / 0.2) ** 2  # rough max
            ) / 2.0  # normalize

            sample_result = {
                'scenario': scenario,
                'difficulty': config['difficulty'],
                'idx': i,
                # MC Dropout results
                'mc_conf_mean': float(mc_conf_mean),
                'mc_conf_std': float(mc_conf_std),
                'mc_entropy_mean': float(mc_entropy_mean),
                'mc_cal_entropy_mean': float(mc_cal_entropy_mean),
                'mc_token_agreement': float(token_agree),
                'perdim_mc_std': [float(s) for s in perdim_mc_std],
                # Prompt ensemble results
                'prompt_conf_std': float(prompt_conf_std),
                'prompt_entropy_std': float(prompt_entropy_std),
                'prompt_token_agreement': float(prompt_token_agree),
                # Combined signals
                'sig_cal_entropy': float(sig_cal_entropy),
                'sig_mc_std': float(sig_mc_std),
                'sig_prompt_disagree': float(sig_prompt_disagree),
                'sig_max_perdim_std': float(sig_max_perdim_std),
                'sig_combined': float(sig_combined),
                'elapsed': float(elapsed),
            }
            scenario_results.append(sample_result)

            if i % 5 == 0 or i == config['n'] - 1:
                print(f"  [{sample_idx}/{total_samples}] {key}: "
                      f"cal_ent={mc_cal_entropy_mean:.3f}, mc_std={mc_conf_std:.4f}, "
                      f"prompt_std={prompt_conf_std:.4f}, combined={sig_combined:.4f} "
                      f"({elapsed:.1f}s)", flush=True)

        all_results[scenario] = scenario_results

    # ===================================================================
    # Analysis
    # ===================================================================
    print("\n" + "=" * 70, flush=True)
    print("COMBINED OPTIMAL UQ ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    # Per-scenario summary
    print("\n1. Per-Scenario Signal Summary", flush=True)
    print("-" * 90, flush=True)
    print(f"{'Scenario':<15} {'N':>3} {'CalEnt':>8} {'MCStd':>8} {'PromptStd':>10} "
          f"{'Combined':>10} {'TokAgree':>10}", flush=True)
    print("-" * 90, flush=True)

    for scenario in SCENARIOS:
        samples = all_results[scenario]
        n = len(samples)
        cal_ent = np.mean([s['sig_cal_entropy'] for s in samples])
        mc_std = np.mean([s['sig_mc_std'] for s in samples])
        prompt_std = np.mean([s['sig_prompt_disagree'] for s in samples])
        combined = np.mean([s['sig_combined'] for s in samples])
        tok_agree = np.mean([s['mc_token_agreement'] for s in samples])
        print(f"{scenario:<15} {n:>3} {cal_ent:>8.3f} {mc_std:>8.4f} {prompt_std:>10.4f} "
              f"{combined:>10.4f} {tok_agree:>10.3f}", flush=True)

    # AUROC comparison
    print("\n2. AUROC Comparison (Easy vs OOD)", flush=True)
    print("-" * 60, flush=True)

    easy_samples = []
    ood_samples = []
    hard_samples = []
    for scenario in SCENARIOS:
        for s in all_results[scenario]:
            if s['difficulty'] == 'easy':
                easy_samples.append(s)
            elif s['difficulty'] == 'ood':
                ood_samples.append(s)
            elif s['difficulty'] == 'hard':
                hard_samples.append(s)

    signals = {
        'Cal Entropy': ('sig_cal_entropy', True),     # higher = more uncertain
        'MC Conf Std': ('sig_mc_std', True),
        'Prompt Disagree': ('sig_prompt_disagree', True),
        'Max PerDim Std': ('sig_max_perdim_std', True),
        'Combined (RMS)': ('sig_combined', True),
        'Raw Entropy': ('mc_entropy_mean', True),
        'Neg Confidence': ('mc_conf_mean', False),     # lower = more uncertain
    }

    print(f"{'Signal':<20} {'AUROC easy-vs-ood':>18} {'AUROC easy-vs-hard':>20}", flush=True)
    print("-" * 60, flush=True)

    for name, (key, higher_uncertain) in signals.items():
        if higher_uncertain:
            easy_scores = [s[key] for s in easy_samples]
            ood_scores = [s[key] for s in ood_samples]
            hard_scores = [s[key] for s in hard_samples]
            auroc_ood = compute_auroc(ood_scores, easy_scores)
            auroc_hard = compute_auroc(hard_scores, easy_scores)
        else:
            easy_scores = [-s[key] for s in easy_samples]
            ood_scores = [-s[key] for s in ood_samples]
            hard_scores = [-s[key] for s in hard_samples]
            auroc_ood = compute_auroc(ood_scores, easy_scores)
            auroc_hard = compute_auroc(hard_scores, easy_scores)

        print(f"{name:<20} {auroc_ood:>18.3f} {auroc_hard:>20.3f}", flush=True)

    # Selective prediction
    print("\n3. Selective Prediction (OOD Rejection at Coverage Levels)", flush=True)
    print("-" * 70, flush=True)

    all_samples = []
    for scenario in SCENARIOS:
        all_samples.extend(all_results[scenario])

    for coverage_pct in [100, 90, 80, 70, 60, 50]:
        n_keep = int(len(all_samples) * coverage_pct / 100)

        print(f"\n  Coverage={coverage_pct}%:", flush=True)
        for name, (key, higher_uncertain) in signals.items():
            if higher_uncertain:
                sorted_samples = sorted(all_samples, key=lambda s: s[key])
            else:
                sorted_samples = sorted(all_samples, key=lambda s: -s[key])

            kept = sorted_samples[:n_keep]
            rejected = sorted_samples[n_keep:]

            n_ood_rejected = sum(1 for s in rejected if s['difficulty'] == 'ood')
            total_ood = sum(1 for s in all_samples if s['difficulty'] == 'ood')
            ood_reject_rate = n_ood_rejected / total_ood * 100 if total_ood > 0 else 0

            n_hard_rejected = sum(1 for s in rejected if s['difficulty'] == 'hard')
            total_hard = sum(1 for s in all_samples if s['difficulty'] == 'hard')
            hard_reject_rate = n_hard_rejected / total_hard * 100 if total_hard > 0 else 0

            print(f"    {name:<20}: OOD reject={ood_reject_rate:>5.1f}%, "
                  f"Hard reject={hard_reject_rate:>5.1f}%", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'timestamp': timestamp,
        'config': {
            'dropout_rate': OPTIMAL_DROPOUT,
            'n_mc_passes': N_MC_PASSES,
            'n_prompts': N_PROMPTS,
            'perdim_temps': PERDIM_TEMPS,
        },
        'per_sample': {
            scenario: [
                {k: v for k, v in s.items()}
                for s in samples
            ]
            for scenario, samples in all_results.items()
        },
    }

    output_path = os.path.join(RESULTS_DIR, f"combined_optimal_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
