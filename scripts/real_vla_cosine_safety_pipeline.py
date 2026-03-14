"""
Updated CalibDrive Safety Pipeline with Cosine Distance on Real OpenVLA-7B.

Replaces action mass with cosine distance as the primary OOD detector
in the 4-level safety pipeline:
1. Cosine distance > threshold → STOP (OOD)
2. Entropy > threshold → SLOW (hard scenario)
3. Per-dim mass below threshold → CAUTION
4. Default → PROCEED

Tests calibrated thresholds at multiple alpha levels.
Compares with action-mass-based pipeline from Experiment 25.

Experiment 30 in the CalibDrive series.
"""
import os
import json
import time
import datetime
import numpy as np
import torch
from PIL import Image, ImageDraw

RESULTS_DIR = "/workspace/Vizuara-VLA-Research/experiments"
os.makedirs(RESULTS_DIR, exist_ok=True)

SCENARIOS = {
    'highway': {'n': 25, 'speed': '30', 'difficulty': 'easy', 'correct_action': 'PROCEED'},
    'urban': {'n': 25, 'speed': '15', 'difficulty': 'easy', 'correct_action': 'PROCEED'},
    'ood_noise': {'n': 12, 'speed': '25', 'difficulty': 'ood', 'correct_action': 'STOP'},
    'ood_blank': {'n': 12, 'speed': '25', 'difficulty': 'ood', 'correct_action': 'STOP'},
    'ood_indoor': {'n': 12, 'speed': '25', 'difficulty': 'ood', 'correct_action': 'STOP'},
    'ood_inverted': {'n': 12, 'speed': '30', 'difficulty': 'ood', 'correct_action': 'STOP'},
    'ood_checker': {'n': 12, 'speed': '25', 'difficulty': 'ood', 'correct_action': 'STOP'},
    'ood_blackout': {'n': 12, 'speed': '25', 'difficulty': 'ood', 'correct_action': 'STOP'},
}


def create_scene_image(scenario, idx, size=(256, 256)):
    np.random.seed(idx * 3000 + hash(scenario) % 30000)
    if scenario == 'highway':
        img = np.zeros((*size, 3), dtype=np.uint8)
        img[:size[0]//2] = [135, 206, 235]
        img[size[0]//2:] = [80, 80, 80]
    elif scenario == 'urban':
        img = np.zeros((*size, 3), dtype=np.uint8)
        img[:size[0]//3] = [135, 206, 235]
        img[size[0]//3:size[0]//2] = [139, 119, 101]
        img[size[0]//2:] = [80, 80, 80]
    elif scenario == 'ood_noise':
        img = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
    elif scenario == 'ood_blank':
        img = np.full((*size, 3), 128, dtype=np.uint8)
    elif scenario == 'ood_indoor':
        img = np.zeros((*size, 3), dtype=np.uint8)
        img[:size[0]//3] = [210, 180, 140]
        img[size[0]//3:2*size[0]//3] = [180, 120, 80]
        img[2*size[0]//3:] = [100, 70, 50]
    elif scenario == 'ood_inverted':
        img = np.zeros((*size, 3), dtype=np.uint8)
        img[:size[0]//2] = [135, 206, 235]
        img[size[0]//2:] = [80, 80, 80]
        img = 255 - img
    elif scenario == 'ood_checker':
        img = np.zeros((*size, 3), dtype=np.uint8)
        block = 32
        for y in range(0, size[0], block):
            for x in range(0, size[1], block):
                if (y // block + x // block) % 2 == 0:
                    img[y:y+block, x:x+block] = [255, 255, 255]
    elif scenario == 'ood_blackout':
        img = np.full((*size, 3), 5, dtype=np.uint8)
    else:
        img = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
    noise = np.random.randint(-3, 3, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def compute_auroc(pos_scores, neg_scores):
    n_correct = sum(1 for p in pos_scores for n in neg_scores if p > n)
    n_ties = sum(0.5 for p in pos_scores for n in neg_scores if p == n)
    n_total = len(pos_scores) * len(neg_scores)
    return (n_correct + n_ties) / n_total if n_total > 0 else 0.5


SAFETY_ORDER = {'STOP': 3, 'SLOW': 2, 'CAUTION': 1, 'PROCEED': 0}


def is_safe(decision, correct):
    return SAFETY_ORDER[decision] >= SAFETY_ORDER[correct]


def main():
    print("=" * 70, flush=True)
    print("COSINE DISTANCE SAFETY PIPELINE ON REAL OpenVLA-7B", flush=True)
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
    all_hidden_states = []
    sample_idx = 0

    for scenario, config in SCENARIOS.items():
        for i in range(config['n']):
            sample_idx += 1
            image = create_scene_image(scenario, i)
            p = prompt.format(speed=config['speed'])

            inputs = processor(p, image).to(model.device, dtype=torch.bfloat16)

            t0 = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=7,
                    do_sample=False,
                    output_scores=True,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                )

            # Extract hidden state
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                last_step_hidden = outputs.hidden_states[-1]
                if isinstance(last_step_hidden, tuple):
                    last_layer = last_step_hidden[-1]
                    hidden = last_layer[0, -1, :].float().cpu().numpy()
                else:
                    hidden = last_step_hidden[0, -1, :].float().cpu().numpy()
            else:
                hidden = np.zeros(4096)

            # Get action mass, entropy, per-dim masses
            vocab_size = outputs.scores[0].shape[-1]
            action_start = vocab_size - 256
            dim_masses = []
            dim_entropies = []
            for score in outputs.scores[:7]:
                full_logits = score[0].float()
                full_probs = torch.softmax(full_logits, dim=0).cpu().numpy()
                action_probs = full_probs[action_start:]
                dim_masses.append(float(action_probs.sum()))
                action_norm = action_probs / (action_probs.sum() + 1e-10)
                dim_entropies.append(float(-(action_norm * np.log(action_norm + 1e-10)).sum()))

            elapsed = time.time() - t0

            sample = {
                'scenario': scenario,
                'difficulty': config['difficulty'],
                'correct_action': config['correct_action'],
                'idx': i,
                'action_mass': float(np.mean(dim_masses)),
                'min_dim_mass': float(min(dim_masses)),
                'entropy': float(np.mean(dim_entropies)),
                'dim_masses': dim_masses,
            }
            all_samples.append(sample)
            all_hidden_states.append(hidden)

            if i % 5 == 0 or i == config['n'] - 1:
                print(f"  [{sample_idx}/{total}] {scenario}_{i}: "
                      f"mass={sample['action_mass']:.4f}, ent={sample['entropy']:.3f} "
                      f"({elapsed:.1f}s)", flush=True)

    # ===================================================================
    # Calibration: Use first half of easy samples
    # ===================================================================
    hidden_arr = np.array(all_hidden_states)
    easy_idxs = [i for i, s in enumerate(all_samples) if s['difficulty'] == 'easy']
    np.random.seed(42)
    np.random.shuffle(easy_idxs)
    n_cal = len(easy_idxs) // 2
    cal_idxs = easy_idxs[:n_cal]
    test_idxs = easy_idxs[n_cal:] + [i for i, s in enumerate(all_samples) if s['difficulty'] != 'easy']

    print(f"\nCalibration: {n_cal} easy samples", flush=True)
    print(f"Test: {len(test_idxs)} samples ({len(easy_idxs) - n_cal} easy + "
          f"{sum(1 for s in all_samples if s['difficulty'] == 'ood')} OOD)", flush=True)

    # Compute cosine distance to calibration centroid
    cal_hidden = hidden_arr[cal_idxs]
    cal_mean = np.mean(cal_hidden, axis=0)
    cal_mean_norm = cal_mean / (np.linalg.norm(cal_mean) + 1e-10)

    for s_idx in range(len(all_samples)):
        h_norm = hidden_arr[s_idx] / (np.linalg.norm(hidden_arr[s_idx]) + 1e-10)
        all_samples[s_idx]['cos_dist'] = 1.0 - float(np.dot(h_norm, cal_mean_norm))

    # Calibration thresholds from calibration set
    cal_cos_dists = [all_samples[i]['cos_dist'] for i in cal_idxs]
    cal_entropies = [all_samples[i]['entropy'] for i in cal_idxs]
    cal_masses = [all_samples[i]['action_mass'] for i in cal_idxs]

    # ===================================================================
    # Pipeline Evaluation at Multiple Alpha Levels
    # ===================================================================
    print("\n" + "=" * 70, flush=True)
    print("SAFETY PIPELINE COMPARISON", flush=True)
    print("=" * 70, flush=True)

    alphas = [0.05, 0.10, 0.15, 0.20, 0.30]
    pipeline_results = {}

    for alpha in alphas:
        # Cosine distance thresholds
        cos_threshold = np.percentile(cal_cos_dists, 100 * (1 - alpha))
        ent_threshold = np.percentile(cal_entropies, 100 * (1 - alpha))
        mass_threshold = np.percentile(cal_masses, 100 * alpha)  # Lower mass = OOD

        # Cosine-based pipeline
        cos_decisions = []
        mass_decisions = []

        for s_idx in test_idxs:
            s = all_samples[s_idx]

            # Cosine pipeline
            if s['cos_dist'] > cos_threshold:
                cos_dec = 'STOP'
            elif s['entropy'] > ent_threshold:
                cos_dec = 'SLOW'
            elif any(dm < mass_threshold for dm in s['dim_masses']):
                cos_dec = 'CAUTION'
            else:
                cos_dec = 'PROCEED'
            cos_decisions.append(cos_dec)

            # Mass-based pipeline (from Exp 25)
            if s['action_mass'] < mass_threshold:
                mass_dec = 'STOP'
            elif s['entropy'] > ent_threshold:
                mass_dec = 'SLOW'
            elif any(dm < mass_threshold for dm in s['dim_masses']):
                mass_dec = 'CAUTION'
            else:
                mass_dec = 'PROCEED'
            mass_decisions.append(mass_dec)

        # Evaluate both pipelines
        test_samples = [all_samples[i] for i in test_idxs]

        for pipeline_name, decisions in [('cosine', cos_decisions), ('mass', mass_decisions)]:
            n_correct = sum(1 for s, d in zip(test_samples, decisions)
                          if d == s['correct_action'])
            n_safe = sum(1 for s, d in zip(test_samples, decisions)
                        if is_safe(d, s['correct_action']))
            n_total = len(test_samples)

            ood_samples = [(s, d) for s, d in zip(test_samples, decisions) if s['difficulty'] == 'ood']
            easy_samples = [(s, d) for s, d in zip(test_samples, decisions) if s['difficulty'] == 'easy']

            ood_stopped = sum(1 for s, d in ood_samples if d == 'STOP')
            easy_proceed = sum(1 for s, d in easy_samples if d == 'PROCEED')

            key = f'{pipeline_name}_a{alpha}'
            pipeline_results[key] = {
                'accuracy': n_correct / n_total,
                'safety': n_safe / n_total,
                'ood_stop': ood_stopped / len(ood_samples) if ood_samples else 0,
                'easy_proceed': easy_proceed / len(easy_samples) if easy_samples else 0,
            }

    # Print comparison
    print("\n1. Pipeline Comparison Across Alpha Levels", flush=True)
    print("-" * 90, flush=True)
    print(f"{'Alpha':>6} | {'Signal':>8} | {'Accuracy':>8} | {'Safety':>8} | "
          f"{'OOD→STOP':>8} | {'Easy→PROCEED':>12}", flush=True)
    print("-" * 90, flush=True)

    for alpha in alphas:
        for pipeline in ['cosine', 'mass']:
            r = pipeline_results[f'{pipeline}_a{alpha}']
            print(f"{alpha:>6.2f} | {pipeline:>8} | {r['accuracy']:>8.1%} | {r['safety']:>8.1%} | "
                  f"{r['ood_stop']:>8.1%} | {r['easy_proceed']:>12.1%}", flush=True)
        print("-" * 90, flush=True)

    # Per-scenario breakdown at best alpha
    print("\n2. Per-Scenario Breakdown (α=0.20)", flush=True)
    print("-" * 80, flush=True)

    alpha = 0.20
    cos_threshold = np.percentile(cal_cos_dists, 100 * (1 - alpha))
    ent_threshold = np.percentile(cal_entropies, 100 * (1 - alpha))
    mass_threshold = np.percentile(cal_masses, 100 * alpha)

    print(f"  Thresholds: cos_dist > {cos_threshold:.4f}, "
          f"entropy > {ent_threshold:.4f}, mass < {mass_threshold:.4f}", flush=True)
    print(flush=True)

    for scenario in SCENARIOS:
        scenario_idxs = [i for i in test_idxs if all_samples[i]['scenario'] == scenario]
        if not scenario_idxs:
            continue

        cos_decs = []
        mass_decs = []
        for s_idx in scenario_idxs:
            s = all_samples[s_idx]

            if s['cos_dist'] > cos_threshold:
                cos_decs.append('STOP')
            elif s['entropy'] > ent_threshold:
                cos_decs.append('SLOW')
            else:
                cos_decs.append('PROCEED')

            if s['action_mass'] < mass_threshold:
                mass_decs.append('STOP')
            elif s['entropy'] > ent_threshold:
                mass_decs.append('SLOW')
            else:
                mass_decs.append('PROCEED')

        correct_action = SCENARIOS[scenario]['correct_action']

        cos_correct = sum(1 for d in cos_decs if d == correct_action)
        cos_safe = sum(1 for d in cos_decs if is_safe(d, correct_action))
        mass_correct = sum(1 for d in mass_decs if d == correct_action)
        mass_safe = sum(1 for d in mass_decs if is_safe(d, correct_action))
        n = len(scenario_idxs)

        # Decision distribution for cosine pipeline
        cos_dist_counts = {}
        for d in cos_decs:
            cos_dist_counts[d] = cos_dist_counts.get(d, 0) + 1

        print(f"  {scenario:<15} (target: {correct_action}):", flush=True)
        print(f"    Cosine: {cos_correct}/{n} correct, {cos_safe}/{n} safe "
              f"| {cos_dist_counts}", flush=True)
        print(f"    Mass:   {mass_correct}/{n} correct, {mass_safe}/{n} safe", flush=True)

    # AUROC comparison
    print("\n3. AUROC Comparison", flush=True)
    print("-" * 40, flush=True)

    test_easy_cos = [all_samples[i]['cos_dist'] for i in test_idxs if all_samples[i]['difficulty'] == 'easy']
    test_ood_cos = [all_samples[i]['cos_dist'] for i in test_idxs if all_samples[i]['difficulty'] == 'ood']
    test_easy_mass = [-all_samples[i]['action_mass'] for i in test_idxs if all_samples[i]['difficulty'] == 'easy']
    test_ood_mass = [-all_samples[i]['action_mass'] for i in test_idxs if all_samples[i]['difficulty'] == 'ood']

    auroc_cos = compute_auroc(test_ood_cos, test_easy_cos)
    auroc_mass = compute_auroc(test_ood_mass, test_easy_mass)

    print(f"  Cosine Distance AUROC: {auroc_cos:.3f}", flush=True)
    print(f"  Action Mass AUROC:     {auroc_mass:.3f}", flush=True)
    print(f"  Δ:                     {auroc_cos - auroc_mass:+.3f}", flush=True)

    # Per-OOD type
    print("\n  Per-OOD-Type:", flush=True)
    for ood_type in [s for s in SCENARIOS if s.startswith('ood_')]:
        ood_cos = [all_samples[i]['cos_dist'] for i in test_idxs if all_samples[i]['scenario'] == ood_type]
        ood_mass = [-all_samples[i]['action_mass'] for i in test_idxs if all_samples[i]['scenario'] == ood_type]

        auroc_c = compute_auroc(ood_cos, test_easy_cos)
        auroc_m = compute_auroc(ood_mass, test_easy_mass)
        print(f"    {ood_type:<15}: cos={auroc_c:.3f}, mass={auroc_m:.3f}, Δ={auroc_c-auroc_m:+.3f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'timestamp': timestamp,
        'pipeline_results': pipeline_results,
        'auroc_cosine': auroc_cos,
        'auroc_mass': auroc_mass,
        'samples': [{k: v for k, v in s.items()} for s in all_samples],
    }

    output_path = os.path.join(RESULTS_DIR, f"cosine_safety_pipeline_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
