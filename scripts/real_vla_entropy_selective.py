"""
Entropy-Based Selective Prediction on Real OpenVLA-7B.

Since our large-scale results showed entropy provides significant (p=0.037)
scenario discrimination while raw confidence does not (p=0.321), this experiment
tests whether entropy-based abstention can improve safety.

We compute the "selective prediction" curve: as we increase the entropy threshold
for abstention (rejecting high-entropy predictions), what happens to the
remaining predictions' scenario composition?

Experiment 8 in the CalibDrive series.
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

N_TOTAL = 120  # 120 samples
SCENARIOS = {
    'highway': {'n': 25, 'speed': '30', 'difficulty': 'easy'},
    'urban': {'n': 25, 'speed': '15', 'difficulty': 'easy'},
    'night': {'n': 15, 'speed': '25', 'difficulty': 'hard'},
    'rain': {'n': 15, 'speed': '20', 'difficulty': 'hard'},
    'fog': {'n': 15, 'speed': '20', 'difficulty': 'hard'},
    'ood_noise': {'n': 25, 'speed': '25', 'difficulty': 'ood'},
}


def create_scene_image(scenario, idx, size=(256, 256)):
    """Create synthetic scene image."""
    np.random.seed(idx * 200 + hash(scenario) % 2000)

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
            y = np.random.randint(0, size[0]-10)
            x = np.random.randint(0, size[1])
            img[y:y+8, x:min(x+1,size[1]-1)] = [180, 180, 200]
    elif scenario == 'fog':
        img = np.full((*size, 3), 180, dtype=np.uint8)
        img[size[0]//2:] = [150, 150, 155]
    elif scenario == 'ood_noise':
        img = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
    else:
        img = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)

    noise = np.random.randint(-10, 10, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def extract_calibration_signals(model, processor, image, prompt):
    """Extract all calibration signals from a single forward pass."""
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

    dim_data = []
    for dim_idx, score in enumerate(outputs.scores[:7]):
        logits = score[0, action_start:].float()
        probs = torch.softmax(logits, dim=0)

        max_prob = probs.max().item()
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        top5_mass = probs.topk(5).values.sum().item()
        margin = (probs.topk(2).values[0] - probs.topk(2).values[1]).item()

        dim_data.append({
            'max_prob': max_prob,
            'entropy': entropy,
            'top5_mass': top5_mass,
            'margin': margin,
        })

    geo_mean_conf = np.exp(np.mean([np.log(d['max_prob'] + 1e-10) for d in dim_data]))
    mean_entropy = np.mean([d['entropy'] for d in dim_data])
    max_entropy = max(d['entropy'] for d in dim_data)
    mean_margin = np.mean([d['margin'] for d in dim_data])
    mean_top5 = np.mean([d['top5_mass'] for d in dim_data])

    return {
        'geo_mean_conf': geo_mean_conf,
        'mean_entropy': mean_entropy,
        'max_entropy': max_entropy,
        'mean_margin': mean_margin,
        'mean_top5': mean_top5,
        'per_dim': dim_data,
    }


def main():
    print("=" * 70)
    print("ENTROPY-BASED SELECTIVE PREDICTION ON REAL OpenVLA-7B")
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
    print(f"Model loaded.")

    # Run inference
    all_samples = []
    sample_idx = 0
    total = sum(s['n'] for s in SCENARIOS.values())

    for scenario, config in SCENARIOS.items():
        print(f"\n--- {scenario} (N={config['n']}) ---")
        for i in range(config['n']):
            sample_idx += 1
            image = create_scene_image(scenario, i)
            prompt = f"In: What action should the robot take to drive forward at {config['speed']} m/s safely?\nOut:"

            t0 = time.time()
            signals = extract_calibration_signals(model, processor, image, prompt)
            elapsed = time.time() - t0

            sample = {
                'scenario': scenario,
                'difficulty': config['difficulty'],
                'idx': i,
                **signals,
            }
            # Remove per_dim for top-level to keep it clean
            sample['per_dim'] = signals['per_dim']
            all_samples.append(sample)

            if i % 5 == 0 or i == config['n'] - 1:
                print(f"  [{sample_idx}/{total}] {scenario}_{i}: "
                      f"conf={signals['geo_mean_conf']:.3f}, "
                      f"ent={signals['mean_entropy']:.3f}, "
                      f"margin={signals['mean_margin']:.3f} ({elapsed:.1f}s)")

    # ===================================================================
    # Selective Prediction Analysis
    # ===================================================================
    print("\n" + "=" * 70)
    print("SELECTIVE PREDICTION ANALYSIS")
    print("=" * 70)

    # Sort by different uncertainty signals
    uncertainty_signals = {
        'entropy': lambda s: s['mean_entropy'],         # higher = more uncertain
        'neg_confidence': lambda s: -s['geo_mean_conf'], # lower conf = more uncertain
        'max_entropy': lambda s: s['max_entropy'],       # highest-dim entropy
        'neg_margin': lambda s: -s['mean_margin'],       # lower margin = more uncertain
        'neg_top5': lambda s: -s['mean_top5'],           # lower top-5 mass = more uncertain
    }

    coverage_levels = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]

    print("\n1. OOD Rejection Rate at Each Coverage Level")
    print("-" * 70)
    header = f"{'Coverage':>10}"
    for sig_name in uncertainty_signals:
        header += f" {sig_name:>14}"
    print(header)
    print("-" * 70)

    selective_results = {}
    for sig_name, sig_fn in uncertainty_signals.items():
        # Sort samples by uncertainty (most certain first for acceptance)
        sorted_samples = sorted(all_samples, key=sig_fn)
        selective_results[sig_name] = {}

        for cov in coverage_levels:
            n_keep = int(len(sorted_samples) * cov)
            kept = sorted_samples[:n_keep]

            # Count OOD samples in kept set
            n_ood_kept = sum(1 for s in kept if s['difficulty'] == 'ood')
            n_ood_total = sum(1 for s in all_samples if s['difficulty'] == 'ood')
            ood_rejection_rate = 1.0 - (n_ood_kept / n_ood_total) if n_ood_total > 0 else 0

            # Count hard samples in kept set
            n_hard_kept = sum(1 for s in kept if s['difficulty'] == 'hard')
            n_hard_total = sum(1 for s in all_samples if s['difficulty'] == 'hard')
            hard_rejection_rate = 1.0 - (n_hard_kept / n_hard_total) if n_hard_total > 0 else 0

            # Fraction easy in kept
            n_easy_kept = sum(1 for s in kept if s['difficulty'] == 'easy')
            easy_fraction = n_easy_kept / n_keep if n_keep > 0 else 0

            selective_results[sig_name][cov] = {
                'n_keep': n_keep,
                'ood_rejection_rate': ood_rejection_rate,
                'hard_rejection_rate': hard_rejection_rate,
                'easy_fraction': easy_fraction,
            }

    for cov in coverage_levels:
        row = f"{cov:>10.0%}"
        for sig_name in uncertainty_signals:
            row += f" {selective_results[sig_name][cov]['ood_rejection_rate']:>14.1%}"
        print(row)

    print("\n2. Hard Scenario Rejection Rate at Each Coverage Level")
    print("-" * 70)
    header = f"{'Coverage':>10}"
    for sig_name in uncertainty_signals:
        header += f" {sig_name:>14}"
    print(header)
    print("-" * 70)
    for cov in coverage_levels:
        row = f"{cov:>10.0%}"
        for sig_name in uncertainty_signals:
            row += f" {selective_results[sig_name][cov]['hard_rejection_rate']:>14.1%}"
        print(row)

    print("\n3. Easy Fraction at Each Coverage Level (higher = better filtering)")
    print("-" * 70)
    header = f"{'Coverage':>10}"
    for sig_name in uncertainty_signals:
        header += f" {sig_name:>14}"
    print(header)
    print("-" * 70)
    for cov in coverage_levels:
        row = f"{cov:>10.0%}"
        for sig_name in uncertainty_signals:
            row += f" {selective_results[sig_name][cov]['easy_fraction']:>14.1%}"
        print(row)

    # 4. Summary: which signal best separates easy from ood?
    print("\n4. AUROC-like Analysis: Signal Discrimination")
    print("-" * 50)
    from itertools import combinations

    for sig_name, sig_fn in uncertainty_signals.items():
        # Compute signal for easy vs ood
        easy_vals = [sig_fn(s) for s in all_samples if s['difficulty'] == 'easy']
        ood_vals = [sig_fn(s) for s in all_samples if s['difficulty'] == 'ood']

        # Simple AUROC: fraction of (easy, ood) pairs where ood has higher uncertainty
        n_correct = 0
        n_total = 0
        for e in easy_vals:
            for o in ood_vals:
                n_total += 1
                if o > e:
                    n_correct += 1
                elif o == e:
                    n_correct += 0.5

        auroc = n_correct / n_total if n_total > 0 else 0.5
        print(f"  {sig_name:>16}: AUROC(easy vs ood) = {auroc:.3f}")

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'timestamp': timestamp,
        'n_samples': len(all_samples),
        'scenario_counts': {k: v['n'] for k, v in SCENARIOS.items()},
        'per_sample': [{k: v for k, v in s.items() if k != 'per_dim'} for s in all_samples],
        'selective_results': {
            sig_name: {str(cov): vals for cov, vals in sig_results.items()}
            for sig_name, sig_results in selective_results.items()
        },
    }

    output_path = os.path.join(RESULTS_DIR, f"entropy_selective_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
