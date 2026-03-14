"""
Temperature Sweep on Real OpenVLA-7B.

Tests temperatures T ∈ {0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0} across scenarios.
For each temperature, computes calibration metrics on the softmax distribution.
Also tests per-dimension optimal temperatures.

Experiment 7 in the CalibDrive series.
"""
import os
import json
import time
import datetime
import numpy as np
import torch
from PIL import Image

# =============================================================================
# Configuration
# =============================================================================
TEMPERATURES = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
N_SAMPLES = 100  # 100 samples across scenarios
SCENARIOS = {
    'highway': {'n': 20, 'speed': '30'},
    'urban': {'n': 20, 'speed': '15'},
    'night': {'n': 15, 'speed': '25'},
    'rain': {'n': 15, 'speed': '20'},
    'fog': {'n': 15, 'speed': '20'},
    'ood_noise': {'n': 15, 'speed': '25'},
}

RESULTS_DIR = "/workspace/Vizuara-VLA-Research/experiments"
os.makedirs(RESULTS_DIR, exist_ok=True)


def create_scene_image(scenario, idx, size=(256, 256)):
    """Create synthetic scene image for each scenario."""
    np.random.seed(idx * 100 + hash(scenario) % 1000)

    if scenario == 'highway':
        img = np.zeros((*size, 3), dtype=np.uint8)
        img[:size[0]//2] = [135, 206, 235]  # sky
        img[size[0]//2:] = [80, 80, 80]  # road
        img[size[0]//2 - 5:size[0]//2 + 5] = [34, 139, 34]  # horizon
        # lane markings
        for x in range(0, size[1], 40):
            img[size[0]*3//4 - 2:size[0]*3//4 + 2, x:x+20] = [255, 255, 255]
    elif scenario == 'urban':
        img = np.zeros((*size, 3), dtype=np.uint8)
        img[:size[0]//3] = [135, 206, 235]
        img[size[0]//3:size[0]//2] = [139, 119, 101]  # buildings
        img[size[0]//2:] = [80, 80, 80]
        # crosswalk
        for x in range(0, size[1], 20):
            img[size[0]*3//4:size[0]*3//4+10, x:x+10] = [255, 255, 255]
    elif scenario == 'night':
        img = np.full((*size, 3), 15, dtype=np.uint8)
        img[size[0]//2:] = [30, 30, 35]
        # headlight cones
        for i in range(20):
            y = size[0]//2 + np.random.randint(0, size[0]//2)
            x = np.random.randint(0, size[1])
            r = np.random.randint(3, 8)
            img[max(0,y-r):y+r, max(0,x-r):x+r] = [200, 200, 150]
    elif scenario == 'rain':
        img = np.zeros((*size, 3), dtype=np.uint8)
        img[:size[0]//2] = [100, 100, 110]
        img[size[0]//2:] = [60, 60, 65]
        # rain streaks
        for _ in range(200):
            y = np.random.randint(0, size[0]-10)
            x = np.random.randint(0, size[1])
            img[y:y+8, x:min(x+1, size[1]-1)] = [180, 180, 200]
    elif scenario == 'fog':
        img = np.full((*size, 3), 180, dtype=np.uint8)  # foggy white-gray
        img[:size[0]//2] += np.random.randint(0, 20, (*img[:size[0]//2].shape,), dtype=np.uint8).clip(0, 255)
        img[size[0]//2:] = [150, 150, 155]
        # dim road markings
        for x in range(0, size[1], 50):
            img[size[0]*3//4-1:size[0]*3//4+1, x:x+15] = [170, 170, 170]
    elif scenario == 'ood_noise':
        img = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
    else:
        img = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)

    # Add variation
    noise = np.random.randint(-10, 10, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(img)


def extract_logits_at_temperatures(model, processor, image, prompt, temperatures):
    """Extract logits and compute metrics at multiple temperatures in a single forward pass."""
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

    results_by_temp = {}
    for T in temperatures:
        dim_metrics = []
        for dim_idx, score in enumerate(outputs.scores[:7]):
            logits = score[0, action_start:].float()
            scaled_logits = logits / T
            probs = torch.softmax(scaled_logits, dim=0)

            max_prob = probs.max().item()
            entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
            perplexity = np.exp(entropy)
            top5_mass = probs.topk(5).values.sum().item()

            # Number of bins with p > 0.01
            sig_bins = (probs > 0.01).sum().item()

            dim_metrics.append({
                'max_prob': max_prob,
                'entropy': entropy,
                'perplexity': perplexity,
                'top5_mass': top5_mass,
                'sig_bins': sig_bins,
            })

        # Geometric mean confidence
        geo_mean_conf = np.exp(np.mean([np.log(d['max_prob'] + 1e-10) for d in dim_metrics]))
        mean_entropy = np.mean([d['entropy'] for d in dim_metrics])
        mean_perplexity = np.mean([d['perplexity'] for d in dim_metrics])

        results_by_temp[T] = {
            'geo_mean_conf': geo_mean_conf,
            'mean_entropy': mean_entropy,
            'mean_perplexity': mean_perplexity,
            'per_dim': dim_metrics,
        }

    return results_by_temp


def main():
    print("=" * 70)
    print("TEMPERATURE SWEEP ON REAL OpenVLA-7B")
    print("=" * 70)
    print(f"Temperatures: {TEMPERATURES}")
    print(f"Samples: {N_SAMPLES}")
    print()

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
    print(f"Model loaded. Device: {model.device}")
    print()

    # Run experiment
    all_results = {}
    sample_idx = 0
    total_samples = sum(s['n'] for s in SCENARIOS.values())

    for scenario, config in SCENARIOS.items():
        print(f"\n--- {scenario} (N={config['n']}) ---")
        scenario_results = {T: [] for T in TEMPERATURES}

        for i in range(config['n']):
            sample_idx += 1
            image = create_scene_image(scenario, i)
            prompt = f"In: What action should the robot take to drive forward at {config['speed']} m/s safely?\nOut:"

            t0 = time.time()
            temp_results = extract_logits_at_temperatures(
                model, processor, image, prompt, TEMPERATURES
            )
            elapsed = time.time() - t0

            for T in TEMPERATURES:
                scenario_results[T].append(temp_results[T])

            if i % 5 == 0 or i == config['n'] - 1:
                r = temp_results[1.0]
                print(f"  [{sample_idx}/{total_samples}] {scenario}_{i}: "
                      f"T=1.0 conf={r['geo_mean_conf']:.3f}, "
                      f"ent={r['mean_entropy']:.3f} ({elapsed:.1f}s)")

        all_results[scenario] = scenario_results

    # =======================================================================
    # Analysis
    # =======================================================================
    print("\n" + "=" * 70)
    print("TEMPERATURE SWEEP ANALYSIS")
    print("=" * 70)

    # 1. Per-temperature aggregate stats
    print("\n1. Aggregate Stats by Temperature")
    print("-" * 60)
    print(f"{'T':>6} {'Conf':>8} {'±':>6} {'Entropy':>8} {'±':>6} {'Perplexity':>10} {'SigBins':>8}")
    print("-" * 60)

    temp_summaries = {}
    for T in TEMPERATURES:
        all_confs = []
        all_entropies = []
        all_perplexities = []
        for scenario in SCENARIOS:
            for r in all_results[scenario][T]:
                all_confs.append(r['geo_mean_conf'])
                all_entropies.append(r['mean_entropy'])
                all_perplexities.append(r['mean_perplexity'])

        temp_summaries[T] = {
            'conf_mean': np.mean(all_confs),
            'conf_std': np.std(all_confs),
            'entropy_mean': np.mean(all_entropies),
            'entropy_std': np.std(all_entropies),
            'perplexity_mean': np.mean(all_perplexities),
        }
        print(f"{T:>6.2f} {np.mean(all_confs):>8.3f} {np.std(all_confs):>6.3f} "
              f"{np.mean(all_entropies):>8.3f} {np.std(all_entropies):>6.3f} "
              f"{np.mean(all_perplexities):>10.1f}")

    # 2. Per-scenario confidence at each temperature
    print("\n2. Per-Scenario Confidence by Temperature")
    print("-" * 80)
    header = f"{'Scenario':>14}"
    for T in TEMPERATURES:
        header += f" {'T='+str(T):>8}"
    print(header)
    print("-" * 80)

    scenario_temp_confs = {}
    for scenario in SCENARIOS:
        row = f"{scenario:>14}"
        scenario_temp_confs[scenario] = {}
        for T in TEMPERATURES:
            confs = [r['geo_mean_conf'] for r in all_results[scenario][T]]
            mean_conf = np.mean(confs)
            scenario_temp_confs[scenario][T] = mean_conf
            row += f" {mean_conf:>8.3f}"
        print(row)

    # 3. Discrimination analysis: does temperature improve scenario separation?
    print("\n3. Discrimination Analysis (Highway - OOD gap)")
    print("-" * 50)
    for T in TEMPERATURES:
        hwy_conf = scenario_temp_confs['highway'][T]
        ood_conf = scenario_temp_confs['ood_noise'][T]
        gap = hwy_conf - ood_conf
        print(f"  T={T:.2f}: Highway={hwy_conf:.3f}, OOD={ood_conf:.3f}, Gap={gap:+.4f}")

    # 4. Per-dimension analysis at T=1.0 vs best T
    print("\n4. Per-Dimension Confidence at T=1.0")
    print("-" * 50)
    for dim_idx in range(7):
        dim_confs = []
        for scenario in SCENARIOS:
            for r in all_results[scenario][1.0]:
                dim_confs.append(r['per_dim'][dim_idx]['max_prob'])
        print(f"  Dim {dim_idx}: {np.mean(dim_confs):.3f} ± {np.std(dim_confs):.3f}")

    # 5. Find optimal per-dimension temperature
    print("\n5. Optimal Per-Dimension Temperature (minimizing confidence spread)")
    print("-" * 60)
    for dim_idx in range(7):
        best_T = None
        best_spread = float('inf')
        for T in TEMPERATURES:
            dim_confs_by_scenario = {}
            for scenario in SCENARIOS:
                confs = [r['per_dim'][dim_idx]['max_prob'] for r in all_results[scenario][T]]
                dim_confs_by_scenario[scenario] = np.mean(confs)
            # Spread = range of confidences across scenarios
            vals = list(dim_confs_by_scenario.values())
            spread = max(vals) - min(vals)
            if spread < best_spread:
                best_spread = spread
                best_T = T
        print(f"  Dim {dim_idx}: Best T={best_T}, spread={best_spread:.4f}")

    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'timestamp': timestamp,
        'config': {
            'temperatures': TEMPERATURES,
            'n_samples': total_samples,
            'scenarios': {k: v['n'] for k, v in SCENARIOS.items()},
        },
        'temp_summaries': {str(T): s for T, s in temp_summaries.items()},
        'scenario_temp_confs': {s: {str(T): v for T, v in tc.items()}
                                for s, tc in scenario_temp_confs.items()},
    }

    output_path = os.path.join(RESULTS_DIR, f"temperature_sweep_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
