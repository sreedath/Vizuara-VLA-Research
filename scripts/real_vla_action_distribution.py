"""
Action Distribution Analysis on Real OpenVLA-7B.

Analyzes the 256-bin action token distributions in detail:
1. Distribution shape analysis (kurtosis, skewness, modality)
2. Bin utilization across scenarios
3. Probability mass concentration (Gini coefficient)
4. Inter-scenario KL divergence between action distributions
5. Calibration vs bin usage correlation

Experiment 12 in the CalibDrive series.
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
    'highway': {'n': 20, 'speed': '30', 'difficulty': 'easy'},
    'urban': {'n': 20, 'speed': '15', 'difficulty': 'easy'},
    'night': {'n': 15, 'speed': '25', 'difficulty': 'hard'},
    'rain': {'n': 15, 'speed': '20', 'difficulty': 'hard'},
    'fog': {'n': 15, 'speed': '20', 'difficulty': 'hard'},
    'construction': {'n': 15, 'speed': '10', 'difficulty': 'hard'},
    'ood_noise': {'n': 20, 'speed': '25', 'difficulty': 'ood'},
    'ood_blank': {'n': 20, 'speed': '25', 'difficulty': 'ood'},
}


def create_scene_image(scenario, idx, size=(256, 256)):
    np.random.seed(idx * 600 + hash(scenario) % 6000)
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


def gini_coefficient(probs):
    """Compute Gini coefficient of probability distribution. 0=uniform, 1=concentrated."""
    sorted_p = np.sort(probs)
    n = len(sorted_p)
    cumsum = np.cumsum(sorted_p)
    return (2 * np.sum((np.arange(1, n+1) * sorted_p)) / (n * np.sum(sorted_p))) - (n + 1) / n


def count_modes(probs, threshold=0.01):
    """Count number of modes (local maxima above threshold)."""
    modes = 0
    for i in range(1, len(probs) - 1):
        if probs[i] > probs[i-1] and probs[i] > probs[i+1] and probs[i] > threshold:
            modes += 1
    return max(modes, 1)  # at least 1


def kl_divergence(p, q, epsilon=1e-10):
    """Compute KL(P || Q)."""
    p = np.array(p) + epsilon
    q = np.array(q) + epsilon
    p = p / p.sum()
    q = q / q.sum()
    return np.sum(p * np.log(p / q))


def main():
    print("=" * 70, flush=True)
    print("ACTION DISTRIBUTION ANALYSIS ON REAL OpenVLA-7B", flush=True)
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

    # Pre-generate images
    images = {}
    prompts = {}
    for scenario, config in SCENARIOS.items():
        for i in range(config['n']):
            key = f"{scenario}_{i}"
            images[key] = create_scene_image(scenario, i)
            prompts[key] = f"In: What action should the robot take to drive forward at {config['speed']} m/s safely?\nOut:"

    total = sum(s['n'] for s in SCENARIOS.values())
    print(f"Total samples: {total}", flush=True)
    print(flush=True)

    all_results = {}
    sample_idx = 0

    # Accumulate per-scenario average distributions
    scenario_avg_dists = {s: [np.zeros(256) for _ in range(7)] for s in SCENARIOS}
    scenario_counts = {s: 0 for s in SCENARIOS}

    for scenario, config in SCENARIOS.items():
        scenario_results = []

        for i in range(config['n']):
            sample_idx += 1
            key = f"{scenario}_{i}"
            image = images[key]
            prompt = prompts[key]

            t0 = time.time()
            inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=7,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            elapsed = time.time() - t0

            vocab_size = outputs.scores[0].shape[-1]
            action_start = vocab_size - 256

            dim_results = []
            for d, score in enumerate(outputs.scores[:7]):
                logits = score[0, action_start:].float()
                probs = torch.softmax(logits, dim=0).cpu().numpy()

                # Distribution statistics
                conf = float(probs.max())
                entropy = float(-(probs * np.log(probs + 1e-10)).sum())
                top_bin = int(probs.argmax())

                # Top-k mass
                sorted_p = np.sort(probs)[::-1]
                top5_mass = float(sorted_p[:5].sum())
                top10_mass = float(sorted_p[:10].sum())
                top20_mass = float(sorted_p[:20].sum())

                # Effective support (bins with p > 0.001)
                eff_support = int((probs > 0.001).sum())
                eff_support_01 = int((probs > 0.01).sum())

                # Perplexity
                perplexity = float(np.exp(entropy))

                # Gini
                gini = float(gini_coefficient(probs))

                # Kurtosis and skewness of prob distribution
                bin_indices = np.arange(256)
                mean_bin = np.sum(bin_indices * probs)
                var_bin = np.sum((bin_indices - mean_bin)**2 * probs)
                std_bin = np.sqrt(var_bin) if var_bin > 0 else 1e-10
                skew = float(np.sum(((bin_indices - mean_bin) / std_bin)**3 * probs)) if std_bin > 0 else 0
                kurt = float(np.sum(((bin_indices - mean_bin) / std_bin)**4 * probs)) - 3  # excess kurtosis

                # Number of modes
                n_modes = count_modes(probs)

                # Accumulate for scenario average
                scenario_avg_dists[scenario][d] += probs

                dim_results.append({
                    'dim': d,
                    'confidence': conf,
                    'entropy': entropy,
                    'perplexity': perplexity,
                    'top_bin': top_bin,
                    'top5_mass': top5_mass,
                    'top10_mass': top10_mass,
                    'top20_mass': top20_mass,
                    'eff_support': eff_support,
                    'eff_support_01': eff_support_01,
                    'gini': gini,
                    'skewness': skew,
                    'kurtosis': kurt,
                    'n_modes': n_modes,
                    'mean_bin': float(mean_bin),
                    'std_bin': float(std_bin),
                })

            scenario_counts[scenario] += 1

            sample_result = {
                'scenario': scenario,
                'difficulty': config['difficulty'],
                'idx': i,
                'dims': dim_results,
                'geo_conf': float(np.exp(np.mean(np.log([d['confidence'] for d in dim_results])))),
                'mean_entropy': float(np.mean([d['entropy'] for d in dim_results])),
                'mean_perplexity': float(np.mean([d['perplexity'] for d in dim_results])),
                'mean_gini': float(np.mean([d['gini'] for d in dim_results])),
                'mean_eff_support': float(np.mean([d['eff_support'] for d in dim_results])),
            }
            scenario_results.append(sample_result)

            if i % 5 == 0 or i == config['n'] - 1:
                print(f"  [{sample_idx}/{total}] {key}: "
                      f"conf={sample_result['geo_conf']:.3f}, "
                      f"perp={sample_result['mean_perplexity']:.1f}, "
                      f"gini={sample_result['mean_gini']:.3f}, "
                      f"supp={sample_result['mean_eff_support']:.1f} "
                      f"({elapsed:.1f}s)", flush=True)

        all_results[scenario] = scenario_results

    # Normalize scenario average distributions
    for s in SCENARIOS:
        for d in range(7):
            scenario_avg_dists[s][d] /= scenario_counts[s]

    # ===================================================================
    # Analysis
    # ===================================================================
    print("\n" + "=" * 70, flush=True)
    print("ACTION DISTRIBUTION ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    # 1. Per-dimension statistics across all scenarios
    print("\n1. Per-Dimension Distribution Statistics (All Scenarios)", flush=True)
    print("-" * 100, flush=True)
    print(f"{'Dim':>4} {'Conf':>7} {'Entropy':>8} {'Perp':>7} {'Gini':>6} "
          f"{'Top5%':>7} {'Supp>0.1%':>10} {'Supp>1%':>8} {'Modes':>6} {'Kurt':>7}", flush=True)
    print("-" * 100, flush=True)

    for d in range(7):
        confs = []
        ents = []
        perps = []
        ginis = []
        top5s = []
        supps = []
        supps01 = []
        modes_list = []
        kurts = []

        for scenario in SCENARIOS:
            for s in all_results[scenario]:
                dim_data = s['dims'][d]
                confs.append(dim_data['confidence'])
                ents.append(dim_data['entropy'])
                perps.append(dim_data['perplexity'])
                ginis.append(dim_data['gini'])
                top5s.append(dim_data['top5_mass'])
                supps.append(dim_data['eff_support'])
                supps01.append(dim_data['eff_support_01'])
                modes_list.append(dim_data['n_modes'])
                kurts.append(dim_data['kurtosis'])

        print(f"{d:>4} {np.mean(confs):>7.3f} {np.mean(ents):>8.3f} {np.mean(perps):>7.1f} "
              f"{np.mean(ginis):>6.3f} {np.mean(top5s):>7.3f} {np.mean(supps):>10.1f} "
              f"{np.mean(supps01):>8.1f} {np.mean(modes_list):>6.1f} {np.mean(kurts):>7.1f}", flush=True)

    # 2. Per-scenario statistics
    print("\n2. Per-Scenario Distribution Statistics", flush=True)
    print("-" * 90, flush=True)
    print(f"{'Scenario':<15} {'Conf':>7} {'Perp':>7} {'Gini':>6} {'Supp':>6} "
          f"{'Modes':>6} {'Kurt':>7} {'MeanBin':>8}", flush=True)
    print("-" * 90, flush=True)

    for scenario in SCENARIOS:
        samples = all_results[scenario]
        confs = [s['geo_conf'] for s in samples]
        perps = [s['mean_perplexity'] for s in samples]
        ginis = [s['mean_gini'] for s in samples]
        supps = [s['mean_eff_support'] for s in samples]
        modes = [np.mean([d['n_modes'] for d in s['dims']]) for s in samples]
        kurts = [np.mean([d['kurtosis'] for d in s['dims']]) for s in samples]
        mean_bins = [np.mean([d['mean_bin'] for d in s['dims']]) for s in samples]

        print(f"{scenario:<15} {np.mean(confs):>7.3f} {np.mean(perps):>7.1f} "
              f"{np.mean(ginis):>6.3f} {np.mean(supps):>6.1f} {np.mean(modes):>6.1f} "
              f"{np.mean(kurts):>7.1f} {np.mean(mean_bins):>8.1f}", flush=True)

    # 3. Inter-scenario KL divergence (per dim, average)
    print("\n3. Inter-Scenario KL Divergence (avg over 7 dims)", flush=True)
    print("-" * 70, flush=True)
    scenarios_list = list(SCENARIOS.keys())
    header = f"{'':>15}" + "".join(f"{s:>12}" for s in scenarios_list)
    print(header, flush=True)
    print("-" * 70, flush=True)

    kl_matrix = {}
    for s1 in scenarios_list:
        kl_matrix[s1] = {}
        row = f"{s1:>15}"
        for s2 in scenarios_list:
            kl_vals = []
            for d in range(7):
                kl = kl_divergence(scenario_avg_dists[s1][d], scenario_avg_dists[s2][d])
                kl_vals.append(kl)
            mean_kl = np.mean(kl_vals)
            kl_matrix[s1][s2] = mean_kl
            row += f"{mean_kl:>12.4f}"
        print(row, flush=True)

    # 4. Top bin consistency across scenarios
    print("\n4. Top Bin Consistency Across Scenarios (per dimension)", flush=True)
    print("-" * 70, flush=True)

    for d in range(7):
        bin_counts = {}
        for scenario in SCENARIOS:
            for s in all_results[scenario]:
                top_bin = s['dims'][d]['top_bin']
                if top_bin not in bin_counts:
                    bin_counts[top_bin] = {'total': 0}
                    for sc in SCENARIOS:
                        bin_counts[top_bin][sc] = 0
                bin_counts[top_bin]['total'] += 1
                bin_counts[top_bin][scenario] += 1

        top_bins = sorted(bin_counts.items(), key=lambda x: -x[1]['total'])[:3]
        print(f"  Dim {d}: Top bins = {[(b, c['total']) for b, c in top_bins]}", flush=True)

        # Check if same bin dominates across all scenarios
        most_common = top_bins[0][0]
        all_same = all(
            all_results[sc][0]['dims'][d]['top_bin'] == most_common
            for sc in SCENARIOS
            if len(all_results[sc]) > 0
        )
        print(f"         Same bin across all scenarios: {all_same}", flush=True)

    # 5. Bin diversity: unique bins used per scenario
    print("\n5. Unique Bins Used Per Scenario (across all dims)", flush=True)
    print("-" * 50, flush=True)
    for scenario in SCENARIOS:
        all_bins = set()
        for s in all_results[scenario]:
            for d in s['dims']:
                all_bins.add((d['dim'], d['top_bin']))
        print(f"  {scenario:<15}: {len(all_bins)} unique (dim, bin) pairs", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'timestamp': timestamp,
        'per_sample': {
            scenario: [
                {k: v for k, v in s.items()}
                for s in samples
            ]
            for scenario, samples in all_results.items()
        },
        'kl_matrix': {s1: {s2: float(v) for s2, v in row.items()} for s1, row in kl_matrix.items()},
    }

    output_path = os.path.join(RESULTS_DIR, f"action_distribution_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
