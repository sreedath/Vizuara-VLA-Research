"""
Uncertainty Estimate Reliability on Real OpenVLA-7B.

Tests the reliability (consistency) of uncertainty estimates:
1. Run MC Dropout (p=0.20, N=20) TWICE on the same samples
2. Measure correlation between the two uncertainty estimates
3. If uncertainty is reliable, both runs should produce similar rankings
4. Compute AUROC stability across repeated estimation

This answers: "Can we trust a single UQ run, or do we need multiple?"

Experiment 16 in the CalibDrive series.
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

N_MC_PASSES = 20
N_REPEATS = 3  # Run MC estimation 3 times to measure reliability
OPTIMAL_DROPOUT = 0.20

SCENARIOS = {
    'highway': {'n': 15, 'speed': '30', 'difficulty': 'easy'},
    'urban': {'n': 15, 'speed': '15', 'difficulty': 'easy'},
    'night': {'n': 10, 'speed': '25', 'difficulty': 'hard'},
    'rain': {'n': 10, 'speed': '20', 'difficulty': 'hard'},
    'ood_noise': {'n': 15, 'speed': '25', 'difficulty': 'ood'},
    'ood_blank': {'n': 15, 'speed': '25', 'difficulty': 'ood'},
}


def create_scene_image(scenario, idx, size=(256, 256)):
    np.random.seed(idx * 1000 + hash(scenario) % 10000)
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


def set_dropout_rate(model, rate):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = rate


def enable_mc_dropout(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


def single_forward(model, processor, image, prompt):
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
    for score in outputs.scores[:7]:
        logits = score[0, action_start:].float()
        probs = torch.softmax(logits, dim=0)
        dim_confs.append(probs.max().item())
        dim_entropies.append(-(probs * torch.log(probs + 1e-10)).sum().item())
    return {
        'geo_conf': float(np.exp(np.mean(np.log(np.array(dim_confs) + 1e-10)))),
        'mean_entropy': float(np.mean(dim_entropies)),
    }


def compute_auroc(pos_scores, neg_scores):
    n_correct = sum(1 for p in pos_scores for n in neg_scores if p > n)
    n_ties = sum(0.5 for p in pos_scores for n in neg_scores if p == n)
    n_total = len(pos_scores) * len(neg_scores)
    return (n_correct + n_ties) / n_total if n_total > 0 else 0.5


def main():
    print("=" * 70, flush=True)
    print("UNCERTAINTY RELIABILITY ON REAL OpenVLA-7B", flush=True)
    print("=" * 70, flush=True)
    print(f"MC passes per repeat: {N_MC_PASSES}", flush=True)
    print(f"Number of repeats: {N_REPEATS}", flush=True)
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
    set_dropout_rate(model, OPTIMAL_DROPOUT)
    enable_mc_dropout(model)
    print("Model loaded.", flush=True)

    total = sum(s['n'] for s in SCENARIOS.values())
    total_inferences = total * N_MC_PASSES * N_REPEATS
    print(f"Total samples: {total}, Total inferences: {total_inferences}", flush=True)
    print(flush=True)

    all_results = {}
    sample_idx = 0

    for scenario, config in SCENARIOS.items():
        scenario_results = []

        for i in range(config['n']):
            sample_idx += 1
            image = create_scene_image(scenario, i)
            prompt = f"In: What action should the robot take to drive forward at {config['speed']} m/s safely?\nOut:"

            t0 = time.time()
            repeats = []

            for rep in range(N_REPEATS):
                mc_results = []
                for mc in range(N_MC_PASSES):
                    r = single_forward(model, processor, image, prompt)
                    mc_results.append(r)

                confs = [r['geo_conf'] for r in mc_results]
                entropies = [r['mean_entropy'] for r in mc_results]

                repeats.append({
                    'conf_mean': float(np.mean(confs)),
                    'conf_std': float(np.std(confs)),
                    'entropy_mean': float(np.mean(entropies)),
                    'entropy_std': float(np.std(entropies)),
                })

            elapsed = time.time() - t0

            # Compute cross-repeat statistics
            conf_means = [r['conf_mean'] for r in repeats]
            entropy_means = [r['entropy_mean'] for r in repeats]
            conf_stds = [r['conf_std'] for r in repeats]

            sample_result = {
                'scenario': scenario,
                'difficulty': config['difficulty'],
                'idx': i,
                'repeats': repeats,
                'conf_mean_mean': float(np.mean(conf_means)),
                'conf_mean_std': float(np.std(conf_means)),
                'entropy_mean_mean': float(np.mean(entropy_means)),
                'entropy_mean_std': float(np.std(entropy_means)),
                'conf_std_mean': float(np.mean(conf_stds)),
                'conf_std_std': float(np.std(conf_stds)),
            }
            scenario_results.append(sample_result)

            if i % 5 == 0 or i == config['n'] - 1:
                print(f"  [{sample_idx}/{total}] {scenario}_{i}: "
                      f"ent={np.mean(entropy_means):.3f}±{np.std(entropy_means):.4f}, "
                      f"conf={np.mean(conf_means):.3f}±{np.std(conf_means):.4f} "
                      f"({elapsed:.1f}s)", flush=True)

        all_results[scenario] = scenario_results

    # ===================================================================
    # Analysis
    # ===================================================================
    print("\n" + "=" * 70, flush=True)
    print("UNCERTAINTY RELIABILITY ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    # 1. Cross-repeat consistency
    print("\n1. Cross-Repeat Consistency (3 independent MC runs)", flush=True)
    print("-" * 80, flush=True)
    print(f"{'Scenario':<12} {'EntMean':>8} {'EntStd':>8} {'ConfMean':>9} {'ConfStd':>9} "
          f"{'MCStdMean':>10} {'MCStdStd':>10}", flush=True)
    print("-" * 80, flush=True)

    for scenario in SCENARIOS:
        samples = all_results[scenario]
        ent_means = [s['entropy_mean_mean'] for s in samples]
        ent_stds = [s['entropy_mean_std'] for s in samples]
        conf_means = [s['conf_mean_mean'] for s in samples]
        conf_stds = [s['conf_mean_std'] for s in samples]
        mc_std_means = [s['conf_std_mean'] for s in samples]
        mc_std_stds = [s['conf_std_std'] for s in samples]

        print(f"{scenario:<12} {np.mean(ent_means):>8.3f} {np.mean(ent_stds):>8.4f} "
              f"{np.mean(conf_means):>9.3f} {np.mean(conf_stds):>9.4f} "
              f"{np.mean(mc_std_means):>10.4f} {np.mean(mc_std_stds):>10.4f}", flush=True)

    # 2. Rank correlation between repeats
    print("\n2. Rank Correlation Between Independent UQ Runs", flush=True)
    print("-" * 60, flush=True)

    from scipy.stats import spearmanr, pearsonr

    for sig_name, rep_key in [('Entropy', 'entropy_mean'), ('Confidence', 'conf_mean')]:
        all_samples_list = []
        for scenario in SCENARIOS:
            all_samples_list.extend(all_results[scenario])

        for rep_i in range(N_REPEATS):
            for rep_j in range(rep_i + 1, N_REPEATS):
                vals_i = [s['repeats'][rep_i][rep_key] for s in all_samples_list]
                vals_j = [s['repeats'][rep_j][rep_key] for s in all_samples_list]
                rho, p = spearmanr(vals_i, vals_j)
                r, p_r = pearsonr(vals_i, vals_j)
                print(f"  {sig_name} Run{rep_i+1} vs Run{rep_j+1}: "
                      f"ρ={rho:.3f} (p={p:.4f}), r={r:.3f} (p={p_r:.4f})", flush=True)

    # 3. AUROC stability
    print("\n3. AUROC Stability Across Runs", flush=True)
    print("-" * 60, flush=True)

    for rep in range(N_REPEATS):
        easy_ent = []
        ood_ent = []
        for scenario in SCENARIOS:
            for s in all_results[scenario]:
                val = s['repeats'][rep]['entropy_mean']
                if s['difficulty'] == 'easy':
                    easy_ent.append(val)
                elif s['difficulty'] == 'ood':
                    ood_ent.append(val)

        auroc = compute_auroc(ood_ent, easy_ent)
        print(f"  Run {rep+1}: AUROC(entropy, easy vs ood) = {auroc:.3f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'timestamp': timestamp,
        'config': {
            'n_mc_passes': N_MC_PASSES,
            'n_repeats': N_REPEATS,
            'dropout_rate': OPTIMAL_DROPOUT,
        },
        'per_sample': {
            scenario: [
                {k: v for k, v in s.items()}
                for s in samples
            ]
            for scenario, samples in all_results.items()
        },
    }

    output_path = os.path.join(RESULTS_DIR, f"uncertainty_reliability_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
