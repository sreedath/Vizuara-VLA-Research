"""
Dropout Rate Sensitivity on Real OpenVLA-7B.

Tests injected dropout rates p ∈ {0.01, 0.05, 0.1, 0.15, 0.2, 0.3}
with N=10 MC passes per sample, across 60 samples (6 scenarios × 10).

Measures how dropout rate affects:
1. Mean confidence shift
2. MC variance magnitude
3. Scenario discrimination (AUROC)
4. Prediction stability (token agreement across passes)

Experiment 10 in the CalibDrive series.
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

DROPOUT_RATES = [0.0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3]
N_MC_PASSES = 10
SCENARIOS = {
    'highway': {'n': 10, 'speed': '30', 'difficulty': 'easy'},
    'urban': {'n': 10, 'speed': '15', 'difficulty': 'easy'},
    'night': {'n': 10, 'speed': '25', 'difficulty': 'hard'},
    'rain': {'n': 10, 'speed': '20', 'difficulty': 'hard'},
    'ood_noise': {'n': 10, 'speed': '25', 'difficulty': 'ood'},
    'ood_blank': {'n': 10, 'speed': '25', 'difficulty': 'ood'},
}


def create_scene_image(scenario, idx, size=(256, 256)):
    np.random.seed(idx * 400 + hash(scenario) % 4000)
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

    noise = np.random.randint(-10, 10, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def set_dropout_rate(model, rate):
    """Set all Dropout layers to a specific rate."""
    count = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = rate
            count += 1
    return count


def enable_mc_dropout(model):
    """Enable dropout at inference by setting dropout modules to train mode."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


def disable_mc_dropout(model):
    """Disable dropout by setting dropout modules to eval mode."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.eval()


def single_forward(model, processor, image, prompt):
    """Run a single forward pass and extract logit metrics."""
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
    for score in outputs.scores[:7]:
        logits = score[0, action_start:].float()
        probs = torch.softmax(logits, dim=0)
        dim_confs.append(probs.max().item())
        dim_entropies.append(-(probs * torch.log(probs + 1e-10)).sum().item())
        dim_tokens.append(probs.argmax().item())

    geo_conf = np.exp(np.mean(np.log(np.array(dim_confs) + 1e-10)))
    mean_ent = np.mean(dim_entropies)

    return {
        'geo_conf': float(geo_conf),
        'mean_entropy': float(mean_ent),
        'dim_confs': [float(c) for c in dim_confs],
        'dim_tokens': dim_tokens,
    }


def main():
    print("=" * 70)
    print("DROPOUT RATE SENSITIVITY ON REAL OpenVLA-7B")
    print("=" * 70)
    print(f"Dropout rates: {DROPOUT_RATES}")
    print(f"MC passes per rate: {N_MC_PASSES}")
    print()

    from transformers import AutoModelForVision2Seq, AutoProcessor
    print("Loading OpenVLA-7B...")
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

    n_dropout = set_dropout_rate(model, 0.0)
    print(f"Model loaded. {n_dropout} Dropout layers found.")
    print()

    # Pre-generate all images
    images = {}
    prompts = {}
    for scenario, config in SCENARIOS.items():
        for i in range(config['n']):
            key = f"{scenario}_{i}"
            images[key] = create_scene_image(scenario, i)
            prompts[key] = f"In: What action should the robot take to drive forward at {config['speed']} m/s safely?\nOut:"

    total_samples = sum(s['n'] for s in SCENARIOS.values())

    # Run for each dropout rate
    all_results = {}

    for p_rate in DROPOUT_RATES:
        print(f"\n{'='*50}")
        print(f"  DROPOUT RATE p = {p_rate}")
        print(f"{'='*50}")

        set_dropout_rate(model, p_rate)

        if p_rate > 0:
            enable_mc_dropout(model)
        else:
            disable_mc_dropout(model)

        rate_results = {}
        sample_idx = 0

        for scenario, config in SCENARIOS.items():
            scenario_samples = []

            for i in range(config['n']):
                sample_idx += 1
                key = f"{scenario}_{i}"
                image = images[key]
                prompt = prompts[key]

                # Run N_MC_PASSES forward passes
                t0 = time.time()
                pass_results = []
                for mc_pass in range(N_MC_PASSES):
                    r = single_forward(model, processor, image, prompt)
                    pass_results.append(r)
                elapsed = time.time() - t0

                # Compute MC statistics
                confs = [r['geo_conf'] for r in pass_results]
                entropies = [r['mean_entropy'] for r in pass_results]
                tokens_per_pass = [r['dim_tokens'] for r in pass_results]

                # Token agreement: fraction of dims where all passes agree
                tokens_arr = np.array(tokens_per_pass)  # (N_MC, 7)
                token_agree = np.mean([
                    len(set(tokens_arr[:, d])) == 1 for d in range(7)
                ])

                sample_result = {
                    'scenario': scenario,
                    'difficulty': config['difficulty'],
                    'idx': i,
                    'conf_mean': float(np.mean(confs)),
                    'conf_std': float(np.std(confs)),
                    'entropy_mean': float(np.mean(entropies)),
                    'entropy_std': float(np.std(entropies)),
                    'token_agreement': float(token_agree),
                    'n_unique_tokens_per_dim': [int(len(set(tokens_arr[:, d]))) for d in range(7)],
                }
                scenario_samples.append(sample_result)

                if i % 5 == 0 or i == config['n'] - 1:
                    print(f"  p={p_rate}: [{sample_idx}/{total_samples}] {key}: "
                          f"conf={np.mean(confs):.3f}±{np.std(confs):.4f}, "
                          f"tok_agree={token_agree:.2f} ({elapsed:.1f}s)")

            rate_results[scenario] = scenario_samples

        all_results[p_rate] = rate_results

    # ===================================================================
    # Analysis
    # ===================================================================
    print("\n" + "=" * 70)
    print("DROPOUT SENSITIVITY ANALYSIS")
    print("=" * 70)

    # 1. MC Variance vs Dropout Rate
    print("\n1. MC Variance by Dropout Rate")
    print("-" * 60)
    print(f"{'p':>6} {'Conf Mean':>10} {'Conf Std':>10} {'Ent Std':>10} {'Tok Agree':>10}")
    print("-" * 60)

    rate_summaries = {}
    for p_rate in DROPOUT_RATES:
        all_conf_stds = []
        all_conf_means = []
        all_ent_stds = []
        all_tok_agrees = []
        for scenario in SCENARIOS:
            for s in all_results[p_rate][scenario]:
                all_conf_stds.append(s['conf_std'])
                all_conf_means.append(s['conf_mean'])
                all_ent_stds.append(s['entropy_std'])
                all_tok_agrees.append(s['token_agreement'])

        rate_summaries[p_rate] = {
            'conf_mean': np.mean(all_conf_means),
            'conf_std': np.mean(all_conf_stds),
            'ent_std': np.mean(all_ent_stds),
            'tok_agree': np.mean(all_tok_agrees),
        }
        print(f"{p_rate:>6.2f} {np.mean(all_conf_means):>10.3f} "
              f"{np.mean(all_conf_stds):>10.4f} {np.mean(all_ent_stds):>10.4f} "
              f"{np.mean(all_tok_agrees):>10.2f}")

    # 2. Scenario Discrimination (AUROC) by Dropout Rate
    print("\n2. AUROC(easy vs ood) by Dropout Rate")
    print("-" * 40)
    for p_rate in DROPOUT_RATES:
        easy_entropies = []
        ood_entropies = []
        for scenario in SCENARIOS:
            for s in all_results[p_rate][scenario]:
                if s['difficulty'] == 'easy':
                    easy_entropies.append(s['entropy_mean'])
                elif s['difficulty'] == 'ood':
                    ood_entropies.append(s['entropy_mean'])

        n_correct = sum(1 for e in easy_entropies for o in ood_entropies if o > e)
        n_ties = sum(0.5 for e in easy_entropies for o in ood_entropies if o == e)
        n_total = len(easy_entropies) * len(ood_entropies)
        auroc = (n_correct + n_ties) / n_total if n_total > 0 else 0.5
        print(f"  p={p_rate:.2f}: AUROC={auroc:.3f}")

    # 3. MC Uncertainty as discrimination signal
    print("\n3. AUROC(easy vs ood) using MC Conf Std as signal")
    print("-" * 40)
    for p_rate in DROPOUT_RATES:
        if p_rate == 0:
            print(f"  p=0.00: AUROC=0.500 (no variance)")
            continue
        easy_stds = []
        ood_stds = []
        for scenario in SCENARIOS:
            for s in all_results[p_rate][scenario]:
                if s['difficulty'] == 'easy':
                    easy_stds.append(s['conf_std'])
                elif s['difficulty'] == 'ood':
                    ood_stds.append(s['conf_std'])

        n_correct = sum(1 for e in easy_stds for o in ood_stds if o > e)
        n_ties = sum(0.5 for e in easy_stds for o in ood_stds if o == e)
        n_total = len(easy_stds) * len(ood_stds)
        auroc = (n_correct + n_ties) / n_total if n_total > 0 else 0.5
        print(f"  p={p_rate:.2f}: AUROC={auroc:.3f}")

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'timestamp': timestamp,
        'dropout_rates': DROPOUT_RATES,
        'n_mc_passes': N_MC_PASSES,
        'rate_summaries': {str(k): v for k, v in rate_summaries.items()},
        'per_sample': {
            str(p_rate): {
                scenario: [
                    {k: v for k, v in s.items()}
                    for s in samples
                ]
                for scenario, samples in rate_results.items()
            }
            for p_rate, rate_results in all_results.items()
        },
    }

    output_path = os.path.join(RESULTS_DIR, f"dropout_sweep_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
