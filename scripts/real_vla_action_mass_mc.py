"""
Action Mass Robustness and MC Enhancement on Real OpenVLA-7B.

Tests whether the action mass signal (AUROC=0.949) can be further improved:
1. Action mass with MC Dropout — does averaging over MC passes stabilize the signal?
2. Action mass across different prompts — is it prompt-invariant?
3. Action mass combined with entropy — does a joint signal beat either alone?
4. Per-dimension action mass as individual OOD signals

Experiment 19 in the CalibDrive series.
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

N_MC_PASSES = 10
OPTIMAL_DROPOUT = 0.20

PROMPTS = [
    "In: What action should the robot take to drive forward at {speed} m/s safely?\nOut:",
    "In: You are driving at {speed} m/s. What is the safe driving action?\nOut:",
    "In: Navigate safely at {speed} m/s. What action to take?\nOut:",
    "In: Predict the driving action to maintain safe driving at {speed} m/s.\nOut:",
]

SCENARIOS = {
    'highway': {'n': 15, 'speed': '30', 'difficulty': 'easy'},
    'urban': {'n': 15, 'speed': '15', 'difficulty': 'easy'},
    'night': {'n': 10, 'speed': '25', 'difficulty': 'hard'},
    'rain': {'n': 10, 'speed': '20', 'difficulty': 'hard'},
    'ood_noise': {'n': 15, 'speed': '25', 'difficulty': 'ood'},
    'ood_blank': {'n': 15, 'speed': '25', 'difficulty': 'ood'},
}


def create_scene_image(scenario, idx, size=(256, 256)):
    np.random.seed(idx * 1400 + hash(scenario) % 14000)
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


def single_forward_full_vocab(model, processor, image, prompt):
    """Forward pass returning both action-bin and full-vocab statistics."""
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

    dim_action_masses = []
    dim_entropies = []
    dim_confs = []

    for score in outputs.scores[:7]:
        full_logits = score[0].float()
        full_probs = torch.softmax(full_logits, dim=0).cpu().numpy()

        action_probs = full_probs[action_start:]
        action_mass = float(action_probs.sum())
        dim_action_masses.append(action_mass)

        # Action-only entropy and confidence
        action_norm = action_probs / (action_probs.sum() + 1e-10)
        entropy = float(-(action_norm * np.log(action_norm + 1e-10)).sum())
        conf = float(action_norm.max())
        dim_entropies.append(entropy)
        dim_confs.append(conf)

    return {
        'mean_action_mass': float(np.mean(dim_action_masses)),
        'min_action_mass': float(np.min(dim_action_masses)),
        'dim_action_masses': dim_action_masses,
        'mean_entropy': float(np.mean(dim_entropies)),
        'geo_conf': float(np.exp(np.mean(np.log(np.array(dim_confs) + 1e-10)))),
    }


def compute_auroc(pos_scores, neg_scores):
    n_correct = sum(1 for p in pos_scores for n in neg_scores if p > n)
    n_ties = sum(0.5 for p in pos_scores for n in neg_scores if p == n)
    n_total = len(pos_scores) * len(neg_scores)
    return (n_correct + n_ties) / n_total if n_total > 0 else 0.5


def main():
    print("=" * 70, flush=True)
    print("ACTION MASS ROBUSTNESS & MC ENHANCEMENT ON REAL OpenVLA-7B", flush=True)
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
    # Phase 1: Single pass (no dropout) — baseline action mass
    # Phase 2: MC Dropout (p=0.20, N=10) — MC-averaged action mass
    # Phase 3: Multi-prompt (4 prompts) — prompt-averaged action mass
    total_inferences = total * (1 + N_MC_PASSES + len(PROMPTS))
    print(f"Total samples: {total}, Total inferences: {total_inferences}", flush=True)
    print(flush=True)

    all_results = {}
    sample_idx = 0

    for scenario, config in SCENARIOS.items():
        scenario_results = []

        for i in range(config['n']):
            sample_idx += 1
            image = create_scene_image(scenario, i)
            base_prompt = PROMPTS[0].format(speed=config['speed'])

            t0 = time.time()

            # Phase 1: Single pass, no dropout
            set_dropout_rate(model, 0.0)
            model.eval()
            r_single = single_forward_full_vocab(model, processor, image, base_prompt)

            # Phase 2: MC Dropout
            set_dropout_rate(model, OPTIMAL_DROPOUT)
            enable_mc_dropout(model)
            mc_results = []
            for mc in range(N_MC_PASSES):
                r = single_forward_full_vocab(model, processor, image, base_prompt)
                mc_results.append(r)

            mc_action_masses = [r['mean_action_mass'] for r in mc_results]
            mc_entropies = [r['mean_entropy'] for r in mc_results]
            mc_confs = [r['geo_conf'] for r in mc_results]

            # Phase 3: Multi-prompt (no dropout)
            set_dropout_rate(model, 0.0)
            model.eval()
            prompt_results = []
            for prompt_tmpl in PROMPTS:
                prompt = prompt_tmpl.format(speed=config['speed'])
                r = single_forward_full_vocab(model, processor, image, prompt)
                prompt_results.append(r)

            prompt_action_masses = [r['mean_action_mass'] for r in prompt_results]

            elapsed = time.time() - t0

            sample_result = {
                'scenario': scenario,
                'difficulty': config['difficulty'],
                'idx': i,
                # Single pass
                'single_action_mass': r_single['mean_action_mass'],
                'single_min_action_mass': r_single['min_action_mass'],
                'single_entropy': r_single['mean_entropy'],
                'single_conf': r_single['geo_conf'],
                'single_dim_masses': r_single['dim_action_masses'],
                # MC averaged
                'mc_action_mass_mean': float(np.mean(mc_action_masses)),
                'mc_action_mass_std': float(np.std(mc_action_masses)),
                'mc_entropy_mean': float(np.mean(mc_entropies)),
                'mc_conf_mean': float(np.mean(mc_confs)),
                # Prompt averaged
                'prompt_action_mass_mean': float(np.mean(prompt_action_masses)),
                'prompt_action_mass_std': float(np.std(prompt_action_masses)),
                'prompt_action_masses': prompt_action_masses,
                # Joint signal: action mass * (1 - entropy/max_entropy)
                'joint_signal': float(r_single['mean_action_mass'] * (1 - r_single['mean_entropy'] / np.log(256))),
            }
            scenario_results.append(sample_result)

            if i % 5 == 0 or i == config['n'] - 1:
                print(f"  [{sample_idx}/{total}] {scenario}_{i}: "
                      f"single_am={r_single['mean_action_mass']:.4f}, "
                      f"mc_am={np.mean(mc_action_masses):.4f}±{np.std(mc_action_masses):.4f}, "
                      f"prompt_am={np.mean(prompt_action_masses):.4f}±{np.std(prompt_action_masses):.4f} "
                      f"({elapsed:.1f}s)", flush=True)

        all_results[scenario] = scenario_results

    # ===================================================================
    # Analysis
    # ===================================================================
    print("\n" + "=" * 70, flush=True)
    print("ACTION MASS ROBUSTNESS ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    # 1. Per-scenario comparison
    print("\n1. Action Mass Signals Per Scenario", flush=True)
    print("-" * 90, flush=True)
    print(f"{'Scenario':<12} {'Single':>8} {'MC Mean':>8} {'MC Std':>7} "
          f"{'Prompt Mean':>12} {'Prompt Std':>11} {'Joint':>7}", flush=True)
    print("-" * 90, flush=True)

    for scenario in SCENARIOS:
        samples = all_results[scenario]
        single = [s['single_action_mass'] for s in samples]
        mc_mean = [s['mc_action_mass_mean'] for s in samples]
        mc_std = [s['mc_action_mass_std'] for s in samples]
        pr_mean = [s['prompt_action_mass_mean'] for s in samples]
        pr_std = [s['prompt_action_mass_std'] for s in samples]
        joint = [s['joint_signal'] for s in samples]

        print(f"{scenario:<12} {np.mean(single):>8.4f} {np.mean(mc_mean):>8.4f} "
              f"{np.mean(mc_std):>7.4f} {np.mean(pr_mean):>12.4f} "
              f"{np.mean(pr_std):>11.4f} {np.mean(joint):>7.4f}", flush=True)

    # 2. AUROC comparison
    print("\n2. AUROC Comparison (Easy vs OOD)", flush=True)
    print("-" * 70, flush=True)

    easy_signals = {}
    ood_signals = {}
    signal_names = [
        'single_action_mass', 'single_min_action_mass', 'single_entropy',
        'mc_action_mass_mean', 'mc_action_mass_std', 'mc_entropy_mean',
        'prompt_action_mass_mean', 'prompt_action_mass_std', 'joint_signal',
    ]

    for sig in signal_names:
        easy_signals[sig] = []
        ood_signals[sig] = []

    for scenario in SCENARIOS:
        for s in all_results[scenario]:
            for sig in signal_names:
                if s['difficulty'] == 'easy':
                    easy_signals[sig].append(s[sig])
                elif s['difficulty'] == 'ood':
                    ood_signals[sig].append(s[sig])

    # For action mass: higher = easier (so negate for OOD detection)
    # For entropy: higher = harder
    higher_for_ood = {
        'single_action_mass': False,   # lower mass = OOD
        'single_min_action_mass': False,
        'single_entropy': True,
        'mc_action_mass_mean': False,
        'mc_action_mass_std': True,    # higher std = OOD
        'mc_entropy_mean': True,
        'prompt_action_mass_mean': False,
        'prompt_action_mass_std': True,  # higher disagreement = OOD
        'joint_signal': False,
    }

    display_names = {
        'single_action_mass': 'Neg Single Action Mass',
        'single_min_action_mass': 'Neg Min Action Mass (worst dim)',
        'single_entropy': 'Single Entropy',
        'mc_action_mass_mean': 'Neg MC Action Mass Mean',
        'mc_action_mass_std': 'MC Action Mass Std',
        'mc_entropy_mean': 'MC Entropy Mean',
        'prompt_action_mass_mean': 'Neg Prompt Action Mass Mean',
        'prompt_action_mass_std': 'Prompt Action Mass Std',
        'joint_signal': 'Neg Joint (mass × (1-ent/log256))',
    }

    auroc_results = {}
    for sig in signal_names:
        if higher_for_ood[sig]:
            auroc = compute_auroc(ood_signals[sig], easy_signals[sig])
        else:
            auroc = compute_auroc(easy_signals[sig], ood_signals[sig])
        auroc_results[sig] = auroc
        print(f"  {display_names[sig]:<40}: AUROC = {auroc:.3f}", flush=True)

    # 3. Per-dimension action mass AUROC
    print("\n3. Per-Dimension Action Mass AUROC (Easy vs OOD)", flush=True)
    print("-" * 50, flush=True)

    for d in range(7):
        easy_dim = []
        ood_dim = []
        for scenario in SCENARIOS:
            for s in all_results[scenario]:
                if s['difficulty'] == 'easy':
                    easy_dim.append(s['single_dim_masses'][d])
                elif s['difficulty'] == 'ood':
                    ood_dim.append(s['single_dim_masses'][d])
        auroc = compute_auroc(easy_dim, ood_dim)
        print(f"  Dim {d}: AUROC = {auroc:.3f} "
              f"(easy={np.mean(easy_dim):.4f}, ood={np.mean(ood_dim):.4f})", flush=True)

    # 4. Prompt robustness
    print("\n4. Prompt Robustness of Action Mass", flush=True)
    print("-" * 50, flush=True)

    for p_idx, prompt in enumerate(PROMPTS):
        easy_mass = []
        ood_mass = []
        for scenario in SCENARIOS:
            for s in all_results[scenario]:
                if s['difficulty'] == 'easy':
                    easy_mass.append(s['prompt_action_masses'][p_idx])
                elif s['difficulty'] == 'ood':
                    ood_mass.append(s['prompt_action_masses'][p_idx])
        auroc = compute_auroc(easy_mass, ood_mass)
        print(f"  Prompt {p_idx+1}: AUROC = {auroc:.3f} "
              f"(easy={np.mean(easy_mass):.4f}, ood={np.mean(ood_mass):.4f})", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'timestamp': timestamp,
        'config': {
            'n_mc_passes': N_MC_PASSES,
            'dropout_rate': OPTIMAL_DROPOUT,
            'n_prompts': len(PROMPTS),
        },
        'auroc_results': auroc_results,
        'per_sample': {
            scenario: [
                {k: v for k, v in s.items()}
                for s in samples
            ]
            for scenario, samples in all_results.items()
        },
    }

    output_path = os.path.join(RESULTS_DIR, f"action_mass_mc_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
