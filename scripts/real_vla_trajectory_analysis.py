"""
Trajectory-Level Calibration Analysis on Real OpenVLA-7B.

Simulates multi-step prediction by:
1. Taking an initial scene image
2. Running K sequential predictions (each conditioned on the same image but
   with modified speed prompts reflecting cumulative effect)
3. Measuring how confidence and entropy evolve over trajectory steps
4. Testing whether uncertainty compounds or stabilizes over multi-step rollouts

This is important because driving requires multi-step planning, not single-shot.

Experiment 15 in the CalibDrive series.
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

N_STEPS = 10  # Trajectory length
N_MC_PASSES = 5  # MC passes per step for uncertainty

SCENARIOS = {
    'highway': {'n': 10, 'base_speed': 30, 'difficulty': 'easy'},
    'urban': {'n': 10, 'base_speed': 15, 'difficulty': 'easy'},
    'night': {'n': 10, 'base_speed': 25, 'difficulty': 'hard'},
    'ood_noise': {'n': 10, 'base_speed': 25, 'difficulty': 'ood'},
}

OPTIMAL_DROPOUT = 0.20


def create_scene_image(scenario, idx, size=(256, 256)):
    np.random.seed(idx * 900 + hash(scenario) % 9000)
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
    elif scenario == 'ood_noise':
        img = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
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
    dim_tokens = []
    for score in outputs.scores[:7]:
        logits = score[0, action_start:].float()
        probs = torch.softmax(logits, dim=0)
        dim_confs.append(probs.max().item())
        dim_entropies.append(-(probs * torch.log(probs + 1e-10)).sum().item())
        dim_tokens.append(probs.argmax().item())
    return {
        'geo_conf': float(np.exp(np.mean(np.log(np.array(dim_confs) + 1e-10)))),
        'mean_entropy': float(np.mean(dim_entropies)),
        'dim_tokens': dim_tokens,
        'dim_confs': [float(c) for c in dim_confs],
    }


def main():
    print("=" * 70, flush=True)
    print("TRAJECTORY-LEVEL CALIBRATION ON REAL OpenVLA-7B", flush=True)
    print("=" * 70, flush=True)
    print(f"Steps per trajectory: {N_STEPS}", flush=True)
    print(f"MC passes per step: {N_MC_PASSES}", flush=True)
    print(f"Dropout rate: {OPTIMAL_DROPOUT}", flush=True)
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
    print("Model loaded with MC Dropout enabled.", flush=True)

    total = sum(s['n'] for s in SCENARIOS.values())
    total_inferences = total * N_STEPS * N_MC_PASSES
    print(f"Total samples: {total}, Total inferences: {total_inferences}", flush=True)
    print(flush=True)

    all_results = {}
    sample_idx = 0

    for scenario, config in SCENARIOS.items():
        scenario_results = []

        for i in range(config['n']):
            sample_idx += 1
            base_img = create_scene_image(scenario, i)

            t0 = time.time()
            trajectory = []

            for step in range(N_STEPS):
                # Vary the prompt slightly per step to simulate trajectory progression
                speed = config['base_speed'] + np.random.uniform(-2, 2)
                prompt = f"In: What action should the robot take to drive forward at {speed:.0f} m/s safely?\nOut:"

                # MC passes for this step
                mc_results = []
                for mc in range(N_MC_PASSES):
                    r = single_forward(model, processor, base_img, prompt)
                    mc_results.append(r)

                confs = [r['geo_conf'] for r in mc_results]
                entropies = [r['mean_entropy'] for r in mc_results]
                tokens_per_pass = [r['dim_tokens'] for r in mc_results]
                tokens_arr = np.array(tokens_per_pass)
                token_agree = np.mean([len(set(tokens_arr[:, d])) == 1 for d in range(7)])

                step_result = {
                    'step': step,
                    'speed': float(speed),
                    'conf_mean': float(np.mean(confs)),
                    'conf_std': float(np.std(confs)),
                    'entropy_mean': float(np.mean(entropies)),
                    'entropy_std': float(np.std(entropies)),
                    'token_agreement': float(token_agree),
                }
                trajectory.append(step_result)

            elapsed = time.time() - t0

            # Compute trajectory-level metrics
            step_confs = [s['conf_mean'] for s in trajectory]
            step_ents = [s['entropy_mean'] for s in trajectory]
            step_mc_stds = [s['conf_std'] for s in trajectory]

            # Trajectory consistency: how much do predictions vary across steps?
            # (ideally they should be consistent for same image)
            conf_drift = np.std(step_confs)
            ent_drift = np.std(step_ents)

            # Cumulative uncertainty: product of per-step confidences
            cum_conf = np.prod(step_confs)

            sample_result = {
                'scenario': scenario,
                'difficulty': config['difficulty'],
                'idx': i,
                'trajectory': trajectory,
                'conf_drift': float(conf_drift),
                'ent_drift': float(ent_drift),
                'cum_conf': float(cum_conf),
                'mean_conf': float(np.mean(step_confs)),
                'mean_mc_std': float(np.mean(step_mc_stds)),
            }
            scenario_results.append(sample_result)

            print(f"  [{sample_idx}/{total}] {scenario}_{i}: "
                  f"mean_conf={np.mean(step_confs):.3f}, "
                  f"conf_drift={conf_drift:.4f}, "
                  f"cum_conf={cum_conf:.6f} ({elapsed:.1f}s)", flush=True)

        all_results[scenario] = scenario_results

    # ===================================================================
    # Analysis
    # ===================================================================
    print("\n" + "=" * 70, flush=True)
    print("TRAJECTORY ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    # 1. Per-scenario trajectory summary
    print("\n1. Per-Scenario Trajectory Summary", flush=True)
    print("-" * 80, flush=True)
    print(f"{'Scenario':<12} {'MeanConf':>9} {'ConfDrift':>10} {'CumConf':>12} "
          f"{'MeanMCStd':>10} {'MeanTokAg':>10}", flush=True)
    print("-" * 80, flush=True)

    for scenario in SCENARIOS:
        samples = all_results[scenario]
        mean_confs = [s['mean_conf'] for s in samples]
        drifts = [s['conf_drift'] for s in samples]
        cum_confs = [s['cum_conf'] for s in samples]
        mc_stds = [s['mean_mc_std'] for s in samples]
        tok_agrees = [np.mean([t['token_agreement'] for t in s['trajectory']]) for s in samples]

        print(f"{scenario:<12} {np.mean(mean_confs):>9.3f} {np.mean(drifts):>10.4f} "
              f"{np.mean(cum_confs):>12.8f} {np.mean(mc_stds):>10.4f} "
              f"{np.mean(tok_agrees):>10.3f}", flush=True)

    # 2. Step-by-step confidence evolution
    print("\n2. Confidence Evolution Over Trajectory Steps", flush=True)
    print("-" * 80, flush=True)
    header = f"{'Step':>5}"
    for sc in SCENARIOS:
        header += f"{sc:>12}"
    print(header, flush=True)
    print("-" * 80, flush=True)

    for step in range(N_STEPS):
        row = f"{step:>5}"
        for scenario in SCENARIOS:
            step_confs = [s['trajectory'][step]['conf_mean'] for s in all_results[scenario]]
            row += f"{np.mean(step_confs):>12.3f}"
        print(row, flush=True)

    # 3. AUROC using cumulative confidence and confidence drift
    print("\n3. AUROC Using Trajectory-Level Signals", flush=True)
    print("-" * 60, flush=True)

    easy_cum = []
    ood_cum = []
    easy_drift = []
    ood_drift = []
    hard_cum = []
    hard_drift = []

    for scenario in SCENARIOS:
        for s in all_results[scenario]:
            if s['difficulty'] == 'easy':
                easy_cum.append(-s['cum_conf'])  # negative so higher=more uncertain
                easy_drift.append(s['conf_drift'])
            elif s['difficulty'] == 'ood':
                ood_cum.append(-s['cum_conf'])
                ood_drift.append(s['conf_drift'])
            elif s['difficulty'] == 'hard':
                hard_cum.append(-s['cum_conf'])
                hard_drift.append(s['conf_drift'])

    for name, easy_vals, ood_vals in [
        ('Neg Cum Conf', easy_cum, ood_cum),
        ('Conf Drift', easy_drift, ood_drift),
    ]:
        n_correct = sum(1 for e in easy_vals for o in ood_vals if o > e)
        n_ties = sum(0.5 for e in easy_vals for o in ood_vals if o == e)
        n_total = len(easy_vals) * len(ood_vals)
        auroc = (n_correct + n_ties) / n_total if n_total > 0 else 0.5
        print(f"  {name:<20}: AUROC(easy vs ood) = {auroc:.3f}", flush=True)

    if hard_cum:
        for name, easy_vals, hard_vals in [
            ('Neg Cum Conf', easy_cum, hard_cum),
            ('Conf Drift', easy_drift, hard_drift),
        ]:
            n_correct = sum(1 for e in easy_vals for h in hard_vals if h > e)
            n_ties = sum(0.5 for e in easy_vals for h in hard_vals if h == e)
            n_total = len(easy_vals) * len(hard_vals)
            auroc = (n_correct + n_ties) / n_total if n_total > 0 else 0.5
            print(f"  {name:<20}: AUROC(easy vs hard) = {auroc:.3f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'timestamp': timestamp,
        'config': {
            'n_steps': N_STEPS,
            'n_mc_passes': N_MC_PASSES,
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

    output_path = os.path.join(RESULTS_DIR, f"trajectory_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
