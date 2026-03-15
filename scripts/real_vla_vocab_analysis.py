"""
Full Vocabulary Analysis on Real OpenVLA-7B.

Examines what happens when we look at the FULL 32k vocabulary
during action token generation, not just the 256 action bins:
1. What fraction of probability mass goes to action bins vs non-action tokens?
2. Do non-action tokens "compete" more in uncertain scenarios?
3. Can the action-vs-non-action mass ratio serve as an uncertainty signal?

Experiment 18 in the CalibDrive series.
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
    'highway': {'n': 15, 'speed': '30', 'difficulty': 'easy'},
    'urban': {'n': 15, 'speed': '15', 'difficulty': 'easy'},
    'night': {'n': 10, 'speed': '25', 'difficulty': 'hard'},
    'rain': {'n': 10, 'speed': '20', 'difficulty': 'hard'},
    'ood_noise': {'n': 15, 'speed': '25', 'difficulty': 'ood'},
    'ood_blank': {'n': 15, 'speed': '25', 'difficulty': 'ood'},
}


def create_scene_image(scenario, idx, size=(256, 256)):
    np.random.seed(idx * 1300 + hash(scenario) % 13000)
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


def main():
    print("=" * 70, flush=True)
    print("FULL VOCABULARY ANALYSIS ON REAL OpenVLA-7B", flush=True)
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

    all_results = {}
    sample_idx = 0

    for scenario, config in SCENARIOS.items():
        scenario_results = []

        for i in range(config['n']):
            sample_idx += 1
            image = create_scene_image(scenario, i)
            prompt = f"In: What action should the robot take to drive forward at {config['speed']} m/s safely?\nOut:"

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

            dim_results = []
            for d, score in enumerate(outputs.scores[:7]):
                full_logits = score[0].float()

                # Full vocabulary softmax
                full_probs = torch.softmax(full_logits, dim=0).cpu().numpy()

                # Action bin probs
                action_probs = full_probs[action_start:]
                non_action_probs = full_probs[:action_start]

                action_mass = float(action_probs.sum())
                non_action_mass = float(non_action_probs.sum())

                # Top non-action tokens
                non_action_sorted_idx = np.argsort(non_action_probs)[::-1]
                top_non_action = [(int(idx), float(non_action_probs[idx]))
                                  for idx in non_action_sorted_idx[:5]]

                # Action bin entropy (normalized to 256 bins)
                action_norm = action_probs / (action_probs.sum() + 1e-10)
                action_entropy = float(-(action_norm * np.log(action_norm + 1e-10)).sum())
                action_conf = float(action_norm.max())

                # Full vocab entropy
                full_entropy = float(-(full_probs * np.log(full_probs + 1e-10)).sum())

                dim_results.append({
                    'dim': d,
                    'action_mass': action_mass,
                    'non_action_mass': non_action_mass,
                    'action_entropy': action_entropy,
                    'action_conf': action_conf,
                    'full_entropy': full_entropy,
                    'top_non_action_mass': float(sum(p for _, p in top_non_action)),
                })

            # Aggregate
            mean_action_mass = np.mean([d['action_mass'] for d in dim_results])
            mean_action_ent = np.mean([d['action_entropy'] for d in dim_results])
            mean_full_ent = np.mean([d['full_entropy'] for d in dim_results])
            geo_action_conf = np.exp(np.mean(np.log(
                [d['action_conf'] for d in dim_results]
            )))

            sample_result = {
                'scenario': scenario,
                'difficulty': config['difficulty'],
                'idx': i,
                'mean_action_mass': float(mean_action_mass),
                'mean_action_entropy': float(mean_action_ent),
                'mean_full_entropy': float(mean_full_ent),
                'geo_action_conf': float(geo_action_conf),
                'dims': dim_results,
            }
            scenario_results.append(sample_result)

            if i % 5 == 0 or i == config['n'] - 1:
                print(f"  [{sample_idx}/{total}] {scenario}_{i}: "
                      f"act_mass={mean_action_mass:.4f}, "
                      f"act_ent={mean_action_ent:.3f}, "
                      f"full_ent={mean_full_ent:.3f}", flush=True)

        all_results[scenario] = scenario_results

    # ===================================================================
    # Analysis
    # ===================================================================
    print("\n" + "=" * 70, flush=True)
    print("VOCABULARY ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    # 1. Per-scenario action mass
    print("\n1. Action vs Non-Action Probability Mass", flush=True)
    print("-" * 80, flush=True)
    print(f"{'Scenario':<12} {'ActMass':>8} {'ActConf':>8} {'ActEnt':>7} {'FullEnt':>8} {'Ratio':>7}", flush=True)
    print("-" * 80, flush=True)

    for scenario in SCENARIOS:
        samples = all_results[scenario]
        act_masses = [s['mean_action_mass'] for s in samples]
        act_confs = [s['geo_action_conf'] for s in samples]
        act_ents = [s['mean_action_entropy'] for s in samples]
        full_ents = [s['mean_full_entropy'] for s in samples]

        print(f"{scenario:<12} {np.mean(act_masses):>8.4f} {np.mean(act_confs):>8.3f} "
              f"{np.mean(act_ents):>7.3f} {np.mean(full_ents):>8.3f} "
              f"{np.mean(full_ents)/np.mean(act_ents):>7.2f}", flush=True)

    # 2. Per-dimension action mass
    print("\n2. Action Mass Per Dimension (averaged over all scenarios)", flush=True)
    print("-" * 50, flush=True)

    for d in range(7):
        masses = []
        for scenario in SCENARIOS:
            for s in all_results[scenario]:
                masses.append(s['dims'][d]['action_mass'])
        print(f"  Dim {d}: action mass = {np.mean(masses):.4f} ± {np.std(masses):.4f}", flush=True)

    # 3. AUROC using action mass as signal
    print("\n3. AUROC Using Action Mass / Full Entropy as Signal", flush=True)
    print("-" * 60, flush=True)

    easy_act_mass = []
    ood_act_mass = []
    easy_full_ent = []
    ood_full_ent = []
    easy_act_ent = []
    ood_act_ent = []

    for scenario in SCENARIOS:
        for s in all_results[scenario]:
            if s['difficulty'] == 'easy':
                easy_act_mass.append(s['mean_action_mass'])
                easy_full_ent.append(s['mean_full_entropy'])
                easy_act_ent.append(s['mean_action_entropy'])
            elif s['difficulty'] == 'ood':
                ood_act_mass.append(s['mean_action_mass'])
                ood_full_ent.append(s['mean_full_entropy'])
                ood_act_ent.append(s['mean_action_entropy'])

    for name, easy_vals, ood_vals, higher_ood in [
        ('Neg Action Mass', easy_act_mass, ood_act_mass, False),
        ('Full Entropy', easy_full_ent, ood_full_ent, True),
        ('Action Entropy', easy_act_ent, ood_act_ent, True),
    ]:
        if higher_ood:
            n_correct = sum(1 for e in easy_vals for o in ood_vals if o > e)
        else:
            n_correct = sum(1 for e in easy_vals for o in ood_vals if e > o)
        n_ties = sum(0.5 for e in easy_vals for o in ood_vals if o == e)
        n_total = len(easy_vals) * len(ood_vals)
        auroc = (n_correct + n_ties) / n_total if n_total > 0 else 0.5
        print(f"  {name:<20}: AUROC(easy vs ood) = {auroc:.3f}", flush=True)

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
    }

    output_path = os.path.join(RESULTS_DIR, f"vocab_analysis_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
