"""
Leaked Token Interpretability on Real OpenVLA-7B.

When probability leaks from action bins to non-action vocabulary tokens,
WHAT tokens receive the leaked probability? This reveals:
1. Does the model try to produce text descriptions instead of actions?
2. Are specific non-action tokens consistently preferred?
3. Do leaked token identities differ between easy and OOD scenarios?
4. Can leaked token identity serve as an additional uncertainty signal?

Experiment 23 in the CalibDrive series.
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
    np.random.seed(idx * 1800 + hash(scenario) % 18000)
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
    print("LEAKED TOKEN INTERPRETABILITY ON REAL OpenVLA-7B", flush=True)
    print("=" * 70, flush=True)

    from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer
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
    # Get tokenizer for decoding
    tokenizer = processor.tokenizer
    model.eval()
    print("Model loaded.", flush=True)

    vocab_size = model.config.text_config.vocab_size if hasattr(model.config, 'text_config') else 32064
    print(f"Vocab size: {vocab_size}", flush=True)

    total = sum(s['n'] for s in SCENARIOS.values())
    print(f"Total samples: {total}", flush=True)
    print(flush=True)

    prompt_tmpl = "In: What action should the robot take to drive forward at {speed} m/s safely?\nOut:"

    # Aggregate: which non-action tokens appear most?
    global_non_action_counts = {}  # token_id -> total probability mass
    scenario_non_action = {s: {} for s in SCENARIOS}

    all_samples = []
    sample_idx = 0

    for scenario, config in SCENARIOS.items():
        for i in range(config['n']):
            sample_idx += 1
            image = create_scene_image(scenario, i)
            prompt = prompt_tmpl.format(speed=config['speed'])

            inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=7,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            actual_vocab_size = outputs.scores[0].shape[-1]
            action_start = actual_vocab_size - 256

            sample_leaked = []
            total_leaked_mass = 0.0

            for d, score in enumerate(outputs.scores[:7]):
                full_logits = score[0].float()
                full_probs = torch.softmax(full_logits, dim=0).cpu().numpy()

                # Non-action tokens
                non_action_probs = full_probs[:action_start]
                action_mass = float(full_probs[action_start:].sum())
                leaked_mass = float(non_action_probs.sum())
                total_leaked_mass += leaked_mass

                # Top-10 non-action tokens for this dimension
                top_indices = np.argsort(non_action_probs)[::-1][:10]
                top_tokens = []
                for idx in top_indices:
                    prob = float(non_action_probs[idx])
                    if prob < 1e-6:
                        break
                    try:
                        text = tokenizer.decode([int(idx)])
                    except Exception:
                        text = f"<id:{idx}>"
                    top_tokens.append({
                        'id': int(idx),
                        'text': text,
                        'prob': prob,
                    })

                    # Aggregate
                    global_non_action_counts[int(idx)] = global_non_action_counts.get(int(idx), 0.0) + prob
                    scenario_non_action[scenario][int(idx)] = scenario_non_action[scenario].get(int(idx), 0.0) + prob

                sample_leaked.append({
                    'dim': d,
                    'action_mass': action_mass,
                    'leaked_mass': leaked_mass,
                    'top_tokens': top_tokens[:5],
                })

            sample = {
                'scenario': scenario,
                'difficulty': config['difficulty'],
                'idx': i,
                'mean_action_mass': 1.0 - total_leaked_mass / 7.0,
                'total_leaked_mass': total_leaked_mass / 7.0,
                'dims': sample_leaked,
            }
            all_samples.append(sample)

            if i % 5 == 0 or i == config['n'] - 1:
                top_token = sample_leaked[0]['top_tokens'][0] if sample_leaked[0]['top_tokens'] else {'text': '?', 'prob': 0}
                print(f"  [{sample_idx}/{total}] {scenario}_{i}: "
                      f"leaked={total_leaked_mass/7:.4f}, "
                      f"top_leaked='{top_token['text']}' ({top_token['prob']:.4f})",
                      flush=True)

    # ===================================================================
    # Analysis
    # ===================================================================
    print("\n" + "=" * 70, flush=True)
    print("LEAKED TOKEN ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    # 1. Global top leaked tokens
    print("\n1. Top 20 Non-Action Tokens by Total Probability Mass", flush=True)
    print("-" * 60, flush=True)

    sorted_global = sorted(global_non_action_counts.items(), key=lambda x: -x[1])
    for rank, (token_id, total_mass) in enumerate(sorted_global[:20]):
        try:
            text = tokenizer.decode([token_id])
        except Exception:
            text = f"<id:{token_id}>"
        avg_per_sample = total_mass / (total * 7)
        print(f"  {rank+1:>3}. Token {token_id:>6} '{text:<15}': "
              f"total_mass={total_mass:.4f}, avg/dim={avg_per_sample:.6f}", flush=True)

    # 2. Per-scenario top leaked tokens
    print("\n2. Top 5 Leaked Tokens Per Scenario", flush=True)
    print("-" * 80, flush=True)

    for scenario in SCENARIOS:
        sorted_scenario = sorted(scenario_non_action[scenario].items(), key=lambda x: -x[1])
        tokens_str = []
        for token_id, mass in sorted_scenario[:5]:
            try:
                text = tokenizer.decode([token_id])
            except Exception:
                text = f"<id:{token_id}>"
            tokens_str.append(f"'{text}'({mass:.3f})")
        print(f"  {scenario:<12}: {', '.join(tokens_str)}", flush=True)

    # 3. Are leaked tokens the same across scenarios?
    print("\n3. Leaked Token Overlap Between Scenarios", flush=True)
    print("-" * 60, flush=True)

    scenario_top_sets = {}
    for scenario in SCENARIOS:
        sorted_s = sorted(scenario_non_action[scenario].items(), key=lambda x: -x[1])
        scenario_top_sets[scenario] = set(t[0] for t in sorted_s[:20])

    for s1 in SCENARIOS:
        for s2 in SCENARIOS:
            if s1 >= s2:
                continue
            overlap = len(scenario_top_sets[s1] & scenario_top_sets[s2])
            jaccard = overlap / len(scenario_top_sets[s1] | scenario_top_sets[s2])
            print(f"  {s1:<12} ∩ {s2:<12}: {overlap}/20 overlap, Jaccard={jaccard:.3f}", flush=True)

    # 4. Leaked mass per dimension
    print("\n4. Mean Leaked Mass Per Dimension", flush=True)
    print("-" * 40, flush=True)

    for d in range(7):
        easy_leaked = [s['dims'][d]['leaked_mass'] for s in all_samples if s['difficulty'] == 'easy']
        ood_leaked = [s['dims'][d]['leaked_mass'] for s in all_samples if s['difficulty'] == 'ood']
        print(f"  Dim {d}: easy={np.mean(easy_leaked):.4f}, "
              f"ood={np.mean(ood_leaked):.4f}, "
              f"ratio={np.mean(ood_leaked)/np.mean(easy_leaked):.2f}x", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'timestamp': timestamp,
        'top_global_tokens': [
            {'id': tid, 'mass': mass, 'text': tokenizer.decode([tid]) if tid < actual_vocab_size else '?'}
            for tid, mass in sorted_global[:50]
        ],
        'per_scenario_tokens': {
            scenario: [
                {'id': tid, 'mass': mass}
                for tid, mass in sorted(counts.items(), key=lambda x: -x[1])[:20]
            ]
            for scenario, counts in scenario_non_action.items()
        },
        'samples': [
            {k: v for k, v in s.items() if k != 'dims'}
            for s in all_samples
        ],
    }

    output_path = os.path.join(RESULTS_DIR, f"leaked_tokens_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
