"""
Cosine Distance Prompt Robustness on Real OpenVLA-7B.

Tests whether cosine distance OOD detection is robust across different prompts.
Action mass was highly prompt-sensitive (AUROC 0.371-0.884 in Exp 19).
Does cosine distance maintain its AUROC across prompts?

Also tests:
1. Cross-prompt centroid transfer: calibrate with prompt A, test with prompt B
2. Prompt-averaged centroid: average centroids across multiple prompts
3. Action mass prompt sensitivity as baseline comparison

Experiment 31 in the CalibDrive series.
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

PROMPTS = {
    'p1': "In: What action should the robot take to drive forward at {speed} m/s safely?\nOut:",
    'p2': "In: You are driving at {speed} m/s. What is the safe driving action?\nOut:",
    'p3': "In: Navigate safely at {speed} m/s. What action to take?\nOut:",
    'p4': "In: Predict the driving action to maintain safe driving at {speed} m/s.\nOut:",
}

SCENARIOS = {
    'highway': {'n': 15, 'speed': '30', 'difficulty': 'easy'},
    'urban': {'n': 15, 'speed': '15', 'difficulty': 'easy'},
    'ood_noise': {'n': 8, 'speed': '25', 'difficulty': 'ood'},
    'ood_blank': {'n': 8, 'speed': '25', 'difficulty': 'ood'},
    'ood_indoor': {'n': 8, 'speed': '25', 'difficulty': 'ood'},
    'ood_inverted': {'n': 8, 'speed': '30', 'difficulty': 'ood'},
    'ood_checker': {'n': 8, 'speed': '25', 'difficulty': 'ood'},
    'ood_blackout': {'n': 8, 'speed': '25', 'difficulty': 'ood'},
}


def create_scene_image(scenario, idx, size=(256, 256)):
    np.random.seed(idx * 3100 + hash(scenario) % 31000)
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


def main():
    print("=" * 70, flush=True)
    print("COSINE DISTANCE PROMPT ROBUSTNESS ON REAL OpenVLA-7B", flush=True)
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

    n_samples = sum(s['n'] for s in SCENARIOS.values())
    n_prompts = len(PROMPTS)
    total = n_samples * n_prompts
    print(f"Total inferences: {total} ({n_samples} samples × {n_prompts} prompts)", flush=True)
    print(flush=True)

    # Data structure: per-prompt, per-sample
    all_data = {p_name: [] for p_name in PROMPTS}
    all_hidden = {p_name: [] for p_name in PROMPTS}
    sample_idx = 0

    for p_name, p_template in PROMPTS.items():
        print(f"\n--- Prompt: {p_name} ---", flush=True)
        s_idx = 0
        for scenario, config in SCENARIOS.items():
            for i in range(config['n']):
                s_idx += 1
                sample_idx += 1
                image = create_scene_image(scenario, i)
                p = p_template.format(speed=config['speed'])

                inputs = processor(p, image).to(model.device, dtype=torch.bfloat16)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=7,
                        do_sample=False,
                        output_scores=True,
                        output_hidden_states=True,
                        return_dict_in_generate=True,
                    )

                # Hidden state
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                    last_step = outputs.hidden_states[-1]
                    if isinstance(last_step, tuple):
                        hidden = last_step[-1][0, -1, :].float().cpu().numpy()
                    else:
                        hidden = last_step[0, -1, :].float().cpu().numpy()
                else:
                    hidden = np.zeros(4096)

                # Action mass
                vocab_size = outputs.scores[0].shape[-1]
                action_start = vocab_size - 256
                dim_masses = []
                for score in outputs.scores[:7]:
                    full_logits = score[0].float()
                    full_probs = torch.softmax(full_logits, dim=0).cpu().numpy()
                    action_probs = full_probs[action_start:]
                    dim_masses.append(float(action_probs.sum()))

                all_data[p_name].append({
                    'scenario': scenario,
                    'difficulty': config['difficulty'],
                    'idx': i,
                    'action_mass': float(np.mean(dim_masses)),
                })
                all_hidden[p_name].append(hidden)

                if s_idx % 20 == 0 or (scenario == list(SCENARIOS.keys())[-1] and i == config['n'] - 1):
                    print(f"  [{sample_idx}/{total}] {p_name}/{scenario}_{i}: "
                          f"mass={float(np.mean(dim_masses)):.4f}", flush=True)

    # ===================================================================
    # Analysis
    # ===================================================================
    print("\n" + "=" * 70, flush=True)
    print("PROMPT ROBUSTNESS ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    easy_idxs = [i for i, s in enumerate(all_data['p1']) if s['difficulty'] == 'easy']
    ood_idxs = [i for i, s in enumerate(all_data['p1']) if s['difficulty'] == 'ood']

    # 1. Per-prompt AUROC for both signals
    print("\n1. Per-Prompt AUROC Comparison", flush=True)
    print("-" * 70, flush=True)
    print(f"  {'Prompt':>6} | {'Cosine AUROC':>12} | {'Mass AUROC':>12} | {'Δ':>8}", flush=True)
    print("  " + "-" * 50, flush=True)

    cos_aurocs = []
    mass_aurocs = []

    for p_name in PROMPTS:
        h_arr = np.array(all_hidden[p_name])

        # Cosine distance using easy centroid
        cal_mean = np.mean(h_arr[easy_idxs], axis=0)
        cal_norm = cal_mean / (np.linalg.norm(cal_mean) + 1e-10)

        cos_dists = []
        for idx in range(len(all_data[p_name])):
            h_norm = h_arr[idx] / (np.linalg.norm(h_arr[idx]) + 1e-10)
            cos_dists.append(1.0 - float(np.dot(h_norm, cal_norm)))

        easy_cos = [cos_dists[i] for i in easy_idxs]
        ood_cos = [cos_dists[i] for i in ood_idxs]
        cos_auroc = compute_auroc(ood_cos, easy_cos)
        cos_aurocs.append(cos_auroc)

        # Action mass
        easy_mass = [-all_data[p_name][i]['action_mass'] for i in easy_idxs]
        ood_mass = [-all_data[p_name][i]['action_mass'] for i in ood_idxs]
        mass_auroc = compute_auroc(ood_mass, easy_mass)
        mass_aurocs.append(mass_auroc)

        print(f"  {p_name:>6} | {cos_auroc:>12.3f} | {mass_auroc:>12.3f} | "
              f"{cos_auroc - mass_auroc:>+8.3f}", flush=True)

    print(f"\n  {'Mean':>6} | {np.mean(cos_aurocs):>12.3f} | {np.mean(mass_aurocs):>12.3f} | "
          f"{np.mean(cos_aurocs) - np.mean(mass_aurocs):>+8.3f}", flush=True)
    print(f"  {'Std':>6} | {np.std(cos_aurocs):>12.3f} | {np.std(mass_aurocs):>12.3f} |", flush=True)
    print(f"  {'Range':>6} | {max(cos_aurocs)-min(cos_aurocs):>12.3f} | "
          f"{max(mass_aurocs)-min(mass_aurocs):>12.3f} |", flush=True)

    # 2. Cross-prompt centroid transfer
    print("\n2. Cross-Prompt Centroid Transfer", flush=True)
    print("-" * 70, flush=True)
    print("  Calibrate on prompt X, test on prompt Y:", flush=True)
    header = 'Cal/Test'
    print(f"  {header:>8}", end="", flush=True)
    for p2 in PROMPTS:
        print(f" | {p2:>6}", end="", flush=True)
    print(flush=True)
    print("  " + "-" * 50, flush=True)

    for p_cal in PROMPTS:
        h_cal = np.array(all_hidden[p_cal])
        cal_mean = np.mean(h_cal[easy_idxs], axis=0)
        cal_norm = cal_mean / (np.linalg.norm(cal_mean) + 1e-10)

        print(f"  {p_cal:>8}", end="", flush=True)
        for p_test in PROMPTS:
            h_test = np.array(all_hidden[p_test])
            cos_dists = []
            for idx in range(len(all_data[p_test])):
                h_norm = h_test[idx] / (np.linalg.norm(h_test[idx]) + 1e-10)
                cos_dists.append(1.0 - float(np.dot(h_norm, cal_norm)))

            easy_cos = [cos_dists[i] for i in easy_idxs]
            ood_cos = [cos_dists[i] for i in ood_idxs]
            auroc = compute_auroc(ood_cos, easy_cos)
            print(f" | {auroc:>6.3f}", end="", flush=True)
        print(flush=True)

    # 3. Averaged centroid across all prompts
    print("\n3. Multi-Prompt Averaged Centroid", flush=True)
    print("-" * 70, flush=True)

    # Average hidden states across prompts for calibration
    avg_cal = np.zeros(4096)
    for p_name in PROMPTS:
        h_arr = np.array(all_hidden[p_name])
        avg_cal += np.mean(h_arr[easy_idxs], axis=0)
    avg_cal /= len(PROMPTS)
    avg_cal_norm = avg_cal / (np.linalg.norm(avg_cal) + 1e-10)

    for p_test in PROMPTS:
        h_test = np.array(all_hidden[p_test])
        cos_dists = []
        for idx in range(len(all_data[p_test])):
            h_norm = h_test[idx] / (np.linalg.norm(h_test[idx]) + 1e-10)
            cos_dists.append(1.0 - float(np.dot(h_norm, avg_cal_norm)))

        easy_cos = [cos_dists[i] for i in easy_idxs]
        ood_cos = [cos_dists[i] for i in ood_idxs]
        auroc = compute_auroc(ood_cos, easy_cos)
        print(f"  Avg centroid → test on {p_test}: AUROC = {auroc:.3f}", flush=True)

    # 4. Per-OOD-type robustness across prompts
    print("\n4. Per-OOD-Type Robustness Across Prompts", flush=True)
    print("-" * 70, flush=True)

    ood_types = [s for s in SCENARIOS if s.startswith('ood_')]
    for ood_type in ood_types:
        ood_type_idxs = [i for i, s in enumerate(all_data['p1']) if s['scenario'] == ood_type]

        cos_per_prompt = []
        mass_per_prompt = []

        for p_name in PROMPTS:
            h_arr = np.array(all_hidden[p_name])
            cal_mean = np.mean(h_arr[easy_idxs], axis=0)
            cal_norm = cal_mean / (np.linalg.norm(cal_mean) + 1e-10)

            cos_dists = []
            for idx in range(len(all_data[p_name])):
                h_norm = h_arr[idx] / (np.linalg.norm(h_arr[idx]) + 1e-10)
                cos_dists.append(1.0 - float(np.dot(h_norm, cal_norm)))

            easy_cos = [cos_dists[i] for i in easy_idxs]
            ood_cos = [cos_dists[i] for i in ood_type_idxs]
            cos_per_prompt.append(compute_auroc(ood_cos, easy_cos))

            easy_mass = [-all_data[p_name][i]['action_mass'] for i in easy_idxs]
            ood_mass = [-all_data[p_name][i]['action_mass'] for i in ood_type_idxs]
            mass_per_prompt.append(compute_auroc(ood_mass, easy_mass))

        print(f"  {ood_type:<15}: cos={np.mean(cos_per_prompt):.3f}±{np.std(cos_per_prompt):.3f}, "
              f"mass={np.mean(mass_per_prompt):.3f}±{np.std(mass_per_prompt):.3f}, "
              f"cos_range={max(cos_per_prompt)-min(cos_per_prompt):.3f}, "
              f"mass_range={max(mass_per_prompt)-min(mass_per_prompt):.3f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'timestamp': timestamp,
        'cos_aurocs': {p: a for p, a in zip(PROMPTS.keys(), cos_aurocs)},
        'mass_aurocs': {p: a for p, a in zip(PROMPTS.keys(), mass_aurocs)},
        'cos_mean': float(np.mean(cos_aurocs)),
        'cos_std': float(np.std(cos_aurocs)),
        'mass_mean': float(np.mean(mass_aurocs)),
        'mass_std': float(np.std(mass_aurocs)),
        'samples': {p: [{k: v for k, v in s.items()} for s in data]
                   for p, data in all_data.items()},
    }

    output_path = os.path.join(RESULTS_DIR, f"cosine_prompt_robustness_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
