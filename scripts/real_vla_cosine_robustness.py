"""
Cosine Distance Robustness Analysis on Real OpenVLA-7B.

Tests the robustness of the cosine distance OOD detector:
1. Calibration set size sensitivity (5, 10, 15, 20, 25 samples)
2. Bootstrap confidence intervals (20 random calibration splits)
3. Layer-wise analysis (extract from multiple LLM layers)
4. Per-scenario detailed breakdown with CIs

Experiment 29 in the CalibDrive series.
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

SCENARIOS = {
    'highway': {'n': 30, 'speed': '30', 'difficulty': 'easy'},
    'urban': {'n': 30, 'speed': '15', 'difficulty': 'easy'},
    'ood_noise': {'n': 15, 'speed': '25', 'difficulty': 'ood'},
    'ood_blank': {'n': 15, 'speed': '25', 'difficulty': 'ood'},
    'ood_indoor': {'n': 15, 'speed': '25', 'difficulty': 'ood'},
    'ood_inverted': {'n': 15, 'speed': '30', 'difficulty': 'ood'},
    'ood_checker': {'n': 15, 'speed': '25', 'difficulty': 'ood'},
    'ood_blackout': {'n': 15, 'speed': '25', 'difficulty': 'ood'},
}


def create_scene_image(scenario, idx, size=(256, 256)):
    np.random.seed(idx * 2900 + hash(scenario) % 29000)
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


def cosine_dist(v1, v2):
    return 1.0 - float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))


def main():
    print("=" * 70, flush=True)
    print("COSINE DISTANCE ROBUSTNESS ON REAL OpenVLA-7B", flush=True)
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

    # Figure out how many layers the model has
    n_layers = model.config.text_config.num_hidden_layers if hasattr(model.config, 'text_config') else 32
    # We'll sample layers: first, 1/4, 1/2, 3/4, last
    layer_indices = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    layer_indices = sorted(set(layer_indices))
    print(f"Extracting from layers: {layer_indices} (of {n_layers} total)", flush=True)

    total = sum(s['n'] for s in SCENARIOS.values())
    print(f"Total samples: {total}", flush=True)
    print(flush=True)

    prompt = "In: What action should the robot take to drive forward at {speed} m/s safely?\nOut:"

    all_samples = []
    all_hidden_per_layer = {l: [] for l in layer_indices}
    sample_idx = 0

    for scenario, config in SCENARIOS.items():
        for i in range(config['n']):
            sample_idx += 1
            image = create_scene_image(scenario, i)
            p = prompt.format(speed=config['speed'])

            inputs = processor(p, image).to(model.device, dtype=torch.bfloat16)

            t0 = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=7,
                    do_sample=False,
                    output_scores=True,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                )

            # Extract hidden states from multiple layers
            layer_hiddens = {}
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                last_step_hidden = outputs.hidden_states[-1]
                if isinstance(last_step_hidden, tuple):
                    for layer_idx in layer_indices:
                        if layer_idx < len(last_step_hidden):
                            h = last_step_hidden[layer_idx][0, -1, :].float().cpu().numpy()
                            layer_hiddens[layer_idx] = h
                        else:
                            layer_hiddens[layer_idx] = np.zeros(4096)
                else:
                    # Fallback: only last layer available
                    for layer_idx in layer_indices:
                        layer_hiddens[layer_idx] = last_step_hidden[0, -1, :].float().cpu().numpy()

            for layer_idx in layer_indices:
                all_hidden_per_layer[layer_idx].append(
                    layer_hiddens.get(layer_idx, np.zeros(4096)))

            # Get action mass
            vocab_size = outputs.scores[0].shape[-1]
            action_start = vocab_size - 256
            dim_masses = []
            for score in outputs.scores[:7]:
                full_logits = score[0].float()
                full_probs = torch.softmax(full_logits, dim=0).cpu().numpy()
                action_probs = full_probs[action_start:]
                dim_masses.append(float(action_probs.sum()))

            elapsed = time.time() - t0

            sample = {
                'scenario': scenario,
                'difficulty': config['difficulty'],
                'idx': i,
                'action_mass': float(np.mean(dim_masses)),
            }
            all_samples.append(sample)

            if i % 10 == 0 or i == config['n'] - 1:
                print(f"  [{sample_idx}/{total}] {scenario}_{i}: "
                      f"mass={sample['action_mass']:.4f} ({elapsed:.1f}s)", flush=True)

    # ===================================================================
    # Analysis
    # ===================================================================
    print("\n" + "=" * 70, flush=True)
    print("ROBUSTNESS ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    easy_idxs = [i for i, s in enumerate(all_samples) if s['difficulty'] == 'easy']
    ood_idxs = [i for i, s in enumerate(all_samples) if s['difficulty'] == 'ood']

    # ===================================================================
    # 1. Layer-wise AUROC
    # ===================================================================
    print("\n1. Layer-wise Cosine Distance AUROC", flush=True)
    print("-" * 70, flush=True)

    best_layer = None
    best_layer_auroc = 0

    for layer_idx in layer_indices:
        hidden_arr = np.array(all_hidden_per_layer[layer_idx])

        # Use all easy as calibration for layer comparison
        cal_mean = np.mean(hidden_arr[easy_idxs], axis=0)
        cal_mean_norm = cal_mean / (np.linalg.norm(cal_mean) + 1e-10)

        cos_dists = []
        for s_idx in range(len(all_samples)):
            h_norm = hidden_arr[s_idx] / (np.linalg.norm(hidden_arr[s_idx]) + 1e-10)
            cos_dists.append(1.0 - float(np.dot(h_norm, cal_mean_norm)))

        easy_cos = [cos_dists[i] for i in easy_idxs]
        ood_cos = [cos_dists[i] for i in ood_idxs]
        auroc = compute_auroc(ood_cos, easy_cos)

        if auroc > best_layer_auroc:
            best_layer_auroc = auroc
            best_layer = layer_idx

        # Per OOD type
        per_type = []
        for ood_type in [s for s in SCENARIOS if s.startswith('ood_')]:
            ood_s_idxs = [i for i in ood_idxs if all_samples[i]['scenario'] == ood_type]
            ood_type_cos = [cos_dists[i] for i in ood_s_idxs]
            per_type.append(compute_auroc(ood_type_cos, easy_cos))

        ood_names = [s for s in SCENARIOS if s.startswith('ood_')]
        print(f"\n  Layer {layer_idx:>2} (of {n_layers}): Overall AUROC = {auroc:.3f}", flush=True)
        for name, val in zip(ood_names, per_type):
            print(f"    {name:<15}: {val:.3f}", flush=True)

    print(f"\n  Best layer: {best_layer} (AUROC = {best_layer_auroc:.3f})", flush=True)

    # ===================================================================
    # 2. Calibration Set Size Sensitivity
    # ===================================================================
    print("\n2. Calibration Set Size Sensitivity (Last Layer)", flush=True)
    print("-" * 70, flush=True)

    hidden_arr = np.array(all_hidden_per_layer[layer_indices[-1]])  # Last layer

    for n_cal in [5, 10, 15, 20, 25, 30]:
        if n_cal > len(easy_idxs):
            continue

        # Bootstrap: 20 random splits
        aurocs_bootstrap = []
        per_type_bootstrap = {s: [] for s in SCENARIOS if s.startswith('ood_')}

        for bootstrap_i in range(20):
            np.random.seed(bootstrap_i * 100 + n_cal)
            shuffled = np.random.permutation(easy_idxs)
            cal_idxs = shuffled[:n_cal]
            test_easy_idxs = shuffled[n_cal:]

            if len(test_easy_idxs) == 0:
                continue

            cal_mean = np.mean(hidden_arr[cal_idxs], axis=0)
            cal_mean_norm = cal_mean / (np.linalg.norm(cal_mean) + 1e-10)

            cos_dists = []
            for s_idx in range(len(all_samples)):
                h_norm = hidden_arr[s_idx] / (np.linalg.norm(hidden_arr[s_idx]) + 1e-10)
                cos_dists.append(1.0 - float(np.dot(h_norm, cal_mean_norm)))

            test_easy_cos = [cos_dists[i] for i in test_easy_idxs]
            ood_cos = [cos_dists[i] for i in ood_idxs]
            auroc = compute_auroc(ood_cos, test_easy_cos)
            aurocs_bootstrap.append(auroc)

            for ood_type in [s for s in SCENARIOS if s.startswith('ood_')]:
                ood_s_idxs = [i for i in ood_idxs if all_samples[i]['scenario'] == ood_type]
                ood_type_cos = [cos_dists[i] for i in ood_s_idxs]
                per_type_bootstrap[ood_type].append(
                    compute_auroc(ood_type_cos, test_easy_cos))

        mean_auroc = np.mean(aurocs_bootstrap)
        std_auroc = np.std(aurocs_bootstrap)
        ci_lo = np.percentile(aurocs_bootstrap, 2.5)
        ci_hi = np.percentile(aurocs_bootstrap, 97.5)

        print(f"\n  n_cal={n_cal:>2}: AUROC = {mean_auroc:.3f} ± {std_auroc:.3f} "
              f"[{ci_lo:.3f}, {ci_hi:.3f}]", flush=True)

        for ood_type in [s for s in SCENARIOS if s.startswith('ood_')]:
            type_mean = np.mean(per_type_bootstrap[ood_type])
            type_std = np.std(per_type_bootstrap[ood_type])
            print(f"    {ood_type:<15}: {type_mean:.3f} ± {type_std:.3f}", flush=True)

    # ===================================================================
    # 3. Comparison with action mass at matched conditions
    # ===================================================================
    print("\n3. Cosine Distance vs Action Mass (Matched Conditions)", flush=True)
    print("-" * 70, flush=True)

    for n_cal in [5, 10, 25]:
        if n_cal > len(easy_idxs):
            continue

        cos_aurocs = []
        mass_aurocs = []

        for bootstrap_i in range(20):
            np.random.seed(bootstrap_i * 100 + n_cal)
            shuffled = np.random.permutation(easy_idxs)
            test_easy_idxs = shuffled[n_cal:]

            if len(test_easy_idxs) == 0:
                continue

            # Cosine
            cal_mean = np.mean(hidden_arr[shuffled[:n_cal]], axis=0)
            cal_mean_norm = cal_mean / (np.linalg.norm(cal_mean) + 1e-10)

            cos_dists = []
            for s_idx in range(len(all_samples)):
                h_norm = hidden_arr[s_idx] / (np.linalg.norm(hidden_arr[s_idx]) + 1e-10)
                cos_dists.append(1.0 - float(np.dot(h_norm, cal_mean_norm)))

            test_easy_cos = [cos_dists[i] for i in test_easy_idxs]
            ood_cos = [cos_dists[i] for i in ood_idxs]
            cos_aurocs.append(compute_auroc(ood_cos, test_easy_cos))

            # Action mass (no calibration needed)
            test_easy_mass = [-all_samples[i]['action_mass'] for i in test_easy_idxs]
            ood_mass = [-all_samples[i]['action_mass'] for i in ood_idxs]
            mass_aurocs.append(compute_auroc(ood_mass, test_easy_mass))

        print(f"\n  n_cal={n_cal:>2}:", flush=True)
        print(f"    Cosine: {np.mean(cos_aurocs):.3f} ± {np.std(cos_aurocs):.3f}", flush=True)
        print(f"    Mass:   {np.mean(mass_aurocs):.3f} ± {np.std(mass_aurocs):.3f}", flush=True)
        print(f"    Δ:      {np.mean(cos_aurocs) - np.mean(mass_aurocs):+.3f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'timestamp': timestamp,
        'layer_indices': layer_indices,
        'n_layers': n_layers,
        'best_layer': best_layer,
        'best_layer_auroc': best_layer_auroc,
        'samples': [{k: v for k, v in s.items()} for s in all_samples],
    }

    output_path = os.path.join(RESULTS_DIR, f"cosine_robustness_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
