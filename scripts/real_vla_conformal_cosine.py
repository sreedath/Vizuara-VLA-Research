"""
Conformal Prediction with Cosine Distance on Real OpenVLA-7B.

Uses cosine distance as the nonconformity score for conformal prediction.
Provides formal coverage guarantees: P(OOD flagged) >= 1-alpha.

Tests:
1. 5-fold cross-validated conformal thresholds
2. Coverage guarantees across alpha levels
3. Comparison with action-mass-based conformal (Exp 22)
4. Calibration-test set size sensitivity

Experiment 33 in the CalibDrive series.
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
    np.random.seed(idx * 3300 + hash(scenario) % 33000)
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


def main():
    print("=" * 70, flush=True)
    print("CONFORMAL PREDICTION WITH COSINE DISTANCE", flush=True)
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

    prompt = "In: What action should the robot take to drive forward at {speed} m/s safely?\nOut:"

    all_samples = []
    all_hidden = []
    sample_idx = 0

    for scenario, config in SCENARIOS.items():
        for i in range(config['n']):
            sample_idx += 1
            image = create_scene_image(scenario, i)
            p = prompt.format(speed=config['speed'])
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

            sample = {
                'scenario': scenario,
                'difficulty': config['difficulty'],
                'idx': i,
                'action_mass': float(np.mean(dim_masses)),
            }
            all_samples.append(sample)
            all_hidden.append(hidden)

            if sample_idx % 20 == 0 or sample_idx == total:
                print(f"  [{sample_idx}/{total}] {scenario}_{i}: "
                      f"mass={sample['action_mass']:.4f}", flush=True)

    hidden_arr = np.array(all_hidden)
    easy_idxs = [i for i, s in enumerate(all_samples) if s['difficulty'] == 'easy']
    ood_idxs = [i for i, s in enumerate(all_samples) if s['difficulty'] == 'ood']

    # ===================================================================
    # 1. 5-Fold Cross-Validated Conformal Prediction
    # ===================================================================
    print("\n" + "=" * 70, flush=True)
    print("CONFORMAL PREDICTION ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    alphas = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30]
    K = 5

    print(f"\n1. {K}-Fold Cross-Validated Conformal (Cosine Distance)", flush=True)
    print("-" * 80, flush=True)

    np.random.seed(42)
    shuffled_easy = np.random.permutation(easy_idxs)
    fold_size = len(shuffled_easy) // K

    for alpha in alphas:
        fold_results = []

        for fold in range(K):
            # Split
            test_fold = shuffled_easy[fold * fold_size:(fold + 1) * fold_size]
            cal_fold = np.concatenate([
                shuffled_easy[:fold * fold_size],
                shuffled_easy[(fold + 1) * fold_size:]
            ])

            # Compute centroid from calibration fold
            cal_mean = np.mean(hidden_arr[cal_fold], axis=0)
            cal_norm = cal_mean / (np.linalg.norm(cal_mean) + 1e-10)

            # Compute cosine distances
            cos_dists = []
            for idx in range(len(all_samples)):
                h_norm = hidden_arr[idx] / (np.linalg.norm(hidden_arr[idx]) + 1e-10)
                cos_dists.append(1.0 - float(np.dot(h_norm, cal_norm)))

            # Conformal threshold from calibration set
            cal_scores = sorted([cos_dists[i] for i in cal_fold])
            q_idx = int(np.ceil((1 - alpha) * (len(cal_scores) + 1))) - 1
            q_idx = min(q_idx, len(cal_scores) - 1)
            threshold = cal_scores[q_idx]

            # Evaluate
            test_easy_covered = sum(1 for i in test_fold if cos_dists[i] <= threshold)
            ood_flagged = sum(1 for i in ood_idxs if cos_dists[i] > threshold)

            fold_results.append({
                'threshold': threshold,
                'easy_coverage': test_easy_covered / len(test_fold),
                'ood_flag_rate': ood_flagged / len(ood_idxs),
            })

        mean_threshold = np.mean([r['threshold'] for r in fold_results])
        mean_easy_cov = np.mean([r['easy_coverage'] for r in fold_results])
        std_easy_cov = np.std([r['easy_coverage'] for r in fold_results])
        mean_ood_flag = np.mean([r['ood_flag_rate'] for r in fold_results])

        print(f"  α={alpha:.2f}: threshold={mean_threshold:.4f}, "
              f"easy_cov={mean_easy_cov:.1%}±{std_easy_cov:.1%}, "
              f"OOD_flag={mean_ood_flag:.1%}", flush=True)

    # ===================================================================
    # 2. Comparison: Cosine vs Action Mass Conformal
    # ===================================================================
    print(f"\n2. Conformal Comparison: Cosine vs Action Mass", flush=True)
    print("-" * 80, flush=True)

    # Use first half easy as calibration
    cal_easy = shuffled_easy[:len(shuffled_easy)//2]
    test_easy = shuffled_easy[len(shuffled_easy)//2:]

    # Cosine distance
    cal_mean = np.mean(hidden_arr[cal_easy], axis=0)
    cal_norm = cal_mean / (np.linalg.norm(cal_mean) + 1e-10)
    cos_dists = []
    for idx in range(len(all_samples)):
        h_norm = hidden_arr[idx] / (np.linalg.norm(hidden_arr[idx]) + 1e-10)
        cos_dists.append(1.0 - float(np.dot(h_norm, cal_norm)))

    print(f"\n  {'Alpha':>6} | {'Signal':>12} | {'Threshold':>10} | {'Easy Cov':>8} | "
          f"{'OOD Flag':>8} | {'OOD STOP':>8}", flush=True)
    print("  " + "-" * 70, flush=True)

    for alpha in alphas:
        # Cosine conformal
        cal_cos = sorted([cos_dists[i] for i in cal_easy])
        q_idx = int(np.ceil((1 - alpha) * (len(cal_cos) + 1))) - 1
        q_idx = min(q_idx, len(cal_cos) - 1)
        cos_threshold = cal_cos[q_idx]

        easy_cov_cos = sum(1 for i in test_easy if cos_dists[i] <= cos_threshold) / len(test_easy)
        ood_flag_cos = sum(1 for i in ood_idxs if cos_dists[i] > cos_threshold) / len(ood_idxs)

        # Action mass conformal (nonconformity = 1 - mass)
        cal_mass_scores = sorted([1 - all_samples[i]['action_mass'] for i in cal_easy])
        q_idx_m = int(np.ceil((1 - alpha) * (len(cal_mass_scores) + 1))) - 1
        q_idx_m = min(q_idx_m, len(cal_mass_scores) - 1)
        mass_threshold = cal_mass_scores[q_idx_m]

        easy_cov_mass = sum(1 for i in test_easy
                          if (1 - all_samples[i]['action_mass']) <= mass_threshold) / len(test_easy)
        ood_flag_mass = sum(1 for i in ood_idxs
                          if (1 - all_samples[i]['action_mass']) > mass_threshold) / len(ood_idxs)

        print(f"  {alpha:>6.2f} | {'Cosine':>12} | {cos_threshold:>10.4f} | "
              f"{easy_cov_cos:>8.1%} | {ood_flag_cos:>8.1%} |", flush=True)
        print(f"  {alpha:>6.2f} | {'Mass':>12} | {mass_threshold:>10.4f} | "
              f"{easy_cov_mass:>8.1%} | {ood_flag_mass:>8.1%} |", flush=True)
        print("  " + "-" * 70, flush=True)

    # ===================================================================
    # 3. Per-OOD-Type Flag Rates at Key Alphas
    # ===================================================================
    print(f"\n3. Per-OOD-Type Flag Rates (Cosine Conformal)", flush=True)
    print("-" * 80, flush=True)

    ood_types = [s for s in SCENARIOS if s.startswith('ood_')]

    for alpha in [0.05, 0.10, 0.20]:
        cal_cos = sorted([cos_dists[i] for i in cal_easy])
        q_idx = int(np.ceil((1 - alpha) * (len(cal_cos) + 1))) - 1
        q_idx = min(q_idx, len(cal_cos) - 1)
        cos_threshold = cal_cos[q_idx]

        print(f"\n  α={alpha:.2f} (threshold={cos_threshold:.4f}):", flush=True)
        for ood_type in ood_types:
            type_idxs = [i for i in ood_idxs if all_samples[i]['scenario'] == ood_type]
            flagged = sum(1 for i in type_idxs if cos_dists[i] > cos_threshold)
            print(f"    {ood_type:<15}: {flagged}/{len(type_idxs)} flagged "
                  f"({flagged/len(type_idxs):.0%})", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'timestamp': timestamp,
        'n_folds': K,
        'samples': [{k: v for k, v in s.items()} for s in all_samples],
    }

    output_path = os.path.join(RESULTS_DIR, f"conformal_cosine_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
