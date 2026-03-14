"""
Mahalanobis Distance OOD Detection on Real OpenVLA-7B.

Improves on Experiment 27's L2 distance with:
1. Mahalanobis distance (accounts for covariance structure)
2. PCA-reduced hidden states (removes noise dimensions)
3. Cosine distance to centroid (direction-based)
4. kNN distance (non-parametric, captures local structure)
5. Combined multi-signal scoring with all distance metrics + action mass + entropy
6. Proper train/test split for evaluation

Experiment 28 in the CalibDrive series.
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
    'highway': {'n': 25, 'speed': '30', 'difficulty': 'easy'},
    'urban': {'n': 25, 'speed': '15', 'difficulty': 'easy'},
    'ood_noise': {'n': 12, 'speed': '25', 'difficulty': 'ood'},
    'ood_blank': {'n': 12, 'speed': '25', 'difficulty': 'ood'},
    'ood_indoor': {'n': 12, 'speed': '25', 'difficulty': 'ood'},
    'ood_inverted': {'n': 12, 'speed': '30', 'difficulty': 'ood'},
    'ood_checker': {'n': 12, 'speed': '25', 'difficulty': 'ood'},
    'ood_blackout': {'n': 12, 'speed': '25', 'difficulty': 'ood'},
}


def create_scene_image(scenario, idx, size=(256, 256)):
    np.random.seed(idx * 2800 + hash(scenario) % 28000)
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
    print("MAHALANOBIS DISTANCE OOD DETECTION ON REAL OpenVLA-7B", flush=True)
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

    prompt = "In: What action should the robot take to drive forward at {speed} m/s safely?\nOut:"

    all_samples = []
    all_hidden_states = []
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

            # Extract hidden state from last generated step, last layer, last token
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                last_step_hidden = outputs.hidden_states[-1]
                if isinstance(last_step_hidden, tuple):
                    last_layer = last_step_hidden[-1]
                    hidden = last_layer[0, -1, :].float().cpu().numpy()
                else:
                    hidden = last_step_hidden[0, -1, :].float().cpu().numpy()
            else:
                hidden = np.zeros(4096)

            # Get action mass and entropy
            vocab_size = outputs.scores[0].shape[-1]
            action_start = vocab_size - 256
            dim_masses = []
            dim_entropies = []
            for score in outputs.scores[:7]:
                full_logits = score[0].float()
                full_probs = torch.softmax(full_logits, dim=0).cpu().numpy()
                action_probs = full_probs[action_start:]
                dim_masses.append(float(action_probs.sum()))
                action_norm = action_probs / (action_probs.sum() + 1e-10)
                dim_entropies.append(float(-(action_norm * np.log(action_norm + 1e-10)).sum()))

            elapsed = time.time() - t0

            sample = {
                'scenario': scenario,
                'difficulty': config['difficulty'],
                'idx': i,
                'action_mass': float(np.mean(dim_masses)),
                'entropy': float(np.mean(dim_entropies)),
                'hidden_norm': float(np.linalg.norm(hidden)),
            }
            all_samples.append(sample)
            all_hidden_states.append(hidden)

            if i % 5 == 0 or i == config['n'] - 1:
                print(f"  [{sample_idx}/{total}] {scenario}_{i}: "
                      f"mass={sample['action_mass']:.4f}, ent={sample['entropy']:.3f}, "
                      f"h_norm={sample['hidden_norm']:.1f} ({elapsed:.1f}s)", flush=True)

    # ===================================================================
    # Analysis with proper train/test split
    # ===================================================================
    print("\n" + "=" * 70, flush=True)
    print("MAHALANOBIS DISTANCE ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    hidden_arr = np.array(all_hidden_states)

    # Split easy samples: first half calibration, second half test
    easy_idxs = [i for i, s in enumerate(all_samples) if s['difficulty'] == 'easy']
    np.random.seed(42)
    np.random.shuffle(easy_idxs)
    n_cal = len(easy_idxs) // 2
    cal_idxs = easy_idxs[:n_cal]
    test_easy_idxs = easy_idxs[n_cal:]
    ood_idxs = [i for i, s in enumerate(all_samples) if s['difficulty'] == 'ood']

    print(f"\nCalibration: {n_cal} easy samples", flush=True)
    print(f"Test: {len(test_easy_idxs)} easy + {len(ood_idxs)} OOD samples", flush=True)

    # Compute calibration statistics
    cal_hidden = hidden_arr[cal_idxs]
    cal_mean = np.mean(cal_hidden, axis=0)

    # ===================================================================
    # 1. L2 Distance (baseline from Exp 27)
    # ===================================================================
    print("\n1. L2 Distance to Calibration Centroid", flush=True)
    print("-" * 60, flush=True)

    for s_idx in range(len(all_samples)):
        all_samples[s_idx]['l2_dist'] = float(np.linalg.norm(hidden_arr[s_idx] - cal_mean))

    test_easy_l2 = [all_samples[i]['l2_dist'] for i in test_easy_idxs]
    for ood_type in [s for s in SCENARIOS if s.startswith('ood_')]:
        ood_s_idxs = [i for i in ood_idxs if all_samples[i]['scenario'] == ood_type]
        ood_l2 = [all_samples[i]['l2_dist'] for i in ood_s_idxs]
        auroc = compute_auroc(ood_l2, test_easy_l2)
        print(f"  {ood_type:<15}: L2 AUROC = {auroc:.3f} "
              f"(easy={np.mean(test_easy_l2):.1f}, ood={np.mean(ood_l2):.1f})", flush=True)

    all_ood_l2 = [all_samples[i]['l2_dist'] for i in ood_idxs]
    auroc_l2 = compute_auroc(all_ood_l2, test_easy_l2)
    print(f"\n  Overall L2 AUROC: {auroc_l2:.3f}", flush=True)

    # ===================================================================
    # 2. Cosine Distance to Centroid
    # ===================================================================
    print("\n2. Cosine Distance to Calibration Centroid", flush=True)
    print("-" * 60, flush=True)

    cal_mean_norm = cal_mean / (np.linalg.norm(cal_mean) + 1e-10)
    for s_idx in range(len(all_samples)):
        h_norm = hidden_arr[s_idx] / (np.linalg.norm(hidden_arr[s_idx]) + 1e-10)
        all_samples[s_idx]['cos_dist'] = 1.0 - float(np.dot(h_norm, cal_mean_norm))

    test_easy_cos = [all_samples[i]['cos_dist'] for i in test_easy_idxs]
    for ood_type in [s for s in SCENARIOS if s.startswith('ood_')]:
        ood_s_idxs = [i for i in ood_idxs if all_samples[i]['scenario'] == ood_type]
        ood_cos = [all_samples[i]['cos_dist'] for i in ood_s_idxs]
        auroc = compute_auroc(ood_cos, test_easy_cos)
        print(f"  {ood_type:<15}: Cos AUROC = {auroc:.3f} "
              f"(easy={np.mean(test_easy_cos):.4f}, ood={np.mean(ood_cos):.4f})", flush=True)

    all_ood_cos = [all_samples[i]['cos_dist'] for i in ood_idxs]
    auroc_cos = compute_auroc(all_ood_cos, test_easy_cos)
    print(f"\n  Overall Cosine AUROC: {auroc_cos:.3f}", flush=True)

    # ===================================================================
    # 3. Mahalanobis Distance (with PCA regularization)
    # ===================================================================
    print("\n3. Mahalanobis Distance (PCA-regularized)", flush=True)
    print("-" * 60, flush=True)

    # PCA on calibration hidden states
    cal_centered = cal_hidden - cal_mean
    # Use SVD for numerical stability (4096-d, n_cal samples)
    U, S, Vt = np.linalg.svd(cal_centered, full_matrices=False)

    for n_components in [10, 50, 100, 256]:
        if n_components > min(cal_centered.shape):
            continue

        # Project onto top PCA components
        V_k = Vt[:n_components].T  # (4096, k)
        S_k = S[:n_components]

        # Mahalanobis in PCA space: d^2 = sum((z_i / sigma_i)^2)
        maha_scores = []
        for s_idx in range(len(all_samples)):
            centered = hidden_arr[s_idx] - cal_mean
            projected = centered @ V_k  # (k,)
            maha = float(np.sqrt(np.sum((projected / (S_k / np.sqrt(n_cal - 1) + 1e-10)) ** 2)))
            maha_scores.append(maha)

        for s_idx in range(len(all_samples)):
            all_samples[s_idx][f'maha_{n_components}'] = maha_scores[s_idx]

        test_easy_maha = [maha_scores[i] for i in test_easy_idxs]
        all_ood_maha = [maha_scores[i] for i in ood_idxs]
        auroc_maha = compute_auroc(all_ood_maha, test_easy_maha)

        print(f"\n  PCA k={n_components}:", flush=True)
        print(f"    Overall Mahalanobis AUROC: {auroc_maha:.3f}", flush=True)

        # Per OOD type
        for ood_type in [s for s in SCENARIOS if s.startswith('ood_')]:
            ood_s_idxs = [i for i in ood_idxs if all_samples[i]['scenario'] == ood_type]
            ood_maha = [maha_scores[i] for i in ood_s_idxs]
            auroc = compute_auroc(ood_maha, test_easy_maha)
            print(f"    {ood_type:<15}: AUROC = {auroc:.3f}", flush=True)

    # ===================================================================
    # 4. kNN Distance
    # ===================================================================
    print("\n4. kNN Distance to Calibration Set", flush=True)
    print("-" * 60, flush=True)

    # Compute pairwise distances from each sample to calibration set
    for k in [1, 3, 5]:
        knn_scores = []
        for s_idx in range(len(all_samples)):
            dists = np.linalg.norm(cal_hidden - hidden_arr[s_idx], axis=1)
            knn_dist = float(np.mean(np.sort(dists)[:k]))
            knn_scores.append(knn_dist)

        for s_idx in range(len(all_samples)):
            all_samples[s_idx][f'knn_{k}'] = knn_scores[s_idx]

        test_easy_knn = [knn_scores[i] for i in test_easy_idxs]
        all_ood_knn = [knn_scores[i] for i in ood_idxs]
        auroc_knn = compute_auroc(all_ood_knn, test_easy_knn)

        print(f"\n  k={k}:", flush=True)
        print(f"    Overall kNN AUROC: {auroc_knn:.3f}", flush=True)

        for ood_type in [s for s in SCENARIOS if s.startswith('ood_')]:
            ood_s_idxs = [i for i in ood_idxs if all_samples[i]['scenario'] == ood_type]
            ood_knn = [knn_scores[i] for i in ood_s_idxs]
            auroc = compute_auroc(ood_knn, test_easy_knn)
            print(f"    {ood_type:<15}: AUROC = {auroc:.3f}", flush=True)

    # ===================================================================
    # 5. Comprehensive Combined Signals
    # ===================================================================
    print("\n5. Combined Multi-Signal OOD Scoring", flush=True)
    print("-" * 60, flush=True)

    # Normalize all signals to [0,1] (higher = more OOD)
    signal_names = ['l2_dist', 'cos_dist', 'maha_50', 'knn_3',
                    'action_mass_inv', 'entropy_score']

    # Prepare signals
    for s_idx in range(len(all_samples)):
        all_samples[s_idx]['action_mass_inv'] = 1.0 - all_samples[s_idx]['action_mass']
        # Higher entropy = more uncertain, but normalize later
        all_samples[s_idx]['entropy_score'] = all_samples[s_idx]['entropy']

    for sig in signal_names:
        vals = [all_samples[i].get(sig, 0) for i in range(len(all_samples))]
        v_min, v_max = min(vals), max(vals)
        for s_idx in range(len(all_samples)):
            raw = all_samples[s_idx].get(sig, 0)
            all_samples[s_idx][f'{sig}_norm'] = (raw - v_min) / (v_max - v_min + 1e-10)

    # Test individual signals
    print("\n  Individual Signal AUROCs:", flush=True)
    print(f"  {'Signal':<25} {'Overall':>8} {'Noise':>8} {'Blank':>8} {'Indoor':>8} "
          f"{'Inverted':>8} {'Checker':>8} {'Blackout':>8}", flush=True)
    print("  " + "-" * 90, flush=True)

    signal_aurocs = {}
    for sig in signal_names:
        test_easy_sig = [all_samples[i][f'{sig}_norm'] for i in test_easy_idxs]
        all_ood_sig = [all_samples[i][f'{sig}_norm'] for i in ood_idxs]
        overall_auroc = compute_auroc(all_ood_sig, test_easy_sig)
        signal_aurocs[sig] = overall_auroc

        per_type = []
        for ood_type in [s for s in SCENARIOS if s.startswith('ood_')]:
            ood_s_idxs = [i for i in ood_idxs if all_samples[i]['scenario'] == ood_type]
            ood_sig = [all_samples[i][f'{sig}_norm'] for i in ood_s_idxs]
            per_type.append(compute_auroc(ood_sig, test_easy_sig))

        print(f"  {sig:<25} {overall_auroc:>8.3f} " +
              " ".join(f"{a:>8.3f}" for a in per_type), flush=True)

    # Test combinations
    print("\n  Combined Signal AUROCs:", flush=True)
    combos = [
        ('L2 + Mass', ['l2_dist', 'action_mass_inv'], [0.5, 0.5]),
        ('L2 + Mass (0.25/0.75)', ['l2_dist', 'action_mass_inv'], [0.25, 0.75]),
        ('Cos + Mass', ['cos_dist', 'action_mass_inv'], [0.5, 0.5]),
        ('Maha + Mass', ['maha_50', 'action_mass_inv'], [0.5, 0.5]),
        ('kNN + Mass', ['knn_3', 'action_mass_inv'], [0.5, 0.5]),
        ('L2 + Mass + Ent', ['l2_dist', 'action_mass_inv', 'entropy_score'], [0.33, 0.34, 0.33]),
        ('Cos + Mass + Ent', ['cos_dist', 'action_mass_inv', 'entropy_score'], [0.33, 0.34, 0.33]),
        ('All hidden', ['l2_dist', 'cos_dist', 'maha_50', 'knn_3'], [0.25, 0.25, 0.25, 0.25]),
        ('All signals', ['l2_dist', 'cos_dist', 'maha_50', 'knn_3', 'action_mass_inv', 'entropy_score'],
         [1/6]*6),
        ('Best3: Cos+Maha+Mass', ['cos_dist', 'maha_50', 'action_mass_inv'], [0.33, 0.34, 0.33]),
    ]

    for combo_name, sigs, weights in combos:
        combined = []
        for s_idx in range(len(all_samples)):
            score = sum(w * all_samples[s_idx][f'{s}_norm'] for w, s in zip(weights, sigs))
            combined.append(score)

        test_easy_comb = [combined[i] for i in test_easy_idxs]
        all_ood_comb = [combined[i] for i in ood_idxs]
        auroc = compute_auroc(all_ood_comb, test_easy_comb)

        per_type_aurocs = []
        for ood_type in [s for s in SCENARIOS if s.startswith('ood_')]:
            ood_s_idxs = [i for i in ood_idxs if all_samples[i]['scenario'] == ood_type]
            ood_comb = [combined[i] for i in ood_s_idxs]
            per_type_aurocs.append(compute_auroc(ood_comb, test_easy_comb))

        print(f"  {combo_name:<25} {auroc:>8.3f} " +
              " ".join(f"{a:>8.3f}" for a in per_type_aurocs), flush=True)

    # ===================================================================
    # 6. PCA Visualization Data (for figure)
    # ===================================================================
    print("\n6. PCA Projection (First 3 Components)", flush=True)
    print("-" * 60, flush=True)

    V_3 = Vt[:3].T  # (4096, 3)
    variance_explained = S[:3] ** 2 / np.sum(S ** 2)
    print(f"  Variance explained: PC1={variance_explained[0]:.3f}, "
          f"PC2={variance_explained[1]:.3f}, PC3={variance_explained[2]:.3f}", flush=True)

    for s_idx in range(len(all_samples)):
        centered = hidden_arr[s_idx] - cal_mean
        projected = centered @ V_3
        all_samples[s_idx]['pca_1'] = float(projected[0])
        all_samples[s_idx]['pca_2'] = float(projected[1])
        all_samples[s_idx]['pca_3'] = float(projected[2])

    # Per-scenario PCA centroids
    print(f"\n  {'Scenario':<15} {'PC1':>8} {'PC2':>8} {'PC3':>8}", flush=True)
    for scenario in SCENARIOS:
        idxs = [i for i, s in enumerate(all_samples) if s['scenario'] == scenario]
        pc1 = np.mean([all_samples[i]['pca_1'] for i in idxs])
        pc2 = np.mean([all_samples[i]['pca_2'] for i in idxs])
        pc3 = np.mean([all_samples[i]['pca_3'] for i in idxs])
        print(f"  {scenario:<15} {pc1:>8.1f} {pc2:>8.1f} {pc3:>8.1f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'timestamp': timestamp,
        'n_calibration': n_cal,
        'n_test_easy': len(test_easy_idxs),
        'n_ood': len(ood_idxs),
        'pca_variance_explained': variance_explained.tolist(),
        'samples': [{k: v for k, v in s.items()} for s in all_samples],
    }

    output_path = os.path.join(RESULTS_DIR, f"mahalanobis_ood_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
