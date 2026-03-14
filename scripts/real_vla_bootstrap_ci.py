"""
Bootstrap Confidence Intervals for Key AUROC Comparisons.

Computes 95% CIs via bootstrap resampling (N=10000) for the critical
AUROC comparisons in the paper, establishing statistical significance.

Uses saved results from Experiments 27-47.

Experiment 48 in the CalibDrive series.
"""
import os
import json
import datetime
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

RESULTS_DIR = "/workspace/Vizuara-VLA-Research/experiments"
os.makedirs(RESULTS_DIR, exist_ok=True)
SIZE = (256, 256)


# Image generators
def create_highway(idx):
    rng = np.random.default_rng(idx * 5001)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]
    img[SIZE[0]//2:] = [80, 80, 80]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_urban(idx):
    rng = np.random.default_rng(idx * 5002)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [135, 206, 235]
    img[SIZE[0]//3:SIZE[0]//2] = [139, 119, 101]
    img[SIZE[0]//2:] = [60, 60, 60]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_noise(idx):
    rng = np.random.default_rng(idx * 5003)
    return rng.integers(0, 256, (*SIZE, 3), dtype=np.uint8)

def create_indoor(idx):
    rng = np.random.default_rng(idx * 5004)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:] = [200, 180, 160]
    img[SIZE[0]//2:] = [139, 90, 43]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_inverted(idx):
    return 255 - create_highway(idx + 3000)

def create_blackout(idx):
    return np.zeros((*SIZE, 3), dtype=np.uint8)

def create_blank(idx):
    rng = np.random.default_rng(idx * 5005)
    val = rng.integers(200, 256)
    return np.full((*SIZE, 3), val, dtype=np.uint8)


def extract_signals(model, processor, image, prompt):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=7, do_sample=False,
            output_scores=True, output_hidden_states=True,
            return_dict_in_generate=True,
        )
    if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
        last_step = outputs.hidden_states[-1]
        if isinstance(last_step, tuple):
            hidden = last_step[-1][0, -1, :].float().cpu().numpy()
        else:
            hidden = last_step[0, -1, :].float().cpu().numpy()
    else:
        hidden = np.zeros(4096)

    vocab_size = outputs.scores[0].shape[-1]
    action_start = vocab_size - 256
    masses = []
    entropies = []
    max_probs = []
    energies = []
    for score in outputs.scores[:7]:
        logits = score[0].float()
        probs = torch.softmax(logits, dim=0)
        masses.append(float(probs[action_start:].sum()))
        entropies.append(float(-torch.sum(probs * torch.log(probs + 1e-10))))
        max_probs.append(float(probs.max()))
        energies.append(float(-torch.logsumexp(logits, dim=0)))

    return {
        'hidden': hidden,
        'action_mass': float(np.mean(masses)),
        'entropy': float(np.mean(entropies)),
        'msp': float(np.mean(max_probs)),
        'energy': float(np.mean(energies)),
    }


def cosine_dist(a, b):
    return 1.0 - float(np.dot(a / (np.linalg.norm(a) + 1e-10),
                               b / (np.linalg.norm(b) + 1e-10)))


def bootstrap_auroc(labels, scores, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(labels)
    labels = np.array(labels)
    scores = np.array(scores)

    base_auroc = roc_auc_score(labels, scores)
    boot_aurocs = []

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        # Ensure both classes present
        if len(set(labels[idx])) < 2:
            continue
        boot_aurocs.append(roc_auc_score(labels[idx], scores[idx]))

    boot_aurocs = np.array(boot_aurocs)
    ci_lo = np.percentile(boot_aurocs, 2.5)
    ci_hi = np.percentile(boot_aurocs, 97.5)
    return base_auroc, ci_lo, ci_hi, np.std(boot_aurocs)


def main():
    print("=" * 70, flush=True)
    print("BOOTSTRAP CONFIDENCE INTERVALS", flush=True)
    print("=" * 70, flush=True)

    from transformers import AutoModelForVision2Seq, AutoProcessor
    print("Loading OpenVLA-7B...", flush=True)
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        "openvla/openvla-7b", trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.", flush=True)

    prompt = "In: What action should the robot take to drive forward at 25 m/s safely?\nOut:"

    # Large evaluation set for tight CIs
    print("\nGenerating large evaluation set (150 samples)...", flush=True)

    # Calibration
    cal_hidden = []
    for fn in [create_highway, create_urban]:
        for i in range(15):
            data = extract_signals(model, processor,
                                   Image.fromarray(fn(i + 8000)), prompt)
            cal_hidden.append(data['hidden'])
    centroid = np.mean(cal_hidden, axis=0)

    # Test set — larger for tight CIs
    test_fns = {
        'highway': (create_highway, False, 20),
        'urban': (create_urban, False, 20),
        'noise': (create_noise, True, 15),
        'blank': (create_blank, True, 15),
        'indoor': (create_indoor, True, 15),
        'inverted': (create_inverted, True, 15),
        'blackout': (create_blackout, True, 15),
    }

    all_data = []
    total = sum(v[2] for v in test_fns.values())
    cnt = 0
    for scene, (fn, is_ood, n) in test_fns.items():
        for i in range(n):
            cnt += 1
            data = extract_signals(model, processor,
                                   Image.fromarray(fn(i + 200)), prompt)
            data['cos_dist'] = cosine_dist(data['hidden'], centroid)
            data['scenario'] = scene
            data['is_ood'] = is_ood
            del data['hidden']
            all_data.append(data)
            if cnt % 20 == 0:
                print(f"  [{cnt}/{total}] {scene}_{i}", flush=True)

    easy = [d for d in all_data if not d['is_ood']]
    ood = [d for d in all_data if d['is_ood']]
    labels = [0]*len(easy) + [1]*len(ood)
    all_r = easy + ood

    print(f"\n  Total: {len(all_data)} ({len(easy)} ID, {len(ood)} OOD)", flush=True)

    # ===================================================================
    # Bootstrap CIs for each method
    # ===================================================================
    print("\n" + "=" * 70, flush=True)
    print("BOOTSTRAP CONFIDENCE INTERVALS (N=10000)", flush=True)
    print("=" * 70, flush=True)

    methods = {
        'Cosine distance': [d['cos_dist'] for d in all_r],
        'MSP (1-max prob)': [1 - d['msp'] for d in all_r],
        'Energy score': [d['energy'] for d in all_r],
        'Entropy': [d['entropy'] for d in all_r],
        'Action mass (1-mass)': [1 - d['action_mass'] for d in all_r],
    }

    print(f"\n  {'Method':<25} {'AUROC':>8} {'95% CI':>20} {'±':>8}", flush=True)
    print("  " + "-" * 65, flush=True)

    ci_results = {}
    for name, scores in sorted(methods.items(), key=lambda x: -roc_auc_score(labels, x[1])):
        auroc, ci_lo, ci_hi, std = bootstrap_auroc(labels, scores)
        print(f"  {name:<25} {auroc:>8.3f} [{ci_lo:.3f}, {ci_hi:.3f}] "
              f"±{std:.3f}", flush=True)
        ci_results[name] = {
            'auroc': auroc, 'ci_lo': ci_lo, 'ci_hi': ci_hi, 'std': std,
        }

    # Pairwise comparisons
    print("\n" + "=" * 70, flush=True)
    print("PAIRWISE SIGNIFICANCE TESTS", flush=True)
    print("=" * 70, flush=True)

    comparisons = [
        ('Cosine distance', 'MSP (1-max prob)'),
        ('Cosine distance', 'Energy score'),
        ('Cosine distance', 'Entropy'),
        ('Cosine distance', 'Action mass (1-mass)'),
        ('Action mass (1-mass)', 'MSP (1-max prob)'),
        ('Action mass (1-mass)', 'Energy score'),
    ]

    print(f"\n  {'Comparison':<50} {'Δ AUROC':>10} {'p-value':>10} {'Sig?':>6}",
          flush=True)
    print("  " + "-" * 80, flush=True)

    pair_results = []
    for method_a, method_b in comparisons:
        scores_a = np.array(methods[method_a])
        scores_b = np.array(methods[method_b])
        labels_arr = np.array(labels)

        base_diff = roc_auc_score(labels_arr, scores_a) - roc_auc_score(labels_arr, scores_b)

        n_boot = 10000
        rng = np.random.default_rng(42)
        n = len(labels_arr)
        count_reversed = 0

        for _ in range(n_boot):
            idx = rng.choice(n, size=n, replace=True)
            if len(set(labels_arr[idx])) < 2:
                continue
            auroc_a = roc_auc_score(labels_arr[idx], scores_a[idx])
            auroc_b = roc_auc_score(labels_arr[idx], scores_b[idx])
            if auroc_a <= auroc_b:
                count_reversed += 1

        p_value = count_reversed / n_boot
        sig = 'Yes' if p_value < 0.05 else 'No'
        name = f"{method_a} vs {method_b}"
        print(f"  {name:<50} {base_diff:>+10.3f} {p_value:>10.4f} {sig:>6}", flush=True)
        pair_results.append({
            'method_a': method_a, 'method_b': method_b,
            'delta': base_diff, 'p_value': p_value,
        })

    # Per-OOD type CIs for cosine distance
    print("\n" + "=" * 70, flush=True)
    print("PER-OOD TYPE CIs FOR COSINE DISTANCE", flush=True)
    print("=" * 70, flush=True)

    ood_types = ['noise', 'blank', 'indoor', 'inverted', 'blackout']
    cos_scores_full = [d['cos_dist'] for d in all_r]

    print(f"\n  {'OOD Type':<15} {'AUROC':>8} {'95% CI':>20}", flush=True)
    print("  " + "-" * 48, flush=True)

    per_type_results = {}
    for ood_type in ood_types:
        type_ood = [d for d in ood if d['scenario'] == ood_type]
        type_labels = [0]*len(easy) + [1]*len(type_ood)
        type_all = easy + type_ood
        type_scores = [d['cos_dist'] for d in type_all]
        auroc, ci_lo, ci_hi, std = bootstrap_auroc(type_labels, type_scores)
        print(f"  {ood_type:<15} {auroc:>8.3f} [{ci_lo:.3f}, {ci_hi:.3f}]", flush=True)
        per_type_results[ood_type] = {
            'auroc': auroc, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
        }

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'bootstrap_ci',
        'experiment_number': 48,
        'timestamp': timestamp,
        'n_cal': len(cal_hidden),
        'n_test': len(all_data),
        'n_bootstrap': 10000,
        'ci_results': ci_results,
        'pairwise': pair_results,
        'per_type': per_type_results,
    }
    output_path = os.path.join(RESULTS_DIR, f"bootstrap_ci_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
