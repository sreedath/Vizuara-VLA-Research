"""
Bootstrap Confidence Intervals for Key AUROC Comparisons.

Computes 95% CIs via bootstrap resampling (N=10000) for the critical
AUROC comparisons in the paper, establishing statistical significance.

Tests:
1. Cosine vs Attention Max on standard OOD
2. Cosine vs Attention Max on near-OOD
3. Method hierarchy: Attention > Hidden State > Output
4. Calibrated vs calibration-free comparison

Experiment 68 in the CalibDrive series.
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
N_BOOTSTRAP = 10000


def create_highway(idx):
    rng = np.random.default_rng(idx * 5001)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]
    img[SIZE[0]//2:] = [80, 80, 80]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [255, 255, 255]
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

# Near-OOD types
def create_twilight_highway(idx):
    rng = np.random.default_rng(idx * 5010)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [70, 50, 80]
    img[SIZE[0]//2:] = [60, 60, 60]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [200, 200, 100]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_wet_highway(idx):
    rng = np.random.default_rng(idx * 5011)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [100, 120, 140]
    img[SIZE[0]//2:] = [50, 55, 65]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [180, 180, 200]
    for y in range(SIZE[0]//2, SIZE[0]):
        if rng.random() > 0.7:
            img[y, :] = np.clip(img[y, :].astype(np.int16) + 20, 0, 255).astype(np.uint8)
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_construction(idx):
    rng = np.random.default_rng(idx * 5012)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]
    img[SIZE[0]//2:] = [80, 80, 80]
    barrier_x = rng.integers(SIZE[1]//4, 3*SIZE[1]//4)
    img[SIZE[0]//2:3*SIZE[0]//4, max(0,barrier_x-20):barrier_x+20] = [255, 140, 0]
    for cone_x in rng.integers(0, SIZE[1], size=5):
        img[SIZE[0]-30:SIZE[0]-10, max(0,cone_x-5):cone_x+5] = [255, 100, 0]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_occluded(idx):
    rng = np.random.default_rng(idx * 5013)
    img = create_highway(idx + 7000)
    for _ in range(rng.integers(3, 8)):
        cx, cy = rng.integers(0, SIZE[1]), rng.integers(0, SIZE[0])
        r = rng.integers(10, 40)
        for dy in range(-r, r):
            for dx in range(-r, r):
                if dx*dx + dy*dy <= r*r:
                    ny, nx = cy+dy, cx+dx
                    if 0 <= ny < SIZE[0] and 0 <= nx < SIZE[1]:
                        img[ny, nx] = np.clip(img[ny, nx].astype(np.int16) - 80, 0, 255).astype(np.uint8)
    return img

def create_snow(idx):
    rng = np.random.default_rng(idx * 5014)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [200, 200, 210]
    img[SIZE[0]//2:] = [220, 220, 230]
    img[SIZE[0]//2:, SIZE[1]//2-2:SIZE[1]//2+2] = [180, 180, 190]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

# Far-OOD
def create_noise(idx):
    rng = np.random.default_rng(idx * 5003)
    return rng.integers(0, 256, (*SIZE, 3), dtype=np.uint8)

def create_indoor(idx):
    rng = np.random.default_rng(idx * 5004)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:] = [200, 180, 160]
    img[SIZE[0]//2:, :] = [100, 80, 60]
    img[:SIZE[0]//3, SIZE[1]//3:2*SIZE[1]//3] = [150, 200, 255]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_blackout(idx):
    return np.zeros((*SIZE, 3), dtype=np.uint8)


def extract_signals(model, processor, image, prompt):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    result = {}

    with torch.no_grad():
        fwd = model(**inputs, output_attentions=True, output_hidden_states=True)

    if hasattr(fwd, 'attentions') and fwd.attentions:
        attn = fwd.attentions[-1][0].float().cpu().numpy()
        n_heads = attn.shape[0]
        last_attn = attn[:, -1, :]
        result['attn_max'] = float(np.mean([np.max(last_attn[h]) for h in range(n_heads)]))
        result['attn_entropy'] = float(np.mean([
            -np.sum((last_attn[h]+1e-10) * np.log(last_attn[h]+1e-10))
            for h in range(n_heads)
        ]))

    if hasattr(fwd, 'hidden_states') and fwd.hidden_states:
        result['hidden'] = fwd.hidden_states[-1][0, -1, :].float().cpu().numpy()

    # Output logits for MSP/energy
    if hasattr(fwd, 'logits') and fwd.logits is not None:
        logits = fwd.logits[0, -1, :].float().cpu().numpy()
        probs = np.exp(logits - np.max(logits))
        probs = probs / probs.sum()
        result['msp'] = float(np.max(probs))
        result['energy'] = float(np.log(np.sum(np.exp(logits - np.max(logits)))) + np.max(logits))

    return result


def cosine_dist(a, b):
    return 1.0 - float(np.dot(a / (np.linalg.norm(a) + 1e-10),
                               b / (np.linalg.norm(b) + 1e-10)))


def bootstrap_auroc(labels, scores, n_bootstrap=N_BOOTSTRAP, seed=42):
    """Compute AUROC with bootstrap 95% CI."""
    rng = np.random.default_rng(seed)
    labels = np.array(labels)
    scores = np.array(scores)
    n = len(labels)

    point_auroc = roc_auc_score(labels, scores)

    boot_aurocs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        b_labels = labels[idx]
        b_scores = scores[idx]
        # Skip if only one class in bootstrap sample
        if len(np.unique(b_labels)) < 2:
            continue
        boot_aurocs.append(roc_auc_score(b_labels, b_scores))

    boot_aurocs = np.array(boot_aurocs)
    ci_low = float(np.percentile(boot_aurocs, 2.5))
    ci_high = float(np.percentile(boot_aurocs, 97.5))
    se = float(np.std(boot_aurocs))

    return {
        'auroc': float(point_auroc),
        'ci_low': ci_low,
        'ci_high': ci_high,
        'se': se,
        'n_valid_boots': len(boot_aurocs),
    }


def bootstrap_auroc_diff(labels, scores_a, scores_b, n_bootstrap=N_BOOTSTRAP, seed=42):
    """Bootstrap test for difference in AUROC between two methods."""
    rng = np.random.default_rng(seed)
    labels = np.array(labels)
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)
    n = len(labels)

    point_diff = roc_auc_score(labels, scores_a) - roc_auc_score(labels, scores_b)

    boot_diffs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        b_labels = labels[idx]
        if len(np.unique(b_labels)) < 2:
            continue
        diff = roc_auc_score(b_labels, scores_a[idx]) - roc_auc_score(b_labels, scores_b[idx])
        boot_diffs.append(diff)

    boot_diffs = np.array(boot_diffs)
    ci_low = float(np.percentile(boot_diffs, 2.5))
    ci_high = float(np.percentile(boot_diffs, 97.5))
    # p-value: proportion of bootstraps where difference crosses zero
    if point_diff > 0:
        p_value = float(np.mean(boot_diffs <= 0))
    else:
        p_value = float(np.mean(boot_diffs >= 0))

    return {
        'diff': float(point_diff),
        'ci_low': ci_low,
        'ci_high': ci_high,
        'p_value': p_value,
        'significant_at_05': bool(p_value < 0.05),
        'n_valid_boots': len(boot_diffs),
    }


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

    # Calibration
    print("\nCalibrating...", flush=True)
    cal_hidden = []
    for fn in [create_highway, create_urban]:
        for i in range(15):
            sig = extract_signals(model, processor,
                                  Image.fromarray(fn(i + 9000)), prompt)
            if 'hidden' in sig:
                cal_hidden.append(sig['hidden'])
    centroid = np.mean(cal_hidden, axis=0)
    print(f"  Calibration: {len(cal_hidden)} samples", flush=True)

    # Collect data with larger sample sizes for tighter CIs
    test_fns = {
        # ID
        'highway': (create_highway, 'id', 15),
        'urban': (create_urban, 'id', 15),
        # Near-OOD
        'twilight': (create_twilight_highway, 'near_ood', 10),
        'wet': (create_wet_highway, 'near_ood', 10),
        'construction': (create_construction, 'near_ood', 10),
        'occluded': (create_occluded, 'near_ood', 10),
        'snow': (create_snow, 'near_ood', 10),
        # Far-OOD
        'noise': (create_noise, 'far_ood', 10),
        'indoor': (create_indoor, 'far_ood', 10),
        'blackout': (create_blackout, 'far_ood', 10),
    }

    all_data = []
    cnt = 0
    total = sum(v[2] for v in test_fns.values())
    for scene, (fn, ood_type, n) in test_fns.items():
        for i in range(n):
            cnt += 1
            sig = extract_signals(model, processor,
                                  Image.fromarray(fn(i + 300)), prompt)
            sig['scenario'] = scene
            sig['ood_type'] = ood_type
            sig['is_ood'] = ood_type != 'id'
            if 'hidden' in sig:
                sig['cosine'] = cosine_dist(sig['hidden'], centroid)
            all_data.append(sig)
            if cnt % 10 == 0:
                print(f"  [{cnt}/{total}] {scene}", flush=True)

    print(f"\nCollected {len(all_data)} samples.", flush=True)

    # Split by type
    id_data = [d for d in all_data if d['ood_type'] == 'id']
    near_ood = [d for d in all_data if d['ood_type'] == 'near_ood']
    far_ood = [d for d in all_data if d['ood_type'] == 'far_ood']
    all_ood = near_ood + far_ood

    print("\n" + "=" * 70, flush=True)
    print("BOOTSTRAP AUROC WITH 95% CIs", flush=True)
    print("=" * 70, flush=True)

    results = {}

    # 1. Individual method AUROCs with CIs
    print("\n  1. Individual Method AUROCs:", flush=True)
    for test_name, ood_group in [('far_ood', far_ood), ('near_ood', near_ood), ('all_ood', all_ood)]:
        labels = [0]*len(id_data) + [1]*len(ood_group)
        results[test_name] = {}

        for sig_name in ['cosine', 'attn_max', 'attn_entropy', 'msp', 'energy']:
            if sig_name == 'cosine':
                scores = [d.get('cosine', 0) for d in id_data] + [d.get('cosine', 0) for d in ood_group]
            elif sig_name == 'attn_max':
                scores = [d.get('attn_max', 0) for d in id_data] + [d.get('attn_max', 0) for d in ood_group]
            elif sig_name == 'attn_entropy':
                scores = [-d.get('attn_entropy', 0) for d in id_data] + [-d.get('attn_entropy', 0) for d in ood_group]
            elif sig_name == 'msp':
                scores = [-d.get('msp', 1) for d in id_data] + [-d.get('msp', 1) for d in ood_group]
            elif sig_name == 'energy':
                scores = [-d.get('energy', 0) for d in id_data] + [-d.get('energy', 0) for d in ood_group]

            boot = bootstrap_auroc(labels, scores)
            results[test_name][sig_name] = boot
            print(f"    {test_name:<12} {sig_name:<15}: AUROC={boot['auroc']:.3f} "
                  f"[{boot['ci_low']:.3f}, {boot['ci_high']:.3f}] SE={boot['se']:.4f}",
                  flush=True)

    # 2. Pairwise comparisons
    print("\n  2. Pairwise Method Comparisons (AUROC difference):", flush=True)
    comparisons = {}

    for test_name, ood_group in [('far_ood', far_ood), ('near_ood', near_ood), ('all_ood', all_ood)]:
        labels = [0]*len(id_data) + [1]*len(ood_group)
        comparisons[test_name] = {}

        # Cosine vs Attn Max
        cos_scores = np.array([d.get('cosine', 0) for d in id_data] + [d.get('cosine', 0) for d in ood_group])
        attn_scores = np.array([d.get('attn_max', 0) for d in id_data] + [d.get('attn_max', 0) for d in ood_group])
        msp_scores = np.array([-d.get('msp', 1) for d in id_data] + [-d.get('msp', 1) for d in ood_group])
        energy_scores = np.array([-d.get('energy', 0) for d in id_data] + [-d.get('energy', 0) for d in ood_group])

        pairs = [
            ('attn_max_vs_cosine', attn_scores, cos_scores),
            ('cosine_vs_msp', cos_scores, msp_scores),
            ('attn_max_vs_msp', attn_scores, msp_scores),
            ('cosine_vs_energy', cos_scores, energy_scores),
        ]

        for pair_name, scores_a, scores_b in pairs:
            diff_result = bootstrap_auroc_diff(labels, scores_a, scores_b)
            comparisons[test_name][pair_name] = diff_result
            sig_marker = "*" if diff_result['significant_at_05'] else " "
            print(f"    {test_name:<12} {pair_name:<25}: Δ={diff_result['diff']:+.3f} "
                  f"[{diff_result['ci_low']:+.3f}, {diff_result['ci_high']:+.3f}] "
                  f"p={diff_result['p_value']:.4f} {sig_marker}", flush=True)

    # 3. Per near-OOD type CIs
    print("\n  3. Per Near-OOD Type CIs:", flush=True)
    per_near_ood = {}
    for scene in ['twilight', 'wet', 'construction', 'occluded', 'snow']:
        scene_data = [d for d in all_data if d['scenario'] == scene]
        labels = [0]*len(id_data) + [1]*len(scene_data)
        per_near_ood[scene] = {}

        for sig_name in ['cosine', 'attn_max']:
            if sig_name == 'cosine':
                scores = [d.get('cosine', 0) for d in id_data] + [d.get('cosine', 0) for d in scene_data]
            else:
                scores = [d.get('attn_max', 0) for d in id_data] + [d.get('attn_max', 0) for d in scene_data]

            boot = bootstrap_auroc(labels, scores)
            per_near_ood[scene][sig_name] = boot
            print(f"    {scene:<15} {sig_name:<12}: AUROC={boot['auroc']:.3f} "
                  f"[{boot['ci_low']:.3f}, {boot['ci_high']:.3f}]", flush=True)

    # 4. Effect size (Cohen's d) for key comparisons
    print("\n  4. Effect Sizes (Cohen's d):", flush=True)
    effect_sizes = {}
    for sig_name in ['cosine', 'attn_max', 'attn_entropy']:
        if sig_name == 'cosine':
            id_vals = np.array([d.get('cosine', 0) for d in id_data])
            ood_vals = np.array([d.get('cosine', 0) for d in all_ood])
        elif sig_name == 'attn_max':
            id_vals = np.array([d.get('attn_max', 0) for d in id_data])
            ood_vals = np.array([d.get('attn_max', 0) for d in all_ood])
        else:
            id_vals = np.array([d.get('attn_entropy', 0) for d in id_data])
            ood_vals = np.array([d.get('attn_entropy', 0) for d in all_ood])

        pooled_std = np.sqrt((np.std(id_vals)**2 + np.std(ood_vals)**2) / 2)
        d = abs(np.mean(id_vals) - np.mean(ood_vals)) / (pooled_std + 1e-10)
        effect_sizes[sig_name] = {
            'cohens_d': float(d),
            'id_mean': float(np.mean(id_vals)),
            'id_std': float(np.std(id_vals)),
            'ood_mean': float(np.mean(ood_vals)),
            'ood_std': float(np.std(ood_vals)),
        }
        magnitude = "large" if d > 0.8 else "medium" if d > 0.5 else "small"
        print(f"    {sig_name:<15}: d={d:.2f} ({magnitude})  "
              f"ID={np.mean(id_vals):.4f}±{np.std(id_vals):.4f}  "
              f"OOD={np.mean(ood_vals):.4f}±{np.std(ood_vals):.4f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'bootstrap_ci',
        'experiment_number': 68,
        'timestamp': timestamp,
        'n_bootstrap': N_BOOTSTRAP,
        'n_id': len(id_data),
        'n_near_ood': len(near_ood),
        'n_far_ood': len(far_ood),
        'individual_aurocs': results,
        'pairwise_comparisons': comparisons,
        'per_near_ood': per_near_ood,
        'effect_sizes': effect_sizes,
    }
    output_path = os.path.join(RESULTS_DIR, f"bootstrap_ci_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
