"""
Temporal Autocorrelation of OOD Scores.

Analyzes how OOD scores evolve over consecutive frames in simulated
trajectories. ID trajectories should show stable, low scores while
OOD trajectories show elevated, potentially drifting scores.

Tests:
1. Autocorrelation of cosine distance within trajectories
2. Score variance within vs between trajectories
3. Transition detection (ID → OOD mid-trajectory)
4. Smoothed vs raw score detection

Experiment 59 in the CalibDrive series.
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

def create_transition(idx, step, total_steps):
    """Create a trajectory that transitions from ID to OOD mid-way."""
    alpha = step / total_steps
    if alpha < 0.4:
        return create_highway(idx * 100 + step)
    elif alpha < 0.6:
        # Blend highway and noise
        blend = (alpha - 0.4) / 0.2
        hw = create_highway(idx * 100 + step).astype(np.float32)
        ns = create_noise(idx * 100 + step).astype(np.float32)
        return ((1 - blend) * hw + blend * ns).astype(np.uint8)
    else:
        return create_noise(idx * 100 + step)


def extract_hidden(model, processor, image, prompt):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=7, do_sample=False,
            output_hidden_states=True,
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
    return hidden


def cosine_dist(a, b):
    return 1.0 - float(np.dot(a / (np.linalg.norm(a) + 1e-10),
                               b / (np.linalg.norm(b) + 1e-10)))


def autocorrelation(x, lag=1):
    """Compute autocorrelation at given lag."""
    n = len(x)
    if n <= lag:
        return 0.0
    x = np.array(x)
    mean = np.mean(x)
    var = np.var(x)
    if var < 1e-10:
        return 0.0
    return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])


def main():
    print("=" * 70, flush=True)
    print("TEMPORAL AUTOCORRELATION ANALYSIS", flush=True)
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
        for i in range(10):
            h = extract_hidden(model, processor,
                               Image.fromarray(fn(i + 9000)), prompt)
            cal_hidden.append(h)
    centroid = np.mean(cal_hidden, axis=0)
    print(f"  Calibration: {len(cal_hidden)} samples", flush=True)

    # Generate trajectories (8 steps each)
    traj_len = 8
    trajectories = []

    # 5 ID trajectories (highway, varying seeds)
    print("\nGenerating ID trajectories...", flush=True)
    for t in range(5):
        traj = {'type': 'highway', 'is_ood': False, 'scores': []}
        for step in range(traj_len):
            h = extract_hidden(model, processor,
                               Image.fromarray(create_highway(t * 100 + step + 500)), prompt)
            score = cosine_dist(h, centroid)
            traj['scores'].append(score)
        trajectories.append(traj)
        print(f"  Highway traj {t}: mean={np.mean(traj['scores']):.4f}", flush=True)

    # 3 ID trajectories (urban)
    for t in range(3):
        traj = {'type': 'urban', 'is_ood': False, 'scores': []}
        for step in range(traj_len):
            h = extract_hidden(model, processor,
                               Image.fromarray(create_urban(t * 100 + step + 500)), prompt)
            score = cosine_dist(h, centroid)
            traj['scores'].append(score)
        trajectories.append(traj)
        print(f"  Urban traj {t}: mean={np.mean(traj['scores']):.4f}", flush=True)

    # 3 OOD trajectories (noise)
    print("\nGenerating OOD trajectories...", flush=True)
    for t in range(3):
        traj = {'type': 'noise', 'is_ood': True, 'scores': []}
        for step in range(traj_len):
            h = extract_hidden(model, processor,
                               Image.fromarray(create_noise(t * 100 + step + 500)), prompt)
            score = cosine_dist(h, centroid)
            traj['scores'].append(score)
        trajectories.append(traj)
        print(f"  Noise traj {t}: mean={np.mean(traj['scores']):.4f}", flush=True)

    # 3 OOD trajectories (indoor)
    for t in range(3):
        traj = {'type': 'indoor', 'is_ood': True, 'scores': []}
        for step in range(traj_len):
            h = extract_hidden(model, processor,
                               Image.fromarray(create_indoor(t * 100 + step + 500)), prompt)
            score = cosine_dist(h, centroid)
            traj['scores'].append(score)
        trajectories.append(traj)
        print(f"  Indoor traj {t}: mean={np.mean(traj['scores']):.4f}", flush=True)

    # 4 Transition trajectories (ID → OOD)
    print("\nGenerating transition trajectories...", flush=True)
    for t in range(4):
        traj = {'type': 'transition', 'is_ood': True, 'scores': [], 'step_labels': []}
        for step in range(traj_len):
            img = create_transition(t + 700, step, traj_len)
            h = extract_hidden(model, processor,
                               Image.fromarray(img), prompt)
            score = cosine_dist(h, centroid)
            traj['scores'].append(score)
            traj['step_labels'].append(0 if step < 3 else 1)
        trajectories.append(traj)
        print(f"  Transition traj {t}: scores={[f'{s:.3f}' for s in traj['scores']]}", flush=True)

    # Analysis
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    # 1. Autocorrelation
    print("\n  Autocorrelation (lag-1):", flush=True)
    id_autocorrs = []
    ood_autocorrs = []
    for traj in trajectories:
        if traj['type'] == 'transition':
            continue
        ac = autocorrelation(traj['scores'], lag=1)
        if traj['is_ood']:
            ood_autocorrs.append(ac)
        else:
            id_autocorrs.append(ac)
        print(f"    {traj['type']}: autocorr={ac:.3f}", flush=True)

    print(f"\n    ID mean autocorr: {np.mean(id_autocorrs):.3f} ± {np.std(id_autocorrs):.3f}",
          flush=True)
    print(f"    OOD mean autocorr: {np.mean(ood_autocorrs):.3f} ± {np.std(ood_autocorrs):.3f}",
          flush=True)

    # 2. Within-trajectory variance
    print("\n  Within-trajectory variance:", flush=True)
    id_vars = []
    ood_vars = []
    for traj in trajectories:
        if traj['type'] == 'transition':
            continue
        v = np.var(traj['scores'])
        if traj['is_ood']:
            ood_vars.append(v)
        else:
            id_vars.append(v)

    print(f"    ID intra-traj var: {np.mean(id_vars):.6f} ± {np.std(id_vars):.6f}", flush=True)
    print(f"    OOD intra-traj var: {np.mean(ood_vars):.6f} ± {np.std(ood_vars):.6f}", flush=True)

    # 3. Trajectory-level detection (mean score per trajectory)
    print("\n  Trajectory-level detection:", flush=True)
    traj_labels = []
    traj_mean_scores = []
    traj_max_scores = []
    traj_std_scores = []
    for traj in trajectories:
        if traj['type'] == 'transition':
            continue
        traj_labels.append(1 if traj['is_ood'] else 0)
        traj_mean_scores.append(np.mean(traj['scores']))
        traj_max_scores.append(np.max(traj['scores']))
        traj_std_scores.append(np.std(traj['scores']))

    for name, scores in [('mean', traj_mean_scores), ('max', traj_max_scores),
                          ('std', traj_std_scores)]:
        auroc = roc_auc_score(traj_labels, scores)
        print(f"    {name}: AUROC={auroc:.3f}", flush=True)

    # 4. Smoothed detection (EMA)
    print("\n  Smoothed detection (EMA):", flush=True)
    for alpha in [0.3, 0.5, 0.7, 0.9]:
        all_labels = []
        all_smoothed = []
        for traj in trajectories:
            if traj['type'] == 'transition':
                continue
            ema = traj['scores'][0]
            for s in traj['scores'][1:]:
                ema = alpha * s + (1 - alpha) * ema
            all_labels.append(1 if traj['is_ood'] else 0)
            all_smoothed.append(ema)
        auroc = roc_auc_score(all_labels, all_smoothed)
        print(f"    EMA alpha={alpha}: AUROC={auroc:.3f}", flush=True)

    # 5. Step-level detection improvement with context
    print("\n  Step-level detection with trajectory context:", flush=True)
    # Individual steps
    step_labels = []
    step_scores_raw = []
    for traj in trajectories:
        if traj['type'] == 'transition':
            continue
        for s in traj['scores']:
            step_labels.append(1 if traj['is_ood'] else 0)
            step_scores_raw.append(s)
    raw_auroc = roc_auc_score(step_labels, step_scores_raw)
    print(f"    Raw per-step: AUROC={raw_auroc:.3f} (n={len(step_labels)})", flush=True)

    # Rolling window (3-step)
    step_labels_w = []
    step_scores_w = []
    for traj in trajectories:
        if traj['type'] == 'transition':
            continue
        scores = traj['scores']
        for i in range(2, len(scores)):
            window_mean = np.mean(scores[max(0, i-2):i+1])
            step_labels_w.append(1 if traj['is_ood'] else 0)
            step_scores_w.append(window_mean)
    window_auroc = roc_auc_score(step_labels_w, step_scores_w)
    print(f"    3-step window: AUROC={window_auroc:.3f} (n={len(step_labels_w)})", flush=True)

    # 6. Transition detection
    print("\n  Transition detection:", flush=True)
    trans_trajs = [t for t in trajectories if t['type'] == 'transition']
    for i, traj in enumerate(trans_trajs):
        scores = traj['scores']
        # Compute score jump
        first_half = np.mean(scores[:3])
        second_half = np.mean(scores[5:])
        jump = second_half - first_half
        print(f"    Traj {i}: first_half={first_half:.4f}, second_half={second_half:.4f}, "
              f"jump={jump:+.4f}", flush=True)

    # Mean transition statistics
    jumps = [np.mean(t['scores'][5:]) - np.mean(t['scores'][:3]) for t in trans_trajs]
    print(f"\n    Mean jump: {np.mean(jumps):+.4f} ± {np.std(jumps):.4f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    traj_data = []
    for traj in trajectories:
        td = {
            'type': traj['type'],
            'is_ood': traj['is_ood'],
            'scores': [float(s) for s in traj['scores']],
        }
        if 'step_labels' in traj:
            td['step_labels'] = traj['step_labels']
        traj_data.append(td)

    output = {
        'experiment': 'temporal_autocorr',
        'experiment_number': 59,
        'timestamp': timestamp,
        'n_cal': len(cal_hidden),
        'n_trajectories': len(trajectories),
        'traj_len': traj_len,
        'total_inferences': len(cal_hidden) + len(trajectories) * traj_len,
        'id_autocorr': {'mean': float(np.mean(id_autocorrs)), 'std': float(np.std(id_autocorrs))},
        'ood_autocorr': {'mean': float(np.mean(ood_autocorrs)), 'std': float(np.std(ood_autocorrs))},
        'raw_step_auroc': float(raw_auroc),
        'window_step_auroc': float(window_auroc),
        'traj_mean_auroc': float(roc_auc_score(traj_labels, traj_mean_scores)),
        'transition_jump': {'mean': float(np.mean(jumps)), 'std': float(np.std(jumps))},
        'trajectories': traj_data,
    }
    output_path = os.path.join(RESULTS_DIR, f"temporal_autocorr_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
