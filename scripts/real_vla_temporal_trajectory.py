"""
Temporal Trajectory Analysis with Cosine Distance on Real OpenVLA-7B.

Tests how cosine distance behaves across multi-step driving sequences
by simulating temporal progression (same scene with slight variations).

Questions:
1. Is cosine distance stable across steps for in-distribution scenes?
2. Does it detect OOD immediately at step 1 or does it need multiple steps?
3. Do cumulative trajectory statistics (mean, max, variance) improve detection?
4. How does early stopping (flag at first detection) compare to full trajectory?

Experiment 37 in the CalibDrive series.
"""
import os
import json
import time
import datetime
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

RESULTS_DIR = "/workspace/Vizuara-VLA-Research/experiments"
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAJECTORY_LENGTH = 8  # steps per trajectory
N_TRAJECTORIES = {
    'highway': 8,
    'urban': 8,
    'ood_noise': 5,
    'ood_blank': 5,
    'ood_indoor': 5,
    'ood_inverted': 5,
    'ood_blackout': 5,
}


def create_trajectory_frame(scenario, traj_idx, step, size=(256, 256)):
    """Create a frame for step `step` of trajectory `traj_idx` in `scenario`.
    For driving scenes, we add minor temporal variation (simulating camera jitter).
    For OOD, each step is essentially the same abnormal input."""
    np.random.seed(traj_idx * 37000 + step * 370 + hash(scenario) % 3700)

    if scenario == 'highway':
        img = np.zeros((*size, 3), dtype=np.uint8)
        img[:size[0]//2] = [135, 206, 235]
        img[size[0]//2:] = [80, 80, 80]
        # Temporal variation: slight color shift + noise
        shift = np.random.randint(-5, 6, 3)
        img[:size[0]//2] = np.clip(img[:size[0]//2].astype(int) + shift, 0, 255).astype(np.uint8)
    elif scenario == 'urban':
        img = np.zeros((*size, 3), dtype=np.uint8)
        img[:size[0]//3] = [135, 206, 235]
        img[size[0]//3:size[0]//2] = [139, 119, 101]
        img[size[0]//2:] = [80, 80, 80]
        shift = np.random.randint(-5, 6, 3)
        img[:size[0]//3] = np.clip(img[:size[0]//3].astype(int) + shift, 0, 255).astype(np.uint8)
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
    elif scenario == 'ood_blackout':
        img = np.full((*size, 3), 5, dtype=np.uint8)
    else:
        img = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)

    noise = np.random.randint(-3, 3, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def main():
    print("=" * 70, flush=True)
    print("TEMPORAL TRAJECTORY ANALYSIS", flush=True)
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

    prompt = "In: What action should the robot take to drive forward at 25 m/s safely?\nOut:"

    # Phase 1: Calibration
    print("\nPhase 1: Calibration centroid...", flush=True)
    cal_hidden = []
    for scene in ['highway', 'urban']:
        for i in range(15):
            np.random.seed(i * 37100 + hash(scene) % 37100)
            if scene == 'highway':
                img = np.zeros((256, 256, 3), dtype=np.uint8)
                img[:128] = [135, 206, 235]
                img[128:] = [80, 80, 80]
            else:
                img = np.zeros((256, 256, 3), dtype=np.uint8)
                img[:85] = [135, 206, 235]
                img[85:128] = [139, 119, 101]
                img[128:] = [80, 80, 80]
            noise = np.random.randint(-2, 3, img.shape, dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            image = Image.fromarray(img)
            inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=7, do_sample=False,
                    output_hidden_states=True, return_dict_in_generate=True,
                )
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                last_step = outputs.hidden_states[-1]
                if isinstance(last_step, tuple):
                    hidden = last_step[-1][0, -1, :].float().cpu().numpy()
                else:
                    hidden = last_step[0, -1, :].float().cpu().numpy()
            else:
                hidden = np.zeros(4096)
            cal_hidden.append(hidden)

    cal_arr = np.array(cal_hidden)
    cal_mean = np.mean(cal_arr, axis=0)
    cal_norm = cal_mean / (np.linalg.norm(cal_mean) + 1e-10)

    # Conformal threshold
    cal_cos = sorted([1.0 - float(np.dot(h / (np.linalg.norm(h) + 1e-10), cal_norm))
                       for h in cal_hidden])
    alpha = 0.10
    q_idx = min(int(np.ceil((1 - alpha) * (len(cal_cos) + 1))) - 1, len(cal_cos) - 1)
    threshold = cal_cos[q_idx]
    print(f"  Centroid from {len(cal_hidden)} samples. Threshold: {threshold:.4f}", flush=True)

    # Phase 2: Run trajectories
    total_steps = sum(n * TRAJECTORY_LENGTH for n in N_TRAJECTORIES.values())
    print(f"\nPhase 2: Running {sum(N_TRAJECTORIES.values())} trajectories "
          f"({total_steps} total steps)...", flush=True)

    trajectories = []
    step_count = 0

    for scenario, n_traj in N_TRAJECTORIES.items():
        is_ood = scenario.startswith('ood_')
        for traj_idx in range(n_traj):
            traj_data = {
                'scenario': scenario,
                'is_ood': is_ood,
                'traj_idx': traj_idx,
                'steps': [],
            }

            for step in range(TRAJECTORY_LENGTH):
                step_count += 1
                image = create_trajectory_frame(scenario, traj_idx, step)
                inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, max_new_tokens=7, do_sample=False,
                        output_scores=True, output_hidden_states=True,
                        return_dict_in_generate=True,
                    )

                if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                    last_step_h = outputs.hidden_states[-1]
                    if isinstance(last_step_h, tuple):
                        hidden = last_step_h[-1][0, -1, :].float().cpu().numpy()
                    else:
                        hidden = last_step_h[0, -1, :].float().cpu().numpy()
                else:
                    hidden = np.zeros(4096)

                h_norm = hidden / (np.linalg.norm(hidden) + 1e-10)
                cos_dist = 1.0 - float(np.dot(h_norm, cal_norm))

                # Action mass
                vocab_size = outputs.scores[0].shape[-1]
                action_start = vocab_size - 256
                dim_masses = []
                for score in outputs.scores[:7]:
                    full_logits = score[0].float()
                    full_probs = torch.softmax(full_logits, dim=0).cpu().numpy()
                    action_probs = full_probs[action_start:]
                    dim_masses.append(float(action_probs.sum()))

                traj_data['steps'].append({
                    'step': step,
                    'cos_dist': cos_dist,
                    'action_mass': float(np.mean(dim_masses)),
                    'flagged': cos_dist > threshold,
                })

            trajectories.append(traj_data)

            if step_count % 40 == 0 or step_count == total_steps:
                mean_cos = np.mean([s['cos_dist'] for s in traj_data['steps']])
                print(f"  [{step_count}/{total_steps}] {scenario}_{traj_idx}: "
                      f"mean_cos={mean_cos:.4f}", flush=True)

    # ===================================================================
    # Analysis
    # ===================================================================
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    # 1. Per-step cosine distance stability
    print("\n1. Mean Cosine Distance by Step", flush=True)
    print("-" * 80, flush=True)
    scenarios_list = list(N_TRAJECTORIES.keys())
    header_s = "Step"
    print(f"  {header_s:>6}", end="", flush=True)
    for s in scenarios_list:
        label = s.replace('ood_', '')[:8]
        print(f" | {label:>10}", end="", flush=True)
    print("", flush=True)

    for step in range(TRAJECTORY_LENGTH):
        print(f"  {step:>6}", end="", flush=True)
        for scenario in scenarios_list:
            vals = [t['steps'][step]['cos_dist']
                    for t in trajectories if t['scenario'] == scenario]
            print(f" | {np.mean(vals):>10.4f}", end="", flush=True)
        print("", flush=True)

    # 2. Trajectory-level statistics
    print("\n2. Trajectory-Level Aggregation AUROC", flush=True)
    print("-" * 80, flush=True)

    easy_trajs = [t for t in trajectories if not t['is_ood']]
    ood_trajs = [t for t in trajectories if t['is_ood']]

    labels = [0] * len(easy_trajs) + [1] * len(ood_trajs)
    all_trajs = easy_trajs + ood_trajs

    agg_methods = {
        'Mean cos dist': lambda t: np.mean([s['cos_dist'] for s in t['steps']]),
        'Max cos dist': lambda t: np.max([s['cos_dist'] for s in t['steps']]),
        'Min cos dist': lambda t: np.min([s['cos_dist'] for s in t['steps']]),
        'Std cos dist': lambda t: np.std([s['cos_dist'] for s in t['steps']]),
        'Step 0 only': lambda t: t['steps'][0]['cos_dist'],
        'Step 0-1 mean': lambda t: np.mean([t['steps'][0]['cos_dist'],
                                             t['steps'][1]['cos_dist']]),
        'Cumulative prod': lambda t: 1.0 - np.prod([1.0 - s['cos_dist']
                                                     for s in t['steps']]),
        'Any flagged': lambda t: float(any(s['flagged'] for s in t['steps'])),
        'Flag fraction': lambda t: np.mean([float(s['flagged']) for s in t['steps']]),
        'Mean mass': lambda t: np.mean([s['action_mass'] for s in t['steps']]),
    }

    print(f"  {'Aggregation':<20} | {'AUROC':>8}", flush=True)
    print("  " + "-" * 35, flush=True)

    for name, fn in agg_methods.items():
        scores = [fn(t) for t in all_trajs]
        if name == 'Mean mass':
            # Lower mass = more OOD, so negate
            scores = [-s for s in scores]
        try:
            auroc = roc_auc_score(labels, scores)
        except ValueError:
            auroc = 0.5
        print(f"  {name:<20} | {auroc:>8.3f}", flush=True)

    # 3. Early stopping: at which step is OOD detectable?
    print("\n3. Cumulative AUROC by Number of Steps Used", flush=True)
    print("-" * 80, flush=True)
    for n_steps in range(1, TRAJECTORY_LENGTH + 1):
        scores = [np.mean([t['steps'][s]['cos_dist']
                          for s in range(n_steps)]) for t in all_trajs]
        auroc = roc_auc_score(labels, scores)
        # Flag rate
        ood_scores = scores[len(easy_trajs):]
        easy_scores = scores[:len(easy_trajs)]
        # Using threshold from calibration
        ood_flagged = sum(1 for s in ood_scores if s > threshold) / len(ood_scores)
        easy_flagged = sum(1 for s in easy_scores if s > threshold) / len(easy_scores)
        print(f"  Steps 0-{n_steps-1} (mean): AUROC={auroc:.3f}, "
              f"OOD_flag={ood_flagged:.1%}, easy_flag={easy_flagged:.1%}", flush=True)

    # 4. Temporal consistency: cosine distance variance within trajectories
    print("\n4. Within-Trajectory Cosine Distance Variance", flush=True)
    print("-" * 80, flush=True)
    for scenario in scenarios_list:
        s_trajs = [t for t in trajectories if t['scenario'] == scenario]
        variances = [np.std([s['cos_dist'] for s in t['steps']]) for t in s_trajs]
        print(f"  {scenario:<15}: std={np.mean(variances):.4f} ± {np.std(variances):.4f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'temporal_trajectory',
        'experiment_number': 37,
        'timestamp': timestamp,
        'trajectory_length': TRAJECTORY_LENGTH,
        'n_trajectories': N_TRAJECTORIES,
        'threshold': threshold,
        'trajectories': [{
            'scenario': t['scenario'],
            'is_ood': t['is_ood'],
            'traj_idx': t['traj_idx'],
            'steps': t['steps'],
        } for t in trajectories],
    }

    output_path = os.path.join(RESULTS_DIR, f"temporal_trajectory_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
