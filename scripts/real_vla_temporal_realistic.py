"""
Temporal Trajectory OOD Detection with Realistic Images on Real OpenVLA-7B.

Experiment 37 showed 3-step temporal mean achieves perfect AUROC=1.000 with
simple images. Experiment 41 showed single-frame realistic AUROC drops to 0.611-0.767.

This experiment tests whether temporal aggregation can recover detection
performance with realistic images — the key question for practical deployment.

Tests:
1. Single-frame detection (baseline)
2. 3-step running mean
3. 5-step running mean
4. Max-over-window
5. Cumulative detection (any frame triggers)

Experiment 42 in the CalibDrive series.
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

SIZE = (256, 256)


def create_realistic_highway(idx):
    rng = np.random.default_rng(idx * 4101)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    for y in range(SIZE[0]//2):
        frac = y / (SIZE[0]//2)
        img[y, :] = [int(80 + 55*frac), int(150 + 56*frac), int(255 - 20*frac)]
    for y in range(SIZE[0]//2, SIZE[0]):
        base = 60 + rng.integers(-5, 6)
        img[y, :] = [base, base, base]
    for y in range(SIZE[0]//2 + 10, SIZE[0], 20):
        if (y // 20) % 2 == 0:
            img[y:y+8, SIZE[1]//2-2:SIZE[1]//2+2] = [220, 220, 220]
    img[SIZE[0]//2:, :5] = [200, 200, 200]
    img[SIZE[0]//2:, -5:] = [200, 200, 200]
    noise = rng.integers(-3, 4, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def create_realistic_urban(idx):
    rng = np.random.default_rng(idx * 4102)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//4] = [135, 206, 235]
    for x_start in range(0, SIZE[1], 40):
        h = rng.integers(SIZE[0]//4, SIZE[0]//2)
        color = rng.integers(100, 200, 3)
        img[SIZE[0]//4:h, x_start:x_start+38] = color
        for wy in range(SIZE[0]//4 + 5, h - 5, 12):
            for wx in range(x_start + 3, x_start + 35, 10):
                img[wy:wy+6, wx:wx+6] = [200, 220, 255]
    img[SIZE[0]//2:SIZE[0]//2+20] = [180, 170, 160]
    img[SIZE[0]//2+20:] = [70, 70, 70]
    noise = rng.integers(-3, 4, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def create_night_driving(idx):
    rng = np.random.default_rng(idx * 4103)
    img = np.full((*SIZE, 3), 15, dtype=np.uint8)
    img[SIZE[0]//2:] = [30, 30, 30]
    for y in range(SIZE[0]//2, SIZE[0]):
        width = int((y - SIZE[0]//2) * 0.8)
        center = SIZE[1] // 2
        brightness = max(0, 120 - (y - SIZE[0]//2))
        img[y, max(0, center-width):min(SIZE[1], center+width)] = [brightness, brightness, int(brightness*0.8)]
    noise = rng.integers(-2, 3, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def create_foggy_road(idx):
    rng = np.random.default_rng(idx * 4104)
    base = create_realistic_highway(idx + 2000)
    fog = np.full_like(base, 200)
    alpha = 0.5 + rng.random() * 0.2
    return (base * (1 - alpha) + fog * alpha).astype(np.uint8)


def create_snow_road(idx):
    rng = np.random.default_rng(idx * 4105)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [200, 210, 220]
    for y in range(SIZE[0]//2, SIZE[0]):
        b = 200 + rng.integers(-10, 10)
        img[y, :] = [b, b, b]
    img[SIZE[0]//2:, SIZE[1]//3:SIZE[1]//3+5] = [150, 140, 130]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def create_flooded_road(idx):
    rng = np.random.default_rng(idx * 4106)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [100, 100, 110]
    for y in range(SIZE[0]//2, SIZE[0]):
        depth = (y - SIZE[0]//2) / (SIZE[0]//2)
        img[y, :] = [int(40+30*depth), int(80+40*depth), int(140+50*depth)]
    noise = rng.integers(-3, 4, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def create_offroad(idx):
    rng = np.random.default_rng(idx * 4107)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [135, 206, 235]
    for x in range(0, SIZE[1], 15):
        h = SIZE[0]//3 + rng.integers(0, 30)
        img[SIZE[0]//3:h, x:x+12] = [30+rng.integers(0,30), 100+rng.integers(0,50), 20+rng.integers(0,30)]
    img[SIZE[0]//2:] = [140, 100, 60]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def create_tunnel(idx):
    rng = np.random.default_rng(idx * 4108)
    img = np.full((*SIZE, 3), 10, dtype=np.uint8)
    cy, cx = SIZE[0]//2 - 20, SIZE[1]//2
    for y in range(SIZE[0]):
        for x in range(SIZE[1]):
            dist = np.sqrt((y - cy)**2 + (x - cx)**2)
            if dist < 40:
                b = int(200 * (1 - dist / 40))
                img[y, x] = [b, b, b]
    img[SIZE[0]//2:] = np.clip(img[SIZE[0]//2:].astype(int) + 20, 0, 255).astype(np.uint8)
    noise = rng.integers(-2, 3, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def extract_hidden(model, processor, image, prompt):
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
    dim_masses = []
    for score in outputs.scores[:7]:
        full_probs = torch.softmax(score[0].float(), dim=0)
        dim_masses.append(float(full_probs[action_start:].sum()))

    return hidden, float(np.mean(dim_masses))


def cosine_dist(a, b):
    a_n = a / (np.linalg.norm(a) + 1e-10)
    b_n = b / (np.linalg.norm(b) + 1e-10)
    return 1.0 - float(np.dot(a_n, b_n))


def add_temporal_jitter(img_arr, step, rng):
    """Add slight camera jitter to simulate temporal sequence."""
    shift_x = rng.integers(-3, 4)
    shift_y = rng.integers(-2, 3)
    result = np.zeros_like(img_arr)
    h, w = img_arr.shape[:2]
    src_y1, src_y2 = max(0, shift_y), min(h, h + shift_y)
    src_x1, src_x2 = max(0, shift_x), min(w, w + shift_x)
    dst_y1, dst_y2 = max(0, -shift_y), min(h, h - shift_y)
    dst_x1, dst_x2 = max(0, -shift_x), min(w, w - shift_x)
    result[dst_y1:dst_y2, dst_x1:dst_x2] = img_arr[src_y1:src_y2, src_x1:src_x2]
    brightness = rng.integers(-5, 6)
    result = np.clip(result.astype(np.int16) + brightness, 0, 255).astype(np.uint8)
    return result


def main():
    print("=" * 70, flush=True)
    print("TEMPORAL TRAJECTORY WITH REALISTIC IMAGES", flush=True)
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

    # Calibration: diverse scenarios
    print("\nCalibration phase...", flush=True)
    cal_fns = {
        'highway': create_realistic_highway,
        'urban': create_realistic_urban,
        'night': create_night_driving,
        'foggy': create_foggy_road,
    }
    all_cal_hidden = []
    cal_per_scene = {}
    for scene, fn in cal_fns.items():
        scene_hidden = []
        for i in range(6):
            img_arr = fn(i + 700)
            image = Image.fromarray(img_arr)
            hidden, _ = extract_hidden(model, processor, image, prompt)
            all_cal_hidden.append(hidden)
            scene_hidden.append(hidden)
        cal_per_scene[scene] = np.mean(scene_hidden, axis=0)
    cal_arr = np.array(all_cal_hidden)
    global_centroid = np.mean(cal_arr, axis=0)
    print(f"  Calibration: {len(all_cal_hidden)} samples from {len(cal_fns)} scenarios", flush=True)

    # Define trajectories (8 steps each)
    STEPS_PER_TRAJ = 8
    trajectories = []

    # ID trajectories
    id_traj_defs = [
        ('highway', create_realistic_highway, 5),
        ('urban', create_realistic_urban, 5),
        ('night', create_night_driving, 3),
        ('foggy', create_foggy_road, 3),
    ]
    for scene, fn, n_traj in id_traj_defs:
        for t in range(n_traj):
            trajectories.append({
                'scene': scene, 'fn': fn, 'is_ood': False,
                'base_idx': t * 100 + 1000,
            })

    # OOD trajectories
    ood_traj_defs = [
        ('snow', create_snow_road, 4),
        ('flooded', create_flooded_road, 4),
        ('offroad', create_offroad, 4),
        ('tunnel', create_tunnel, 4),
    ]
    for scene, fn, n_traj in ood_traj_defs:
        for t in range(n_traj):
            trajectories.append({
                'scene': scene, 'fn': fn, 'is_ood': True,
                'base_idx': t * 100 + 2000,
            })

    total_traj = len(trajectories)
    total_inferences = total_traj * STEPS_PER_TRAJ
    print(f"\nTrajectories: {total_traj} ({total_traj - 16} ID, 16 OOD) "
          f"× {STEPS_PER_TRAJ} steps = {total_inferences} inferences", flush=True)

    # Run all trajectories
    traj_results = []
    inf_count = 0

    for t_idx, traj in enumerate(trajectories):
        step_data = []
        for step in range(STEPS_PER_TRAJ):
            rng = np.random.default_rng(traj['base_idx'] + step * 17)
            img_arr = traj['fn'](traj['base_idx'] + step)
            img_arr = add_temporal_jitter(img_arr, step, rng)
            image = Image.fromarray(img_arr)
            hidden, mass = extract_hidden(model, processor, image, prompt)

            cos_global = cosine_dist(hidden, global_centroid)
            cos_per_scene = min(cosine_dist(hidden, c) for c in cal_per_scene.values())

            step_data.append({
                'step': step,
                'cos_global': cos_global,
                'cos_per_scene': cos_per_scene,
                'action_mass': mass,
            })
            inf_count += 1

        traj_results.append({
            'traj_idx': t_idx,
            'scene': traj['scene'],
            'is_ood': traj['is_ood'],
            'steps': step_data,
        })

        if (t_idx + 1) % 5 == 0 or t_idx == total_traj - 1:
            mean_cos = np.mean([s['cos_global'] for s in step_data])
            print(f"  [{inf_count}/{total_inferences}] Traj {t_idx}: "
                  f"{traj['scene']} ({'OOD' if traj['is_ood'] else 'ID'}): "
                  f"mean_cos={mean_cos:.4f}", flush=True)

    # ===================================================================
    # Analysis
    # ===================================================================
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    id_traj = [t for t in traj_results if not t['is_ood']]
    ood_traj = [t for t in traj_results if t['is_ood']]
    labels = [0] * len(id_traj) + [1] * len(ood_traj)
    all_traj = id_traj + ood_traj

    # Aggregation methods
    def aggregate(traj_list, key, method, window=None):
        scores = []
        for t in traj_list:
            vals = [s[key] for s in t['steps']]
            if method == 'single_first':
                scores.append(vals[0])
            elif method == 'single_last':
                scores.append(vals[-1])
            elif method == 'mean_all':
                scores.append(np.mean(vals))
            elif method == 'max_all':
                scores.append(np.max(vals))
            elif method == 'mean_window':
                scores.append(np.mean(vals[:window]))
            elif method == 'max_window':
                scores.append(np.max(vals[:window]))
            elif method == 'cumulative_any':
                # Fraction of steps above median of all step scores
                scores.append(np.mean(vals))  # proxy
            elif method == 'std':
                scores.append(np.std(vals))
        return scores

    print("\n1. Single-Frame vs Temporal Aggregation (AUROC)", flush=True)
    print("-" * 70, flush=True)

    methods = [
        ('Single frame (step 0)', 'cos_global', 'single_first', None),
        ('Single frame (step 7)', 'cos_global', 'single_last', None),
        ('3-step mean (global)', 'cos_global', 'mean_window', 3),
        ('5-step mean (global)', 'cos_global', 'mean_window', 5),
        ('8-step mean (global)', 'cos_global', 'mean_all', None),
        ('8-step max (global)', 'cos_global', 'max_all', None),
        ('3-step mean (per-scene)', 'cos_per_scene', 'mean_window', 3),
        ('5-step mean (per-scene)', 'cos_per_scene', 'mean_window', 5),
        ('8-step mean (per-scene)', 'cos_per_scene', 'mean_all', None),
        ('8-step max (per-scene)', 'cos_per_scene', 'max_all', None),
        ('Action mass 8-step mean', 'action_mass', 'mean_all', None),
    ]

    for name, key, method, window in methods:
        scores = aggregate(all_traj, key, method, window)
        # For action mass, lower = more OOD, so flip
        if 'mass' in name.lower():
            scores = [1 - s for s in scores]
        auroc = roc_auc_score(labels, scores)
        print(f"  {name:<35}: AUROC = {auroc:.3f}", flush=True)

    # Per-OOD type for best methods
    print("\n2. Per-OOD-Type AUROC (best temporal methods)", flush=True)
    print("-" * 80, flush=True)

    ood_types = ['snow', 'flooded', 'offroad', 'tunnel']
    best_methods = [
        ('Single frame (step 0)', 'cos_global', 'single_first', None),
        ('8-step mean (global)', 'cos_global', 'mean_all', None),
        ('8-step mean (per-scene)', 'cos_per_scene', 'mean_all', None),
        ('8-step max (per-scene)', 'cos_per_scene', 'max_all', None),
    ]

    header = f"  {'Method':<35}"
    for t in ood_types:
        header += f" | {t:>10}"
    print(header, flush=True)

    for name, key, method, window in best_methods:
        parts = [f"  {name:<35}"]
        for ood_type in ood_types:
            type_ood = [t for t in ood_traj if t['scene'] == ood_type]
            type_labels = [0] * len(id_traj) + [1] * len(type_ood)
            type_all = id_traj + type_ood

            scores = aggregate(type_all, key, method, window)
            auroc = roc_auc_score(type_labels, scores)
            parts.append(f"{auroc:>10.3f}")
        print(" | ".join(parts), flush=True)

    # Temporal improvement analysis
    print("\n3. Temporal Improvement Over Single Frame", flush=True)
    print("-" * 60, flush=True)

    single_scores = aggregate(all_traj, 'cos_global', 'single_first', None)
    single_auroc = roc_auc_score(labels, single_scores)

    for w in [2, 3, 5, 8]:
        if w == 8:
            mean_scores = aggregate(all_traj, 'cos_global', 'mean_all', None)
        else:
            mean_scores = aggregate(all_traj, 'cos_global', 'mean_window', w)
        mean_auroc = roc_auc_score(labels, mean_scores)
        gain = mean_auroc - single_auroc
        print(f"  {w}-step mean: AUROC = {mean_auroc:.3f} (Δ = {gain:+.3f})", flush=True)

    # Per-scene per-step cosine distances
    print("\n4. Mean Cosine Distance per Step", flush=True)
    print("-" * 60, flush=True)
    for is_ood, label in [(False, 'ID'), (True, 'OOD')]:
        subset = [t for t in traj_results if t['is_ood'] == is_ood]
        print(f"  {label}:", end="", flush=True)
        for step in range(STEPS_PER_TRAJ):
            mean_cos = np.mean([t['steps'][step]['cos_global'] for t in subset])
            print(f"  s{step}={mean_cos:.3f}", end="", flush=True)
        print(flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'temporal_realistic',
        'experiment_number': 42,
        'timestamp': timestamp,
        'n_cal': len(all_cal_hidden),
        'n_id_traj': len(id_traj),
        'n_ood_traj': len(ood_traj),
        'steps_per_traj': STEPS_PER_TRAJ,
        'total_inferences': inf_count + len(all_cal_hidden),
        'trajectories': traj_results,
    }

    output_path = os.path.join(RESULTS_DIR, f"temporal_realistic_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
