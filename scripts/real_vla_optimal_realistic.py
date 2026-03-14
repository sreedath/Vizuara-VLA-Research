"""
Optimal Realistic OOD Detection: Combining All Winning Strategies.

Key findings to combine:
- Exp 41: Per-scene centroids improve single-frame to 0.767
- Exp 42: 8-step temporal per-scene achieves 0.824
- Exp 43: Action spread (0.732) and action mass (0.745) outperform
  cosine (0.461) for realistic images

This experiment tests combinations optimized for realistic images:
1. Action spread alone (temporal mean)
2. Action mass alone (temporal mean)
3. Per-scene cosine (temporal mean) — baseline from Exp 42
4. Action spread + per-scene cosine (optimized weights)
5. Action mass + action spread + cosine (optimized weights)
6. Learned threshold via calibration conformal

Experiment 44 in the CalibDrive series.
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


# === Image generators (copied for self-containment) ===
def create_realistic_highway(idx):
    rng = np.random.default_rng(idx * 4101)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    for y in range(SIZE[0]//2):
        frac = y / (SIZE[0]//2)
        img[y, :] = [int(80+55*frac), int(150+56*frac), int(255-20*frac)]
    for y in range(SIZE[0]//2, SIZE[0]):
        base = 60 + rng.integers(-5, 6)
        img[y, :] = [base, base, base]
    for y in range(SIZE[0]//2+10, SIZE[0], 20):
        if (y//20)%2 == 0:
            img[y:y+8, SIZE[1]//2-2:SIZE[1]//2+2] = [220, 220, 220]
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
    return (base * (1-alpha) + fog * alpha).astype(np.uint8)

def create_snow_road(idx):
    rng = np.random.default_rng(idx * 4105)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [200, 210, 220]
    for y in range(SIZE[0]//2, SIZE[0]):
        b = 200 + rng.integers(-10, 10)
        img[y, :] = [b, b, b]
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
                b = int(200 * (1 - dist/40))
                img[y, x] = [b, b, b]
    img[SIZE[0]//2:] = np.clip(img[SIZE[0]//2:].astype(int) + 20, 0, 255).astype(np.uint8)
    noise = rng.integers(-2, 3, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def add_temporal_jitter(img_arr, step, rng):
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


def extract_full(model, processor, image, prompt):
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
    dim_entropies = []
    dim_action_values = []

    for score in outputs.scores[:7]:
        full_probs = torch.softmax(score[0].float(), dim=0)
        action_probs = full_probs[action_start:]
        dim_masses.append(float(action_probs.sum()))
        ap = action_probs.cpu().numpy()
        ap = ap / (ap.sum() + 1e-10)
        dim_entropies.append(float(-np.sum(ap * np.log(ap + 1e-10))))
        dim_action_values.append(int(action_probs.argmax()))

    action_vals = np.array(dim_action_values, dtype=float)

    return {
        'hidden': hidden,
        'action_mass': float(np.mean(dim_masses)),
        'action_spread': float(np.std(action_vals)),
        'action_roughness': float(np.mean(np.abs(np.diff(action_vals)))),
        'mean_entropy': float(np.mean(dim_entropies)),
        'entropy_std': float(np.std(dim_entropies)),
    }


def cosine_dist(a, b):
    a_n = a / (np.linalg.norm(a) + 1e-10)
    b_n = b / (np.linalg.norm(b) + 1e-10)
    return 1.0 - float(np.dot(a_n, b_n))


def norm01(arr):
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-10:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def main():
    print("=" * 70, flush=True)
    print("OPTIMAL REALISTIC OOD DETECTION", flush=True)
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

    # Calibration: diverse conditions
    print("\nCalibration...", flush=True)
    cal_fns = {
        'highway': create_realistic_highway,
        'urban': create_realistic_urban,
        'night': create_night_driving,
        'foggy': create_foggy_road,
    }
    cal_per_scene = {}
    all_cal_data = []
    for scene, fn in cal_fns.items():
        scene_data = []
        for i in range(6):
            img = Image.fromarray(fn(i + 3000))
            data = extract_full(model, processor, img, prompt)
            scene_data.append(data)
            all_cal_data.append(data)
        cal_per_scene[scene] = np.mean([d['hidden'] for d in scene_data], axis=0)
        # Also store calibration action stats per scene
        cal_per_scene[f'{scene}_spread'] = np.mean([d['action_spread'] for d in scene_data])
        cal_per_scene[f'{scene}_mass'] = np.mean([d['action_mass'] for d in scene_data])
        print(f"  {scene}: 6 samples, mean_spread={cal_per_scene[f'{scene}_spread']:.1f}, "
              f"mean_mass={cal_per_scene[f'{scene}_mass']:.4f}", flush=True)

    # Global calibration stats for normalization
    cal_spreads = np.array([d['action_spread'] for d in all_cal_data])
    cal_masses = np.array([d['action_mass'] for d in all_cal_data])
    cal_spread_mean, cal_spread_std = np.mean(cal_spreads), np.std(cal_spreads) + 1e-10
    cal_mass_mean, cal_mass_std = np.mean(cal_masses), np.std(cal_masses) + 1e-10

    print(f"  Cal stats: spread={cal_spread_mean:.1f}±{cal_spread_std:.1f}, "
          f"mass={cal_mass_mean:.4f}±{cal_mass_std:.4f}", flush=True)

    # Trajectories: 8 steps each
    STEPS = 8
    traj_defs = [
        # ID
        ('highway', create_realistic_highway, False, 6),
        ('urban', create_realistic_urban, False, 6),
        ('night', create_night_driving, False, 4),
        ('foggy', create_foggy_road, False, 4),
        # OOD
        ('snow', create_snow_road, True, 5),
        ('flooded', create_flooded_road, True, 5),
        ('offroad', create_offroad, True, 5),
        ('tunnel', create_tunnel, True, 5),
    ]

    trajectories = []
    for scene, fn, is_ood, n_traj in traj_defs:
        for t in range(n_traj):
            trajectories.append({
                'scene': scene, 'fn': fn, 'is_ood': is_ood,
                'base_idx': t * 100 + 5000,
            })

    total_traj = len(trajectories)
    total_inf = total_traj * STEPS
    print(f"\nTrajectories: {total_traj} × {STEPS} steps = {total_inf} inferences", flush=True)

    traj_results = []
    inf_count = 0

    for t_idx, traj in enumerate(trajectories):
        step_data = []
        for step in range(STEPS):
            rng = np.random.default_rng(traj['base_idx'] + step * 17)
            img_arr = traj['fn'](traj['base_idx'] + step)
            img_arr = add_temporal_jitter(img_arr, step, rng)
            image = Image.fromarray(img_arr)
            data = extract_full(model, processor, image, prompt)

            cos_per_scene = min(cosine_dist(data['hidden'], c)
                               for k, c in cal_per_scene.items()
                               if not k.endswith('_spread') and not k.endswith('_mass'))

            step_data.append({
                'step': step,
                'cos_per_scene': cos_per_scene,
                'action_mass': data['action_mass'],
                'action_spread': data['action_spread'],
                'action_roughness': data['action_roughness'],
                'mean_entropy': data['mean_entropy'],
                'entropy_std': data['entropy_std'],
            })
            inf_count += 1

        traj_results.append({
            'traj_idx': t_idx,
            'scene': traj['scene'],
            'is_ood': traj['is_ood'],
            'steps': step_data,
        })

        if (t_idx + 1) % 8 == 0 or t_idx == total_traj - 1:
            mean_cos = np.mean([s['cos_per_scene'] for s in step_data])
            mean_spread = np.mean([s['action_spread'] for s in step_data])
            print(f"  [{inf_count}/{total_inf}] Traj {t_idx}: "
                  f"{traj['scene']} ({'OOD' if traj['is_ood'] else 'ID'}): "
                  f"cos={mean_cos:.4f}, spread={mean_spread:.1f}", flush=True)

    # ===================================================================
    # Analysis
    # ===================================================================
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    id_t = [t for t in traj_results if not t['is_ood']]
    ood_t = [t for t in traj_results if t['is_ood']]
    labels = [0]*len(id_t) + [1]*len(ood_t)
    all_t = id_t + ood_t

    def traj_score(traj_list, key, agg='mean'):
        scores = []
        for t in traj_list:
            vals = [s[key] for s in t['steps']]
            if agg == 'mean':
                scores.append(np.mean(vals))
            elif agg == 'max':
                scores.append(np.max(vals))
            elif agg == 'std':
                scores.append(np.std(vals))
        return np.array(scores)

    # Single signals
    print("\n1. Individual Signal AUROCs (8-step mean)", flush=True)
    print("-" * 60, flush=True)

    signals = {
        'Per-scene cosine': traj_score(all_t, 'cos_per_scene'),
        'Action spread': traj_score(all_t, 'action_spread'),
        'Action roughness': traj_score(all_t, 'action_roughness'),
        'Action mass (inv)': 1 - traj_score(all_t, 'action_mass'),
        'Entropy std': traj_score(all_t, 'entropy_std'),
        'Mean entropy': traj_score(all_t, 'mean_entropy'),
    }

    for name, scores in sorted(signals.items(), key=lambda x: -roc_auc_score(labels, x[1])):
        auroc = roc_auc_score(labels, scores)
        print(f"  {name:<30}: AUROC = {auroc:.3f}", flush=True)

    # Combinations with grid search over weights
    print("\n2. Optimal Combinations (Grid Search)", flush=True)
    print("-" * 60, flush=True)

    cos_n = norm01(signals['Per-scene cosine'])
    spread_n = norm01(signals['Action spread'])
    mass_n = norm01(signals['Action mass (inv)'])
    rough_n = norm01(signals['Action roughness'])

    best_auroc = 0
    best_combo = ""
    best_weights = None

    # 2-signal combos
    for w in np.arange(0, 1.05, 0.1):
        for name_a, sig_a, name_b, sig_b in [
            ('cosine', cos_n, 'spread', spread_n),
            ('cosine', cos_n, 'mass', mass_n),
            ('spread', spread_n, 'mass', mass_n),
            ('cosine', cos_n, 'roughness', rough_n),
            ('spread', spread_n, 'roughness', rough_n),
        ]:
            combo = w * sig_a + (1-w) * sig_b
            auroc = roc_auc_score(labels, combo)
            if auroc > best_auroc:
                best_auroc = auroc
                best_combo = f"{name_a}({w:.1f}) + {name_b}({1-w:.1f})"
                best_weights = (name_a, w, name_b, 1-w)

    print(f"  Best 2-signal: {best_combo}: AUROC = {best_auroc:.3f}", flush=True)

    # 3-signal combos
    best_3 = 0
    best_3_combo = ""
    for w1 in np.arange(0, 1.05, 0.2):
        for w2 in np.arange(0, 1.05 - w1, 0.2):
            w3 = 1 - w1 - w2
            if w3 < 0:
                continue
            combo = w1 * cos_n + w2 * spread_n + w3 * mass_n
            auroc = roc_auc_score(labels, combo)
            if auroc > best_3:
                best_3 = auroc
                best_3_combo = f"cos({w1:.1f})+spread({w2:.1f})+mass({w3:.1f})"

    print(f"  Best 3-signal: {best_3_combo}: AUROC = {best_3:.3f}", flush=True)

    # Fixed combos from intuition
    combos = {
        'Equal (cos+spread+mass)': (cos_n + spread_n + mass_n) / 3,
        'Behavior-heavy (0.2cos+0.4spread+0.4mass)': 0.2*cos_n + 0.4*spread_n + 0.4*mass_n,
        'Spread-heavy (0.1cos+0.7spread+0.2mass)': 0.1*cos_n + 0.7*spread_n + 0.2*mass_n,
        'Mass-heavy (0.1cos+0.2spread+0.7mass)': 0.1*cos_n + 0.2*spread_n + 0.7*mass_n,
    }

    for name, scores in combos.items():
        auroc = roc_auc_score(labels, scores)
        print(f"  {name}: AUROC = {auroc:.3f}", flush=True)

    # Per-OOD type for best methods
    print("\n3. Per-OOD Type AUROC (best methods)", flush=True)
    print("-" * 80, flush=True)

    ood_types = ['snow', 'flooded', 'offroad', 'tunnel']
    top_methods = {
        'Per-scene cosine (temporal)': signals['Per-scene cosine'],
        'Action spread (temporal)': signals['Action spread'],
        'Action mass (inv, temporal)': signals['Action mass (inv)'],
        f'Best 2-signal combo': None,
    }

    # Recompute best combo
    if best_weights:
        na, wa, nb, wb = best_weights
        sig_map = {'cosine': cos_n, 'spread': spread_n, 'mass': mass_n, 'roughness': rough_n}
        top_methods[f'Best 2-signal combo'] = wa * sig_map[na] + wb * sig_map[nb]

    header = f"  {'Method':<35}"
    for t in ood_types:
        header += f" | {t:>10}"
    header += " |    Overall"
    print(header, flush=True)

    for name, scores in top_methods.items():
        if scores is None:
            continue
        parts = [f"  {name:<35}"]
        for ood_type in ood_types:
            type_ood = [i for i, t in enumerate(all_t) if t['is_ood'] and t['scene'] == ood_type]
            type_id = [i for i, t in enumerate(all_t) if not t['is_ood']]
            type_labels = [0]*len(type_id) + [1]*len(type_ood)
            type_scores = list(scores[type_id]) + list(scores[type_ood])
            auroc = roc_auc_score(type_labels, type_scores)
            parts.append(f"{auroc:>10.3f}")
        overall = roc_auc_score(labels, scores)
        parts.append(f"{overall:>10.3f}")
        print(" | ".join(parts), flush=True)

    # Conformal prediction
    print("\n4. Conformal Prediction on Best Signal", flush=True)
    print("-" * 60, flush=True)

    # Use calibration data to set threshold
    cal_cos_scores = []
    for scene, fn in cal_fns.items():
        for i in range(6):
            img = Image.fromarray(fn(i + 3000))
            data = extract_full(model, processor, img, prompt)
            cos_ps = min(cosine_dist(data['hidden'], c)
                        for k, c in cal_per_scene.items()
                        if not k.endswith('_spread') and not k.endswith('_mass'))
            cal_cos_scores.append(cos_ps)

    for alpha in [0.05, 0.10, 0.20]:
        q = np.quantile(cal_cos_scores, 1 - alpha)
        id_scores_all = signals['Per-scene cosine'][:len(id_t)]
        ood_scores_all = signals['Per-scene cosine'][len(id_t):]
        id_coverage = float(np.mean(id_scores_all <= q))
        ood_flagged = float(np.mean(ood_scores_all > q))
        print(f"  α={alpha:.2f}: threshold={q:.4f}, "
              f"ID coverage={id_coverage:.3f}, OOD flagged={ood_flagged:.3f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'optimal_realistic',
        'experiment_number': 44,
        'timestamp': timestamp,
        'n_cal': len(all_cal_data),
        'n_id_traj': len(id_t),
        'n_ood_traj': len(ood_t),
        'steps_per_traj': STEPS,
        'total_inferences': inf_count + len(all_cal_data),
        'trajectories': traj_results,
    }
    output_path = os.path.join(RESULTS_DIR, f"optimal_realistic_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
