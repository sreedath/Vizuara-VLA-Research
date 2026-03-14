"""
Calibration Robustness Study on Real OpenVLA-7B.

Tests how robust the OOD detection performance is to:
1. Different calibration set sizes (4, 8, 16, 24, 32 samples)
2. Different calibration compositions (highway-only, urban-only, mixed)
3. Calibration from one condition, test on another (transfer)
4. Random subsampling stability (5 random seeds per size)

Uses the optimal 0.7*cosine + 0.3*mass combination from Exp 44.

Experiment 45 in the CalibDrive series.
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
    base = create_realistic_highway(idx + 2000)
    rng = np.random.default_rng(idx * 4104)
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


def extract_features(model, processor, image, prompt):
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
    for score in outputs.scores[:7]:
        full_probs = torch.softmax(score[0].float(), dim=0)
        masses.append(float(full_probs[action_start:].sum()))

    return hidden, float(np.mean(masses))


def cosine_dist(a, b):
    return 1.0 - float(np.dot(a / (np.linalg.norm(a) + 1e-10),
                               b / (np.linalg.norm(b) + 1e-10)))


def main():
    print("=" * 70, flush=True)
    print("CALIBRATION ROBUSTNESS STUDY", flush=True)
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

    # Step 1: Generate a large calibration pool (40 samples, 10 per condition)
    print("\nGenerating calibration pool...", flush=True)
    cal_fns = {
        'highway': create_realistic_highway,
        'urban': create_realistic_urban,
        'night': create_night_driving,
        'foggy': create_foggy_road,
    }
    cal_pool = {}  # scene -> list of (hidden, mass) tuples
    for scene, fn in cal_fns.items():
        cal_pool[scene] = []
        for i in range(10):
            img = Image.fromarray(fn(i + 6000))
            hidden, mass = extract_features(model, processor, img, prompt)
            cal_pool[scene].append({'hidden': hidden, 'mass': mass, 'idx': i})
        print(f"  {scene}: {len(cal_pool[scene])} samples", flush=True)

    # Step 2: Generate fixed test set
    print("\nGenerating test set...", flush=True)
    test_data = []
    test_fns = {
        'highway_r': (create_realistic_highway, False, 8),
        'urban_r': (create_realistic_urban, False, 8),
        'night_r': (create_night_driving, False, 6),
        'foggy_r': (create_foggy_road, False, 6),
        'snow': (create_snow_road, True, 6),
        'flooded': (create_flooded_road, True, 6),
        'offroad': (create_offroad, True, 6),
        'tunnel': (create_tunnel, True, 6),
    }
    for scene, (fn, is_ood, n) in test_fns.items():
        for i in range(n):
            img = Image.fromarray(fn(i + 7000))
            hidden, mass = extract_features(model, processor, img, prompt)
            test_data.append({
                'hidden': hidden, 'mass': mass,
                'scenario': scene, 'is_ood': is_ood,
            })
    print(f"  Test set: {len(test_data)} ({sum(1 for t in test_data if not t['is_ood'])} ID, "
          f"{sum(1 for t in test_data if t['is_ood'])} OOD)", flush=True)

    easy = [t for t in test_data if not t['is_ood']]
    ood = [t for t in test_data if t['is_ood']]
    labels = [0]*len(easy) + [1]*len(ood)
    all_test = easy + ood

    def compute_auroc(cal_hidden_list, cal_mass_list, use_per_scene_centroids=None):
        """Compute AUROC using optimal 0.7*cosine + 0.3*mass combo."""
        global_centroid = np.mean(cal_hidden_list, axis=0)

        if use_per_scene_centroids:
            scene_centroids = {}
            for scene_name, indices in use_per_scene_centroids.items():
                if indices:
                    scene_centroids[scene_name] = np.mean(
                        [cal_hidden_list[i] for i in indices], axis=0)

        cos_scores = []
        mass_scores = []
        for t in all_test:
            if use_per_scene_centroids and scene_centroids:
                cos = min(cosine_dist(t['hidden'], c) for c in scene_centroids.values())
            else:
                cos = cosine_dist(t['hidden'], global_centroid)
            cos_scores.append(cos)
            mass_scores.append(1 - t['mass'])

        cos_n = np.array(cos_scores)
        mass_n = np.array(mass_scores)
        # Normalize to [0,1]
        cos_min, cos_max = cos_n.min(), cos_n.max()
        mass_min, mass_max = mass_n.min(), mass_n.max()
        if cos_max > cos_min:
            cos_n = (cos_n - cos_min) / (cos_max - cos_min)
        if mass_max > mass_min:
            mass_n = (mass_n - mass_min) / (mass_max - mass_min)

        combo = 0.7 * cos_n + 0.3 * mass_n
        auroc_combo = roc_auc_score(labels, combo)
        auroc_cos = roc_auc_score(labels, cos_scores)
        return auroc_combo, auroc_cos

    # ===================================================================
    # Experiment A: Calibration size sensitivity
    # ===================================================================
    print("\n" + "=" * 60, flush=True)
    print("A. Calibration Size Sensitivity", flush=True)
    print("=" * 60, flush=True)

    all_cal = []
    for scene in ['highway', 'urban', 'night', 'foggy']:
        all_cal.extend(cal_pool[scene])

    sizes = [2, 4, 8, 12, 16, 24, 32, 40]
    n_seeds = 5

    print(f"\n  {'Size':>6} | {'Combo AUROC':>12} ({'±std':>6}) | "
          f"{'Cos AUROC':>12} ({'±std':>6})", flush=True)
    print("  " + "-" * 60, flush=True)

    size_results = []
    for size in sizes:
        combo_aurocs = []
        cos_aurocs = []
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed * 42 + size)
            if size >= len(all_cal):
                indices = list(range(len(all_cal)))
            else:
                indices = rng.choice(len(all_cal), size=size, replace=False)
            cal_h = [all_cal[i]['hidden'] for i in indices]
            cal_m = [all_cal[i]['mass'] for i in indices]
            combo, cos = compute_auroc(cal_h, cal_m)
            combo_aurocs.append(combo)
            cos_aurocs.append(cos)

        mean_combo = np.mean(combo_aurocs)
        std_combo = np.std(combo_aurocs)
        mean_cos = np.mean(cos_aurocs)
        std_cos = np.std(cos_aurocs)
        print(f"  {size:>6} | {mean_combo:>12.3f} ({std_combo:>6.3f}) | "
              f"{mean_cos:>12.3f} ({std_cos:>6.3f})", flush=True)
        size_results.append({
            'size': size, 'combo_mean': mean_combo, 'combo_std': std_combo,
            'cos_mean': mean_cos, 'cos_std': std_cos,
        })

    # ===================================================================
    # Experiment B: Composition sensitivity
    # ===================================================================
    print("\n" + "=" * 60, flush=True)
    print("B. Calibration Composition", flush=True)
    print("=" * 60, flush=True)

    compositions = {
        'highway only (10)': {'highway': list(range(10))},
        'urban only (10)': {'urban': list(range(10))},
        'night only (10)': {'night': list(range(10))},
        'foggy only (10)': {'foggy': list(range(10))},
        'highway+urban (20)': {'highway': list(range(10)), 'urban': list(range(10))},
        'highway+night (20)': {'highway': list(range(10)), 'night': list(range(10))},
        'all 4 scenes (40)': {s: list(range(10)) for s in cal_fns},
        'all 4 scenes (20)': {s: list(range(5)) for s in cal_fns},
    }

    print(f"\n  {'Composition':<30} | {'Combo':>8} | {'Cosine':>8}", flush=True)
    print("  " + "-" * 55, flush=True)

    comp_results = []
    for name, comp in compositions.items():
        cal_h = []
        cal_m = []
        per_scene_indices = {}
        idx = 0
        for scene, sample_ids in comp.items():
            scene_start = idx
            for sid in sample_ids:
                cal_h.append(cal_pool[scene][sid]['hidden'])
                cal_m.append(cal_pool[scene][sid]['mass'])
                idx += 1
            per_scene_indices[scene] = list(range(scene_start, idx))

        combo_g, cos_g = compute_auroc(cal_h, cal_m)
        combo_ps, cos_ps = compute_auroc(cal_h, cal_m,
                                          use_per_scene_centroids=per_scene_indices)
        print(f"  {name:<30} | {combo_g:>8.3f} | {cos_g:>8.3f} | "
              f"(per-scene: {combo_ps:.3f})", flush=True)
        comp_results.append({
            'name': name, 'combo_global': combo_g, 'cos_global': cos_g,
            'combo_perscene': combo_ps, 'cos_perscene': cos_ps,
        })

    # ===================================================================
    # Experiment C: Transfer across conditions
    # ===================================================================
    print("\n" + "=" * 60, flush=True)
    print("C. Transfer Across Conditions", flush=True)
    print("=" * 60, flush=True)

    print(f"\n  {'Cal condition':<20} | {'Overall':>8} | {'Snow':>8} | "
          f"{'Flood':>8} | {'Offroad':>8} | {'Tunnel':>8}", flush=True)
    print("  " + "-" * 75, flush=True)

    transfer_results = []
    for cal_scene in cal_fns:
        cal_h = [cal_pool[cal_scene][i]['hidden'] for i in range(10)]
        cal_m = [cal_pool[cal_scene][i]['mass'] for i in range(10)]
        centroid = np.mean(cal_h, axis=0)

        # Overall
        cos_all = [cosine_dist(t['hidden'], centroid) for t in all_test]
        mass_all = [1 - t['mass'] for t in all_test]
        cos_n = np.array(cos_all)
        mass_n = np.array(mass_all)
        cos_n = (cos_n - cos_n.min()) / (cos_n.max() - cos_n.min() + 1e-10)
        mass_n = (mass_n - mass_n.min()) / (mass_n.max() - mass_n.min() + 1e-10)
        overall = roc_auc_score(labels, 0.7 * cos_n + 0.3 * mass_n)

        # Per-OOD type
        per_ood = {}
        for ood_type in ['snow', 'flooded', 'offroad', 'tunnel']:
            type_ood = [t for t in ood if t['scenario'] == ood_type]
            type_labels = [0]*len(easy) + [1]*len(type_ood)
            type_all = easy + type_ood
            type_cos = [cosine_dist(t['hidden'], centroid) for t in type_all]
            type_mass = [1 - t['mass'] for t in type_all]
            tc_n = np.array(type_cos)
            tm_n = np.array(type_mass)
            tc_n = (tc_n - tc_n.min()) / (tc_n.max() - tc_n.min() + 1e-10)
            tm_n = (tm_n - tm_n.min()) / (tm_n.max() - tm_n.min() + 1e-10)
            per_ood[ood_type] = roc_auc_score(type_labels, 0.7 * tc_n + 0.3 * tm_n)

        print(f"  {cal_scene:<20} | {overall:>8.3f} | {per_ood['snow']:>8.3f} | "
              f"{per_ood['flooded']:>8.3f} | {per_ood['offroad']:>8.3f} | "
              f"{per_ood['tunnel']:>8.3f}", flush=True)
        transfer_results.append({
            'cal_scene': cal_scene, 'overall': overall, **per_ood,
        })

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'calibration_robustness',
        'experiment_number': 45,
        'timestamp': timestamp,
        'n_cal_pool': sum(len(v) for v in cal_pool.values()),
        'n_test': len(test_data),
        'size_sensitivity': size_results,
        'composition': comp_results,
        'transfer': transfer_results,
    }
    output_path = os.path.join(RESULTS_DIR, f"calibration_robustness_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
