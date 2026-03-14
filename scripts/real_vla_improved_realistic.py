"""
Improved Realistic Image Detection on Real OpenVLA-7B.

Following the AUROC drop with realistic images (Exp 40: 0.668),
this experiment tests strategies to improve detection:

1. Diverse calibration: include night + foggy in calibration set
2. Per-feature normalization: z-score hidden states before cosine
3. Multiple centroids: separate highway, urban, night centroids
4. Combined signal: cosine + action mass + entropy fusion
5. Contrastive threshold: use calibration variance for adaptive threshold

Experiment 41 in the CalibDrive series.
"""
import os
import json
import time
import datetime
import numpy as np
import torch
from PIL import Image, ImageFilter
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


CAL_SCENARIOS = {
    'highway_realistic': create_realistic_highway,
    'urban_realistic': create_realistic_urban,
    'night_driving': create_night_driving,
    'foggy_road': create_foggy_road,
}

TEST_EASY = {
    'highway_realistic': {'n': 10, 'fn': create_realistic_highway},
    'urban_realistic': {'n': 10, 'fn': create_realistic_urban},
    'night_driving': {'n': 8, 'fn': create_night_driving},
    'foggy_road': {'n': 8, 'fn': create_foggy_road},
}

TEST_OOD = {
    'snow_road': {'n': 8, 'fn': create_snow_road},
    'flooded_road': {'n': 8, 'fn': create_flooded_road},
    'offroad': {'n': 8, 'fn': create_offroad},
    'tunnel': {'n': 8, 'fn': create_tunnel},
}


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
    dim_ents = []
    for score in outputs.scores[:7]:
        full_logits = score[0].float()
        full_probs = torch.softmax(full_logits, dim=0)
        action_probs = full_probs[action_start:].cpu().numpy()
        dim_masses.append(float(action_probs.sum()))
        ent = -torch.sum(full_probs * torch.log(full_probs + 1e-10)).item()
        dim_ents.append(ent)

    return hidden, float(np.mean(dim_masses)), float(np.mean(dim_ents))


def cosine_dist(a, b):
    a_n = a / (np.linalg.norm(a) + 1e-10)
    b_n = b / (np.linalg.norm(b) + 1e-10)
    return 1.0 - float(np.dot(a_n, b_n))


def main():
    print("=" * 70, flush=True)
    print("IMPROVED REALISTIC IMAGE DETECTION", flush=True)
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

    # Phase 1: Diverse calibration (ALL driving scenarios)
    print("\nPhase 1: Diverse calibration...", flush=True)
    cal_data = {}  # scene -> list of (hidden, mass, ent)
    all_cal_hidden = []

    for scene, fn in CAL_SCENARIOS.items():
        cal_data[scene] = []
        for i in range(8):
            img_arr = fn(i + 500)
            image = Image.fromarray(img_arr)
            hidden, mass, ent = extract_hidden(model, processor, image, prompt)
            cal_data[scene].append({'hidden': hidden, 'mass': mass, 'ent': ent})
            all_cal_hidden.append(hidden)
        print(f"  {scene}: {len(cal_data[scene])} cal samples", flush=True)

    cal_arr = np.array(all_cal_hidden)  # (32, 4096)

    # Method 1: Global centroid (baseline, like Exp 40)
    global_mean = np.mean(cal_arr, axis=0)
    global_norm = global_mean / (np.linalg.norm(global_mean) + 1e-10)

    # Method 2: Per-scene centroids
    scene_centroids = {}
    for scene in CAL_SCENARIOS:
        sc = np.array([d['hidden'] for d in cal_data[scene]])
        scene_centroids[scene] = np.mean(sc, axis=0)

    # Method 3: Z-scored hidden states
    cal_std = np.std(cal_arr, axis=0) + 1e-10
    cal_mean_full = np.mean(cal_arr, axis=0)

    print(f"  Total calibration: {len(all_cal_hidden)} samples from "
          f"{len(CAL_SCENARIOS)} scenarios", flush=True)

    # Phase 2: Test
    print(f"\nPhase 2: Testing...", flush=True)
    test_results = []
    sample_idx = 0
    total = sum(s['n'] for s in TEST_EASY.values()) + sum(s['n'] for s in TEST_OOD.values())

    for scenario, config in {**TEST_EASY, **TEST_OOD}.items():
        is_ood = scenario in TEST_OOD
        fn = config['fn']
        for i in range(config['n']):
            sample_idx += 1
            img_arr = fn(i)
            image = Image.fromarray(img_arr)
            hidden, mass, ent = extract_hidden(model, processor, image, prompt)

            # Method 1: Global cosine
            cos_global = cosine_dist(hidden, global_mean)

            # Method 2: Min per-scene cosine
            cos_per_scene = min(cosine_dist(hidden, c) for c in scene_centroids.values())

            # Method 3: Z-scored cosine
            h_z = (hidden - cal_mean_full) / cal_std
            gm_z = np.zeros_like(cal_mean_full)  # mean is 0 after z-scoring
            cos_z = cosine_dist(h_z, gm_z)

            # Method 4: L2 distance (raw)
            l2_global = float(np.linalg.norm(hidden - global_mean))

            # Method 5: Combined signal
            # Normalize each to [0, 1] range using calibration stats
            cos_combined = cos_global * 0.5 + (1 - mass) * 0.3 + (ent / 10.0) * 0.2

            # Method 6: Norm-aware cosine
            h_norm_val = float(np.linalg.norm(hidden))
            cal_norm_mean = float(np.mean(np.linalg.norm(cal_arr, axis=1)))
            norm_ratio = abs(h_norm_val - cal_norm_mean) / cal_norm_mean
            cos_norm_aware = cos_global + norm_ratio * 0.5

            result = {
                'scenario': scenario,
                'is_ood': is_ood,
                'idx': i,
                'cos_global': cos_global,
                'cos_per_scene': cos_per_scene,
                'cos_z': cos_z,
                'l2_global': l2_global,
                'cos_combined': cos_combined,
                'cos_norm_aware': cos_norm_aware,
                'action_mass': mass,
                'entropy': ent,
                'hidden_norm': h_norm_val,
            }
            test_results.append(result)

            if sample_idx % 10 == 0 or sample_idx == total:
                print(f"  [{sample_idx}/{total}] {scenario}_{i}: "
                      f"cos={cos_global:.4f}, per_scene={cos_per_scene:.4f}, "
                      f"mass={mass:.4f}", flush=True)

    # ===================================================================
    # Analysis
    # ===================================================================
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    easy_r = [r for r in test_results if not r['is_ood']]
    ood_r = [r for r in test_results if r['is_ood']]
    labels = [0] * len(easy_r) + [1] * len(ood_r)

    methods = {
        'Global centroid (Exp 40 baseline)': [r['cos_global'] for r in easy_r + ood_r],
        'Per-scene min centroid': [r['cos_per_scene'] for r in easy_r + ood_r],
        'Z-scored cosine': [r['cos_z'] for r in easy_r + ood_r],
        'L2 distance': [r['l2_global'] for r in easy_r + ood_r],
        'Combined (cos+mass+ent)': [r['cos_combined'] for r in easy_r + ood_r],
        'Norm-aware cosine': [r['cos_norm_aware'] for r in easy_r + ood_r],
        'Action mass (1-mass)': [1 - r['action_mass'] for r in easy_r + ood_r],
        'Entropy': [r['entropy'] for r in easy_r + ood_r],
    }

    print("\n1. Overall AUROC Comparison", flush=True)
    print("-" * 60, flush=True)
    for name, scores in methods.items():
        auroc = roc_auc_score(labels, scores)
        print(f"  {name:<35}: AUROC = {auroc:.3f}", flush=True)

    # Per-OOD type for top methods
    print("\n2. Per-OOD-Type AUROC (top methods)", flush=True)
    print("-" * 80, flush=True)
    ood_types = list(TEST_OOD.keys())
    top_methods = ['Global centroid (Exp 40 baseline)', 'Per-scene min centroid',
                   'Norm-aware cosine', 'Combined (cos+mass+ent)']

    header_parts = [f"{'Method':<35}"]
    for t in ood_types:
        header_parts.append(f"{t:>12}")
    print("  " + " | ".join(header_parts), flush=True)

    for name in top_methods:
        parts = [f"{name:<33}"]
        for ood_type in ood_types:
            type_ood = [r for r in ood_r if r['scenario'] == ood_type]
            type_labels = [0] * len(easy_r) + [1] * len(type_ood)

            if name == 'Global centroid (Exp 40 baseline)':
                type_scores = [r['cos_global'] for r in easy_r + type_ood]
            elif name == 'Per-scene min centroid':
                type_scores = [r['cos_per_scene'] for r in easy_r + type_ood]
            elif name == 'Norm-aware cosine':
                type_scores = [r['cos_norm_aware'] for r in easy_r + type_ood]
            elif name == 'Combined (cos+mass+ent)':
                type_scores = [r['cos_combined'] for r in easy_r + type_ood]
            else:
                type_scores = [0] * len(type_labels)

            auroc = roc_auc_score(type_labels, type_scores)
            parts.append(f"{auroc:>12.3f}")
        print("  " + " | ".join(parts), flush=True)

    # Per-scenario mean scores
    print("\n3. Per-Scenario Mean Scores", flush=True)
    print("-" * 80, flush=True)
    for scenario in {**TEST_EASY, **TEST_OOD}:
        s_r = [r for r in test_results if r['scenario'] == scenario]
        is_ood = scenario in TEST_OOD
        mean_cos = np.mean([r['cos_global'] for r in s_r])
        mean_ps = np.mean([r['cos_per_scene'] for r in s_r])
        mean_na = np.mean([r['cos_norm_aware'] for r in s_r])
        label = 'OOD' if is_ood else 'ID'
        print(f"  {scenario:<20} [{label:>3}]: global={mean_cos:.4f}, "
              f"per_scene={mean_ps:.4f}, norm_aware={mean_na:.4f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'improved_realistic',
        'experiment_number': 41,
        'timestamp': timestamp,
        'n_cal': len(all_cal_hidden),
        'n_test_easy': len(easy_r),
        'n_test_ood': len(ood_r),
        'results': [{k: v for k, v in r.items()} for r in test_results],
    }

    output_path = os.path.join(RESULTS_DIR, f"improved_realistic_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
