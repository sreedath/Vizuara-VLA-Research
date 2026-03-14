"""
Realistic Image Test for OOD Detection on Real OpenVLA-7B.

Tests cosine distance with more visually complex/realistic synthetic images
instead of simple color blocks. Creates images with:
1. Gradient skies, textured roads, lane markings
2. Urban scenes with building facades, traffic lights
3. Night driving (dark with headlight cones)
4. Weather corruptions (fog overlay, rain streaks)

OOD types also made more realistic:
- Snow-covered road (white overlay on road area)
- Flooded road (blue-tinted road)
- Wrong-way driving (scene vertically flipped)
- Off-road/dirt path
- Tunnel (dark with small bright opening)

Experiment 40 in the CalibDrive series.
"""
import os
import json
import time
import datetime
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from sklearn.metrics import roc_auc_score

RESULTS_DIR = "/workspace/Vizuara-VLA-Research/experiments"
os.makedirs(RESULTS_DIR, exist_ok=True)

SIZE = (256, 256)


def create_realistic_highway(idx):
    """Highway with gradient sky, textured road, lane markings."""
    rng = np.random.default_rng(idx * 4001)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    # Gradient sky
    for y in range(SIZE[0]//2):
        frac = y / (SIZE[0]//2)
        r = int(80 + (135 - 80) * frac)
        g = int(150 + (206 - 150) * frac)
        b = int(255 + (235 - 255) * frac)
        img[y, :] = [r, g, b]
    # Textured road
    for y in range(SIZE[0]//2, SIZE[0]):
        base = 60 + rng.integers(-5, 6)
        img[y, :] = [base, base, base]
    # Lane markings (dashed white lines)
    for y in range(SIZE[0]//2 + 10, SIZE[0], 20):
        if (y // 20) % 2 == 0:
            img[y:y+8, SIZE[1]//2-2:SIZE[1]//2+2] = [220, 220, 220]
    # Road edges
    img[SIZE[0]//2:, :5] = [200, 200, 200]
    img[SIZE[0]//2:, -5:] = [200, 200, 200]
    noise = rng.integers(-3, 4, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def create_realistic_urban(idx):
    """Urban scene with buildings, road, sidewalk."""
    rng = np.random.default_rng(idx * 4002)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    # Sky
    img[:SIZE[0]//4] = [135, 206, 235]
    # Buildings (varied colors)
    for x_start in range(0, SIZE[1], 40):
        h = rng.integers(SIZE[0]//4, SIZE[0]//2)
        color = rng.integers(100, 200, 3)
        img[SIZE[0]//4:h, x_start:x_start+38] = color
        # Windows
        for wy in range(SIZE[0]//4 + 5, h - 5, 12):
            for wx in range(x_start + 3, x_start + 35, 10):
                img[wy:wy+6, wx:wx+6] = [200, 220, 255]
    # Sidewalk
    img[SIZE[0]//2:SIZE[0]//2+20] = [180, 170, 160]
    # Road
    img[SIZE[0]//2+20:] = [70, 70, 70]
    noise = rng.integers(-3, 4, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def create_night_driving(idx):
    """Night scene with dark sky, headlight cones."""
    rng = np.random.default_rng(idx * 4003)
    img = np.full((*SIZE, 3), 15, dtype=np.uint8)
    # Road slightly lighter
    img[SIZE[0]//2:] = [30, 30, 30]
    # Headlight cone
    for y in range(SIZE[0]//2, SIZE[0]):
        width = int((y - SIZE[0]//2) * 0.8)
        center = SIZE[1] // 2
        brightness = max(0, 120 - (y - SIZE[0]//2))
        img[y, max(0, center-width):min(SIZE[1], center+width)] = [brightness, brightness, int(brightness*0.8)]
    # Stars
    for _ in range(20):
        sy, sx = rng.integers(0, SIZE[0]//2), rng.integers(0, SIZE[1])
        img[sy, sx] = [255, 255, 255]
    noise = rng.integers(-2, 3, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def create_foggy_road(idx):
    """Foggy road — reduced contrast with white overlay."""
    rng = np.random.default_rng(idx * 4004)
    base = create_realistic_highway(idx + 1000)
    fog = np.full_like(base, 200)
    alpha = 0.5 + rng.random() * 0.2  # 50-70% fog
    img = (base * (1 - alpha) + fog * alpha).astype(np.uint8)
    return img


def create_snow_road(idx):
    """Snow-covered road — white on road area (OOD)."""
    rng = np.random.default_rng(idx * 4005)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [200, 210, 220]  # overcast sky
    # Snow-covered ground
    for y in range(SIZE[0]//2, SIZE[0]):
        brightness = 200 + rng.integers(-10, 10)
        img[y, :] = [brightness, brightness, brightness]
    # Tire tracks
    img[SIZE[0]//2:, SIZE[1]//3:SIZE[1]//3+5] = [150, 140, 130]
    img[SIZE[0]//2:, 2*SIZE[1]//3:2*SIZE[1]//3+5] = [150, 140, 130]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def create_flooded_road(idx):
    """Flooded road — blue-tinted road surface (OOD)."""
    rng = np.random.default_rng(idx * 4006)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [100, 100, 110]  # dark sky
    # Water on road
    for y in range(SIZE[0]//2, SIZE[0]):
        depth = (y - SIZE[0]//2) / (SIZE[0]//2)
        r = int(40 + 30 * depth)
        g = int(80 + 40 * depth)
        b = int(140 + 50 * depth)
        img[y, :] = [r, g, b]
        # Reflections
        if rng.random() > 0.7:
            img[y, rng.integers(0, SIZE[1])] = [150, 180, 220]
    noise = rng.integers(-3, 4, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def create_offroad(idx):
    """Off-road/dirt path (OOD)."""
    rng = np.random.default_rng(idx * 4007)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [135, 206, 235]  # sky
    # Trees/vegetation
    for x in range(0, SIZE[1], 15):
        h = SIZE[0]//3 + rng.integers(0, 30)
        img[SIZE[0]//3:h, x:x+12] = [30 + rng.integers(0, 30),
                                       100 + rng.integers(0, 50),
                                       20 + rng.integers(0, 30)]
    # Dirt path
    img[SIZE[0]//2:] = [140, 100, 60]
    # Rocks
    for _ in range(15):
        ry = rng.integers(SIZE[0]//2, SIZE[0])
        rx = rng.integers(0, SIZE[1])
        rs = rng.integers(3, 8)
        gray = rng.integers(80, 130)
        img[max(0,ry-rs):ry+rs, max(0,rx-rs):rx+rs] = [gray, gray, gray]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def create_tunnel(idx):
    """Tunnel — dark with bright opening (OOD)."""
    rng = np.random.default_rng(idx * 4008)
    img = np.full((*SIZE, 3), 10, dtype=np.uint8)
    # Bright opening in center
    cy, cx = SIZE[0]//2 - 20, SIZE[1]//2
    for y in range(SIZE[0]):
        for x in range(SIZE[1]):
            dist = np.sqrt((y - cy)**2 + (x - cx)**2)
            if dist < 40:
                brightness = int(200 * (1 - dist / 40))
                img[y, x] = [brightness, brightness, brightness]
    # Road
    img[SIZE[0]//2:, :] = np.clip(img[SIZE[0]//2:, :].astype(int) + 20, 0, 255).astype(np.uint8)
    noise = rng.integers(-2, 3, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


SCENARIOS = {
    # In-distribution
    'highway_realistic': {'n': 15, 'fn': create_realistic_highway, 'difficulty': 'easy'},
    'urban_realistic': {'n': 15, 'fn': create_realistic_urban, 'difficulty': 'easy'},
    'night_driving': {'n': 10, 'fn': create_night_driving, 'difficulty': 'hard'},
    'foggy_road': {'n': 10, 'fn': create_foggy_road, 'difficulty': 'hard'},
    # OOD
    'snow_road': {'n': 8, 'fn': create_snow_road, 'difficulty': 'ood'},
    'flooded_road': {'n': 8, 'fn': create_flooded_road, 'difficulty': 'ood'},
    'offroad': {'n': 8, 'fn': create_offroad, 'difficulty': 'ood'},
    'tunnel': {'n': 8, 'fn': create_tunnel, 'difficulty': 'ood'},
}


def main():
    print("=" * 70, flush=True)
    print("REALISTIC IMAGE OOD DETECTION", flush=True)
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
    total = sum(s['n'] for s in SCENARIOS.values())
    print(f"Total samples: {total}", flush=True)

    # Phase 1: Calibration from highway+urban
    print("\nPhase 1: Calibration...", flush=True)
    cal_hidden = []
    for scene in ['highway_realistic', 'urban_realistic']:
        fn = SCENARIOS[scene]['fn']
        for i in range(10):
            img_arr = fn(i + 500)
            image = Image.fromarray(img_arr)
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

    cal_cos = sorted([1.0 - float(np.dot(h / (np.linalg.norm(h) + 1e-10), cal_norm))
                       for h in cal_hidden])
    alpha = 0.10
    q_idx = min(int(np.ceil((1 - alpha) * (len(cal_cos) + 1))) - 1, len(cal_cos) - 1)
    threshold = cal_cos[q_idx]
    print(f"  Centroid from {len(cal_hidden)} samples. Threshold: {threshold:.4f}", flush=True)

    # Phase 2: Test all scenarios
    print(f"\nPhase 2: Testing all scenarios...", flush=True)
    all_results = []
    sample_idx = 0

    for scenario, config in SCENARIOS.items():
        fn = config['fn']
        for i in range(config['n']):
            sample_idx += 1
            img_arr = fn(i)
            image = Image.fromarray(img_arr)
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

            h_norm = hidden / (np.linalg.norm(hidden) + 1e-10)
            cos_dist = 1.0 - float(np.dot(h_norm, cal_norm))

            # Action mass
            vocab_size = outputs.scores[0].shape[-1]
            action_start = vocab_size - 256
            dim_masses = []
            for score in outputs.scores[:7]:
                full_logits = score[0].float()
                full_probs = torch.softmax(full_logits, dim=0).cpu().numpy()
                dim_masses.append(float(full_probs[action_start:].sum()))

            result = {
                'scenario': scenario,
                'difficulty': config['difficulty'],
                'idx': i,
                'cos_dist': cos_dist,
                'action_mass': float(np.mean(dim_masses)),
                'flagged': cos_dist > threshold,
            }
            all_results.append(result)

            if sample_idx % 10 == 0 or sample_idx == total:
                print(f"  [{sample_idx}/{total}] {scenario}_{i}: "
                      f"cos={cos_dist:.4f}, mass={result['action_mass']:.4f}, "
                      f"{'FLAGGED' if result['flagged'] else 'ok'}", flush=True)

    # ===================================================================
    # Analysis
    # ===================================================================
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    # 1. Per-scenario statistics
    print("\n1. Per-Scenario Statistics", flush=True)
    print("-" * 80, flush=True)
    print(f"  {'Scenario':<20} | {'Diff':>6} | {'Cos mean':>10} | {'Mass mean':>10} | {'Flag%':>8}", flush=True)
    print("  " + "-" * 65, flush=True)

    for scenario in SCENARIOS:
        s_results = [r for r in all_results if r['scenario'] == scenario]
        mean_cos = np.mean([r['cos_dist'] for r in s_results])
        mean_mass = np.mean([r['action_mass'] for r in s_results])
        flag_rate = sum(1 for r in s_results if r['flagged']) / len(s_results)
        diff = SCENARIOS[scenario]['difficulty']
        print(f"  {scenario:<20} | {diff:>6} | {mean_cos:>10.4f} | {mean_mass:>10.4f} | {flag_rate:>7.0%}", flush=True)

    # 2. AUROC: easy vs OOD
    print("\n2. AUROC: Easy vs OOD", flush=True)
    print("-" * 80, flush=True)
    easy_idxs = [i for i, r in enumerate(all_results) if r['difficulty'] == 'easy']
    ood_idxs = [i for i, r in enumerate(all_results) if r['difficulty'] == 'ood']
    hard_idxs = [i for i, r in enumerate(all_results) if r['difficulty'] == 'hard']

    if ood_idxs:
        labels_eo = [0] * len(easy_idxs) + [1] * len(ood_idxs)
        cos_eo = [all_results[i]['cos_dist'] for i in easy_idxs + ood_idxs]
        mass_eo = [-all_results[i]['action_mass'] for i in easy_idxs + ood_idxs]
        auroc_cos_eo = roc_auc_score(labels_eo, cos_eo)
        auroc_mass_eo = roc_auc_score(labels_eo, mass_eo)
        print(f"  Easy vs OOD — Cosine: {auroc_cos_eo:.3f}, Mass: {auroc_mass_eo:.3f}", flush=True)

    if hard_idxs and ood_idxs:
        labels_ho = [0] * len(hard_idxs) + [1] * len(ood_idxs)
        cos_ho = [all_results[i]['cos_dist'] for i in hard_idxs + ood_idxs]
        mass_ho = [-all_results[i]['action_mass'] for i in hard_idxs + ood_idxs]
        auroc_cos_ho = roc_auc_score(labels_ho, cos_ho)
        auroc_mass_ho = roc_auc_score(labels_ho, mass_ho)
        print(f"  Hard vs OOD — Cosine: {auroc_cos_ho:.3f}, Mass: {auroc_mass_ho:.3f}", flush=True)

    if easy_idxs and hard_idxs:
        labels_eh = [0] * len(easy_idxs) + [1] * len(hard_idxs)
        cos_eh = [all_results[i]['cos_dist'] for i in easy_idxs + hard_idxs]
        auroc_cos_eh = roc_auc_score(labels_eh, cos_eh)
        print(f"  Easy vs Hard — Cosine: {auroc_cos_eh:.3f}", flush=True)

    # 3. Per-OOD-type AUROC
    print("\n3. Per-OOD-Type AUROC (Easy vs Each Type)", flush=True)
    print("-" * 80, flush=True)
    ood_types = [s for s in SCENARIOS if SCENARIOS[s]['difficulty'] == 'ood']
    for ood_type in ood_types:
        type_idxs = [i for i in ood_idxs if all_results[i]['scenario'] == ood_type]
        type_labels = [0] * len(easy_idxs) + [1] * len(type_idxs)
        cos_scores = [all_results[i]['cos_dist'] for i in easy_idxs + type_idxs]
        mass_scores = [-all_results[i]['action_mass'] for i in easy_idxs + type_idxs]
        auroc_cos = roc_auc_score(type_labels, cos_scores)
        auroc_mass = roc_auc_score(type_labels, mass_scores)
        print(f"  {ood_type:<20}: Cosine={auroc_cos:.3f}, Mass={auroc_mass:.3f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'realistic_images',
        'experiment_number': 40,
        'timestamp': timestamp,
        'threshold': threshold,
        'results': all_results,
    }

    output_path = os.path.join(RESULTS_DIR, f"realistic_images_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
