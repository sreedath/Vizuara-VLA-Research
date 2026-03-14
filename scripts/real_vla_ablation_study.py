"""
Systematic Ablation Study of the OOD Detection Pipeline.

Tests each component's individual and cumulative contribution:
1. Cosine distance (single global centroid)
2. + Per-scene centroids
3. + Temporal aggregation (multi-step mean)
4. + Action mass combination
5. + Conformal threshold selection

Also tests removal of each component from the full pipeline to measure
its marginal contribution (leave-one-out ablation).

Uses realistic test images to show the pipeline in its hardest setting.

Experiment 51 in the CalibDrive series.
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


# === Realistic image generators ===
def create_highway_realistic(idx):
    rng = np.random.default_rng(idx * 7001)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]  # sky
    for row in range(SIZE[0]//2, SIZE[0]):
        t = (row - SIZE[0]//2) / (SIZE[0]//2)
        gray = int(80 + t * 30)
        img[row, :] = [gray, gray, gray]
    # Lane markings
    cx = SIZE[1]//2 + rng.integers(-5, 6)
    img[SIZE[0]//2:, cx-2:cx+2] = [255, 255, 200]
    # Road edge lines
    img[SIZE[0]//2:, SIZE[1]//4:SIZE[1]//4+2] = [200, 200, 200]
    img[SIZE[0]//2:, 3*SIZE[1]//4:3*SIZE[1]//4+2] = [200, 200, 200]
    noise = rng.integers(-8, 9, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_urban_realistic(idx):
    rng = np.random.default_rng(idx * 7002)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [135, 206, 235]  # sky
    img[SIZE[0]//3:SIZE[0]//2] = [139, 119, 101]  # buildings
    # Windows
    for bx in range(0, SIZE[1], 40):
        for by in range(SIZE[0]//3 + 5, SIZE[0]//2 - 5, 15):
            if rng.random() > 0.3:
                img[by:by+8, bx+5:bx+15] = [200, 220, 255]
    img[SIZE[0]//2:] = [60, 60, 60]  # road
    noise = rng.integers(-8, 9, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_night_realistic(idx):
    rng = np.random.default_rng(idx * 7003)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [15, 15, 30]  # dark sky
    img[SIZE[0]//2:] = [25, 25, 25]  # dark road
    # Headlight cone
    cy, cx = SIZE[0]*3//4, SIZE[1]//2
    for r in range(60, 0, -1):
        brightness = int(40 * (1 - r/60))
        for dy in range(-r, r):
            for dx in range(-r*2, r*2):
                py, px = cy + dy, cx + dx
                if 0 <= py < SIZE[0] and 0 <= px < SIZE[1]:
                    img[py, px] = np.clip(img[py, px].astype(int) + brightness, 0, 255)
    noise = rng.integers(-3, 4, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_foggy_realistic(idx):
    rng = np.random.default_rng(idx * 7004)
    img = create_highway_realistic(idx + 2000)
    fog = np.full_like(img, 200)
    alpha = 0.55 + rng.random() * 0.15
    blended = (alpha * fog + (1 - alpha) * img).astype(np.uint8)
    return blended

# OOD generators
def create_offroad(idx):
    rng = np.random.default_rng(idx * 7010)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]  # sky
    for row in range(SIZE[0]//2, SIZE[0]):
        t = (row - SIZE[0]//2) / (SIZE[0]//2)
        g = int(80 + rng.integers(-20, 20))
        img[row, :] = [34 + int(t*20), min(139, g + int(t*30)), 34]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_flooded(idx):
    rng = np.random.default_rng(idx * 7011)
    img = create_highway_realistic(idx + 3000)
    water_start = SIZE[0]*2//3
    for row in range(water_start, SIZE[0]):
        t = (row - water_start) / (SIZE[0] - water_start)
        blue = [int(30 + 60*t), int(60 + 80*t), int(120 + 80*t)]
        img[row, :] = np.clip(
            (0.4 * img[row, :].astype(float) + 0.6 * np.array(blue)).astype(int),
            0, 255
        ).astype(np.uint8)
    return img

def create_tunnel(idx):
    rng = np.random.default_rng(idx * 7012)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:] = [30, 25, 20]  # dark
    # Tunnel walls
    for col in range(SIZE[1]//4, 3*SIZE[1]//4):
        for row in range(SIZE[0]//4, 3*SIZE[0]//4):
            img[row, col] = [40, 35, 30]
    # Road
    img[SIZE[0]//2:, SIZE[1]//4:3*SIZE[1]//4] = [50, 50, 50]
    # Lights
    for lx in range(SIZE[1]//4 + 20, 3*SIZE[1]//4, 40):
        img[SIZE[0]//4:SIZE[0]//4+5, lx:lx+5] = [255, 200, 100]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_snow(idx):
    rng = np.random.default_rng(idx * 7013)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [200, 200, 210]  # overcast sky
    img[SIZE[0]//2:] = [220, 220, 230]  # snow-covered road
    # Snow particles
    for _ in range(200):
        sy, sx = rng.integers(0, SIZE[0]), rng.integers(0, SIZE[1])
        img[sy:sy+2, sx:sx+2] = 255
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def extract_signals(model, processor, image, prompt):
    """Extract hidden state and action mass."""
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
        probs = torch.softmax(score[0].float(), dim=0)
        masses.append(float(probs[action_start:].sum()))

    return {
        'hidden': hidden,
        'action_mass': float(np.mean(masses)),
    }


def cosine_dist(a, b):
    return 1.0 - float(np.dot(a / (np.linalg.norm(a) + 1e-10),
                               b / (np.linalg.norm(b) + 1e-10)))


def main():
    print("=" * 70, flush=True)
    print("SYSTEMATIC ABLATION STUDY", flush=True)
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

    # === Calibration: 8 per scene × 4 scenes = 32 ===
    print("\nCalibration...", flush=True)
    cal_scenes = {
        'highway': create_highway_realistic,
        'urban': create_urban_realistic,
        'night': create_night_realistic,
        'foggy': create_foggy_realistic,
    }
    cal_per_scene = 8
    cal_data = {scene: [] for scene in cal_scenes}
    all_cal = []

    for scene, fn in cal_scenes.items():
        for i in range(cal_per_scene):
            data = extract_signals(model, processor,
                                    Image.fromarray(fn(i + 9000)), prompt)
            cal_data[scene].append(data)
            all_cal.append(data)
            print(f"  Cal {scene}_{i}: mass={data['action_mass']:.3f}", flush=True)

    # Compute centroids
    global_centroid = np.mean([d['hidden'] for d in all_cal], axis=0)
    per_scene_centroids = {
        scene: np.mean([d['hidden'] for d in cal_data[scene]], axis=0)
        for scene in cal_scenes
    }
    print(f"  Global centroid norm: {np.linalg.norm(global_centroid):.2f}", flush=True)

    # === Test trajectories: 5 per scene × 5 steps ===
    print("\nGenerating test trajectories...", flush=True)
    id_scenes = {
        'highway': (create_highway_realistic, False),
        'urban': (create_urban_realistic, False),
        'night': (create_night_realistic, False),
        'foggy': (create_foggy_realistic, False),
    }
    ood_scenes = {
        'offroad': (create_offroad, True),
        'flooded': (create_flooded, True),
        'tunnel': (create_tunnel, True),
        'snow': (create_snow, True),
    }
    all_scenes = {**id_scenes, **ood_scenes}
    n_traj_per_scene = 5
    traj_len = 5

    trajectories = []
    total_inferences = sum(n_traj_per_scene * traj_len for _ in all_scenes)
    cnt = 0

    for scene, (fn, is_ood) in all_scenes.items():
        for t in range(n_traj_per_scene):
            traj_frames = []
            for step in range(traj_len):
                cnt += 1
                idx = t * 100 + step + 200
                data = extract_signals(model, processor,
                                        Image.fromarray(fn(idx)), prompt)
                # Compute all signals
                cos_global = cosine_dist(data['hidden'], global_centroid)
                cos_per_scene = min(
                    cosine_dist(data['hidden'], c) for c in per_scene_centroids.values()
                )
                traj_frames.append({
                    'cos_global': cos_global,
                    'cos_per_scene': cos_per_scene,
                    'action_mass': data['action_mass'],
                })
                if cnt % 20 == 0:
                    print(f"  [{cnt}/{total_inferences}] {scene}_t{t}_s{step}: "
                          f"glob={cos_global:.3f}, scene={cos_per_scene:.3f}, "
                          f"mass={data['action_mass']:.3f}", flush=True)

            trajectories.append({
                'scenario': scene,
                'is_ood': is_ood,
                'traj_id': t,
                'frames': traj_frames,
            })

    # ===================================================================
    # Ablation Analysis
    # ===================================================================
    print("\n" + "=" * 70, flush=True)
    print("ABLATION RESULTS", flush=True)
    print("=" * 70, flush=True)

    labels = [0 if not tr['is_ood'] else 1 for tr in trajectories]

    def compute_ablation_auroc(score_fn, name):
        scores = [score_fn(tr) for tr in trajectories]
        try:
            auroc = roc_auc_score(labels, scores)
        except ValueError:
            auroc = 0.5
        return auroc

    # A. Single-frame methods (use first frame only)
    print("\nA. Single-Frame Ablation", flush=True)
    print("-" * 60, flush=True)

    single_frame_results = {}

    # 1. Global cosine (single frame)
    auroc = compute_ablation_auroc(
        lambda tr: tr['frames'][0]['cos_global'],
        "Global cosine (1 frame)"
    )
    single_frame_results['global_cos_1f'] = auroc
    print(f"  Global cosine (1 frame):          {auroc:.3f}", flush=True)

    # 2. Per-scene cosine (single frame)
    auroc = compute_ablation_auroc(
        lambda tr: tr['frames'][0]['cos_per_scene'],
        "Per-scene cosine (1 frame)"
    )
    single_frame_results['scene_cos_1f'] = auroc
    print(f"  Per-scene cosine (1 frame):       {auroc:.3f}", flush=True)

    # 3. Action mass (single frame)
    auroc = compute_ablation_auroc(
        lambda tr: 1 - tr['frames'][0]['action_mass'],
        "Action mass (1 frame)"
    )
    single_frame_results['mass_1f'] = auroc
    print(f"  Action mass (1 frame):            {auroc:.3f}", flush=True)

    # 4. Global cosine + mass (single frame, 0.7+0.3)
    auroc = compute_ablation_auroc(
        lambda tr: 0.7 * tr['frames'][0]['cos_global'] + 0.3 * (1 - tr['frames'][0]['action_mass']),
        "Global cos + mass (1 frame)"
    )
    single_frame_results['global_combo_1f'] = auroc
    print(f"  Global cos + mass (1 frame):      {auroc:.3f}", flush=True)

    # 5. Per-scene cosine + mass (single frame, 0.7+0.3)
    auroc = compute_ablation_auroc(
        lambda tr: 0.7 * tr['frames'][0]['cos_per_scene'] + 0.3 * (1 - tr['frames'][0]['action_mass']),
        "Per-scene cos + mass (1 frame)"
    )
    single_frame_results['scene_combo_1f'] = auroc
    print(f"  Per-scene cos + mass (1 frame):   {auroc:.3f}", flush=True)

    # B. Temporal aggregation methods
    print("\nB. Temporal Ablation (5-step trajectories)", flush=True)
    print("-" * 60, flush=True)

    temporal_results = {}

    for steps in [1, 2, 3, 5]:
        # Global cosine, temporal mean
        auroc = compute_ablation_auroc(
            lambda tr, s=steps: np.mean([f['cos_global'] for f in tr['frames'][:s]]),
            f"Global cos ({steps}-step)"
        )
        temporal_results[f'global_cos_{steps}s'] = auroc
        print(f"  Global cosine ({steps}-step mean):     {auroc:.3f}", flush=True)

    print()
    for steps in [1, 2, 3, 5]:
        # Per-scene cosine, temporal mean
        auroc = compute_ablation_auroc(
            lambda tr, s=steps: np.mean([f['cos_per_scene'] for f in tr['frames'][:s]]),
            f"Scene cos ({steps}-step)"
        )
        temporal_results[f'scene_cos_{steps}s'] = auroc
        print(f"  Per-scene cosine ({steps}-step mean): {auroc:.3f}", flush=True)

    # C. Full pipeline cumulative ablation
    print("\nC. Cumulative Pipeline Ablation", flush=True)
    print("-" * 60, flush=True)

    cumulative = {}

    # Stage 1: Global cosine, 1 frame
    auroc = compute_ablation_auroc(
        lambda tr: tr['frames'][0]['cos_global'],
        "Stage 1"
    )
    cumulative['1_global_1f'] = auroc
    print(f"  1. Global cosine, 1 frame:            {auroc:.3f}", flush=True)

    # Stage 2: + Per-scene centroids
    auroc = compute_ablation_auroc(
        lambda tr: tr['frames'][0]['cos_per_scene'],
        "Stage 2"
    )
    cumulative['2_scene_1f'] = auroc
    print(f"  2. + Per-scene centroids:             {auroc:.3f} "
          f"(Δ={auroc - cumulative['1_global_1f']:+.3f})", flush=True)

    # Stage 3: + Temporal aggregation (5-step)
    auroc = compute_ablation_auroc(
        lambda tr: np.mean([f['cos_per_scene'] for f in tr['frames']]),
        "Stage 3"
    )
    cumulative['3_scene_temporal'] = auroc
    print(f"  3. + Temporal aggregation (5-step):   {auroc:.3f} "
          f"(Δ={auroc - cumulative['2_scene_1f']:+.3f})", flush=True)

    # Stage 4: + Action mass combination (0.7cos + 0.3mass)
    auroc = compute_ablation_auroc(
        lambda tr: 0.7 * np.mean([f['cos_per_scene'] for f in tr['frames']]) +
                   0.3 * np.mean([1 - f['action_mass'] for f in tr['frames']]),
        "Stage 4"
    )
    cumulative['4_full_pipeline'] = auroc
    print(f"  4. + Action mass combination:         {auroc:.3f} "
          f"(Δ={auroc - cumulative['3_scene_temporal']:+.3f})", flush=True)

    total_gain = cumulative['4_full_pipeline'] - cumulative['1_global_1f']
    print(f"\n  Total pipeline gain: {total_gain:+.3f}", flush=True)

    # D. Leave-one-out ablation from full pipeline
    print("\nD. Leave-One-Out Ablation (from full pipeline)", flush=True)
    print("-" * 60, flush=True)

    full_auroc = cumulative['4_full_pipeline']

    # Remove per-scene (use global instead)
    auroc = compute_ablation_auroc(
        lambda tr: 0.7 * np.mean([f['cos_global'] for f in tr['frames']]) +
                   0.3 * np.mean([1 - f['action_mass'] for f in tr['frames']]),
        "w/o per-scene"
    )
    print(f"  Full pipeline:                        {full_auroc:.3f}", flush=True)
    print(f"  w/o per-scene centroids (use global): {auroc:.3f} "
          f"(drop={full_auroc - auroc:+.3f})", flush=True)

    # Remove temporal (use single frame)
    auroc_no_temp = compute_ablation_auroc(
        lambda tr: 0.7 * tr['frames'][0]['cos_per_scene'] +
                   0.3 * (1 - tr['frames'][0]['action_mass']),
        "w/o temporal"
    )
    print(f"  w/o temporal aggregation:             {auroc_no_temp:.3f} "
          f"(drop={full_auroc - auroc_no_temp:+.3f})", flush=True)

    # Remove action mass (cosine only)
    auroc_no_mass = compute_ablation_auroc(
        lambda tr: np.mean([f['cos_per_scene'] for f in tr['frames']]),
        "w/o mass"
    )
    print(f"  w/o action mass (cosine only):        {auroc_no_mass:.3f} "
          f"(drop={full_auroc - auroc_no_mass:+.3f})", flush=True)

    # Remove cosine (mass only)
    auroc_mass_only = compute_ablation_auroc(
        lambda tr: np.mean([1 - f['action_mass'] for f in tr['frames']]),
        "mass only"
    )
    print(f"  w/o cosine (mass only):               {auroc_mass_only:.3f} "
          f"(drop={full_auroc - auroc_mass_only:+.3f})", flush=True)

    # E. Per-OOD type breakdown
    print("\nE. Per-OOD Type Performance", flush=True)
    print("-" * 60, flush=True)

    ood_types = ['offroad', 'flooded', 'tunnel', 'snow']
    id_trajs = [tr for tr in trajectories if not tr['is_ood']]

    print(f"\n  {'Method':<35}", end='', flush=True)
    for ood_type in ood_types:
        print(f" {ood_type:>10}", end='', flush=True)
    print(f" {'Mean':>8}", flush=True)
    print("  " + "-" * (35 + 11*len(ood_types) + 8), flush=True)

    methods = {
        'Global cos (1f)': lambda tr: tr['frames'][0]['cos_global'],
        'Per-scene cos (1f)': lambda tr: tr['frames'][0]['cos_per_scene'],
        'Action mass (1f)': lambda tr: 1 - tr['frames'][0]['action_mass'],
        'Per-scene cos (5-step)': lambda tr: np.mean([f['cos_per_scene'] for f in tr['frames']]),
        'Full pipeline': lambda tr: 0.7 * np.mean([f['cos_per_scene'] for f in tr['frames']]) +
                                     0.3 * np.mean([1 - f['action_mass'] for f in tr['frames']]),
    }

    per_type_results = {}
    for method_name, score_fn in methods.items():
        per_type_results[method_name] = {}
        print(f"  {method_name:<35}", end='', flush=True)
        aurocs_list = []
        for ood_type in ood_types:
            type_ood = [tr for tr in trajectories if tr['scenario'] == ood_type]
            type_all = id_trajs + type_ood
            type_labels = [0]*len(id_trajs) + [1]*len(type_ood)
            type_scores = [score_fn(tr) for tr in type_all]
            try:
                auroc = roc_auc_score(type_labels, type_scores)
            except ValueError:
                auroc = 0.5
            per_type_results[method_name][ood_type] = auroc
            aurocs_list.append(auroc)
            print(f" {auroc:>10.3f}", end='', flush=True)
        mean_auroc = np.mean(aurocs_list)
        per_type_results[method_name]['mean'] = mean_auroc
        print(f" {mean_auroc:>8.3f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'ablation_study',
        'experiment_number': 51,
        'timestamp': timestamp,
        'n_cal': len(all_cal),
        'n_trajectories': len(trajectories),
        'traj_length': traj_len,
        'total_inferences': total_inferences + len(all_cal),
        'single_frame': single_frame_results,
        'temporal': temporal_results,
        'cumulative': cumulative,
        'per_type': per_type_results,
        'trajectories': [{
            'scenario': tr['scenario'],
            'is_ood': tr['is_ood'],
            'traj_id': tr['traj_id'],
            'frames': tr['frames'],
        } for tr in trajectories],
    }
    output_path = os.path.join(RESULTS_DIR, f"ablation_study_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
