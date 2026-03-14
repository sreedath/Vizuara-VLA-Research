"""
Action Output Analysis Under OOD (v2).

Analyzes predicted actions (7-dim, bins 0-255 mapped from token IDs
31744-31999) for ID vs OOD inputs.

Experiment 126 in the CalibDrive series.
"""
import os
import json
import datetime
import numpy as np
import torch
from PIL import Image

RESULTS_DIR = "/workspace/Vizuara-VLA-Research/experiments"
os.makedirs(RESULTS_DIR, exist_ok=True)
SIZE = (256, 256)
ACTION_TOKEN_BASE = 31744  # OpenVLA action tokens: 31744-31999


def create_highway(idx):
    rng = np.random.default_rng(idx * 19001)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]
    img[SIZE[0]//2:] = [80, 80, 80]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [255, 255, 255]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_urban(idx):
    rng = np.random.default_rng(idx * 19002)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [135, 206, 235]
    img[SIZE[0]//3:SIZE[0]//2] = [139, 119, 101]
    img[SIZE[0]//2:] = [60, 60, 60]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_noise(idx):
    rng = np.random.default_rng(idx * 19003)
    return rng.integers(0, 256, (*SIZE, 3), dtype=np.uint8)

def create_indoor(idx):
    rng = np.random.default_rng(idx * 19004)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:] = [200, 180, 160]
    img[SIZE[0]//2:, :] = [100, 80, 60]
    img[:SIZE[0]//3, SIZE[1]//3:2*SIZE[1]//3] = [150, 200, 255]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_twilight_highway(idx):
    rng = np.random.default_rng(idx * 19010)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [70, 50, 80]
    img[SIZE[0]//2:] = [60, 60, 60]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [200, 200, 100]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_snow(idx):
    rng = np.random.default_rng(idx * 19014)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [200, 200, 210]
    img[SIZE[0]//2:] = [220, 220, 230]
    img[SIZE[0]//2:, SIZE[1]//2-2:SIZE[1]//2+2] = [180, 180, 190]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def predict_action(model, processor, image, prompt):
    """Get predicted action bins from OpenVLA."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    gen_ids = output[0, inputs['input_ids'].shape[1]:]

    # Extract action bins (token IDs 31744-31999 → bins 0-255)
    action_bins = []
    for tid in gen_ids.tolist():
        if ACTION_TOKEN_BASE <= tid < ACTION_TOKEN_BASE + 256:
            action_bins.append(tid - ACTION_TOKEN_BASE)

    return action_bins


def main():
    print("=" * 70, flush=True)
    print("ACTION OUTPUT ANALYSIS UNDER OOD (v2)", flush=True)
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

    categories = {
        'highway': (create_highway, 'ID'),
        'urban': (create_urban, 'ID'),
        'noise': (create_noise, 'OOD'),
        'indoor': (create_indoor, 'OOD'),
        'twilight': (create_twilight_highway, 'OOD'),
        'snow': (create_snow, 'OOD'),
    }

    dim_names = ['x_trans', 'y_trans', 'z_trans', 'x_rot', 'y_rot', 'z_rot', 'gripper']

    print("\n--- Predicting actions ---", flush=True)
    results = {}
    for cat_name, (fn, group) in categories.items():
        print(f"\n  {cat_name} ({group}):", flush=True)
        all_actions = []
        for i in range(10):
            img = Image.fromarray(fn(i + 2600))
            bins = predict_action(model, processor, img, prompt)
            all_actions.append(bins)
            bins_7 = bins[:7] if len(bins) >= 7 else bins
            print(f"    {i}: bins={bins_7}, n={len(bins)}", flush=True)

        # Analyze
        valid = [a[:7] for a in all_actions if len(a) >= 7]
        if valid:
            arr = np.array(valid, dtype=float)
            per_dim = {
                'mean': arr.mean(axis=0).tolist(),
                'std': arr.std(axis=0).tolist(),
                'min': arr.min(axis=0).tolist(),
                'max': arr.max(axis=0).tolist(),
            }
            # Intra-category spread (L2 distance to category centroid)
            centroid = arr.mean(axis=0)
            spread = float(np.mean([np.linalg.norm(a - centroid) for a in arr]))
        else:
            per_dim = {'mean': [], 'std': [], 'min': [], 'max': []}
            spread = 0.0

        results[cat_name] = {
            'group': group,
            'n_valid': len(valid),
            'n_total': len(all_actions),
            'per_dim': per_dim,
            'spread': spread,
            'raw_actions': [a[:7] for a in all_actions if len(a) >= 7],
        }

    # Cross-category analysis
    print("\n--- Per-Dimension Summary ---", flush=True)
    print(f"  {'Category':12s} | {'x_t':>6s} {'y_t':>6s} {'z_t':>6s} {'x_r':>6s} {'y_r':>6s} {'z_r':>6s} {'grip':>6s} | {'spread':>7s}", flush=True)
    print("  " + "-" * 80, flush=True)
    for cat_name, data in results.items():
        if data['per_dim']['mean']:
            means = data['per_dim']['mean']
            means_str = " ".join(f"{m:6.1f}" for m in means)
            print(f"  {cat_name:12s} | {means_str} | {data['spread']:7.1f}", flush=True)

    # ID vs OOD divergence
    print("\n--- ID vs OOD Action Divergence ---", flush=True)
    id_all = []
    for cat_name, data in results.items():
        if data['group'] == 'ID':
            id_all.extend(data['raw_actions'])

    if id_all:
        id_arr = np.array(id_all, dtype=float)
        id_center = id_arr.mean(axis=0)
        id_spread = float(np.mean([np.linalg.norm(a - id_center) for a in id_arr]))
        print(f"  ID center: [{', '.join(f'{v:.1f}' for v in id_center)}]", flush=True)
        print(f"  ID spread: {id_spread:.1f}", flush=True)

        for cat_name, data in results.items():
            if data['group'] == 'OOD' and data['raw_actions']:
                ood_arr = np.array(data['raw_actions'], dtype=float)
                ood_center = ood_arr.mean(axis=0)
                divergence = float(np.linalg.norm(ood_center - id_center))
                per_dim_diff = (ood_center - id_center).tolist()
                # Max absolute shift
                max_shift_dim = int(np.argmax(np.abs(per_dim_diff)))
                max_shift = per_dim_diff[max_shift_dim]
                print(f"  {cat_name:12s}: divergence={divergence:.1f}, max_shift=dim{max_shift_dim}({dim_names[max_shift_dim]})={max_shift:+.1f}", flush=True)

    # Action entropy per category
    print("\n--- Action Variability ---", flush=True)
    for cat_name, data in results.items():
        if data['raw_actions']:
            unique = len(set(tuple(a) for a in data['raw_actions']))
            print(f"  {cat_name:12s}: {unique}/{len(data['raw_actions'])} unique actions, spread={data['spread']:.1f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'action_under_ood',
        'experiment_number': 126,
        'timestamp': timestamp,
        'action_token_base': ACTION_TOKEN_BASE,
        'dim_names': dim_names,
        'categories': {
            k: {kk: vv for kk, vv in v.items()}
            for k, v in results.items()
        },
    }
    if id_all:
        output['id_center'] = id_center.tolist()
        output['id_spread'] = id_spread

    output_path = os.path.join(RESULTS_DIR, f"action_under_ood_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
