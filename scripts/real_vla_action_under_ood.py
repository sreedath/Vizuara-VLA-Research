"""
Action Output Analysis Under OOD.

Analyzes the actual predicted actions (7-dim action vectors) for ID
vs OOD inputs. Key questions:
1. Are OOD actions random or systematic?
2. Are specific action dimensions more affected?
3. Do OOD actions cluster or scatter?
4. How dangerous are the predicted OOD actions?

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
    """Get the predicted action tokens and decode to action vector."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
        )
    # Decode generated tokens
    gen_ids = output[0, inputs['input_ids'].shape[1]:]
    gen_text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)

    # Try to extract action values from token IDs
    # OpenVLA uses 256-bin action tokenization with special tokens 32000-32255
    action_token_ids = []
    for tid in gen_ids.tolist():
        if 32000 <= tid <= 32255:
            action_token_ids.append(tid - 32000)

    return {
        'text': gen_text,
        'token_ids': gen_ids.tolist(),
        'action_bins': action_token_ids,
        'n_action_tokens': len(action_token_ids),
    }


def main():
    print("=" * 70, flush=True)
    print("ACTION OUTPUT ANALYSIS UNDER OOD", flush=True)
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

    print("\n--- Predicting actions ---", flush=True)
    results = {}
    for cat_name, (fn, group) in categories.items():
        print(f"\n  {cat_name} ({group}):", flush=True)
        actions = []
        for i in range(10):
            img = Image.fromarray(fn(i + 2600))
            act = predict_action(model, processor, img, prompt)
            actions.append(act)
            print(f"    {i}: bins={act['action_bins'][:7]}, n_tokens={act['n_action_tokens']}, text={act['text'][:60]}", flush=True)

        # Analyze action distributions
        all_bins = [a['action_bins'] for a in actions if len(a['action_bins']) >= 7]
        n_tokens_list = [a['n_action_tokens'] for a in actions]

        if all_bins:
            bin_array = np.array([b[:7] for b in all_bins])  # first 7 dims
            per_dim_mean = bin_array.mean(axis=0).tolist()
            per_dim_std = bin_array.std(axis=0).tolist()
            per_dim_min = bin_array.min(axis=0).tolist()
            per_dim_max = bin_array.max(axis=0).tolist()

            # Action consistency (how similar are actions within this category?)
            if len(bin_array) > 1:
                centroid = bin_array.mean(axis=0)
                dists = [np.linalg.norm(b - centroid) for b in bin_array]
                consistency = float(1.0 / (np.mean(dists) + 1e-10))
            else:
                consistency = float('inf')
        else:
            per_dim_mean = per_dim_std = per_dim_min = per_dim_max = []
            consistency = 0.0

        results[cat_name] = {
            'group': group,
            'n_samples': len(actions),
            'n_with_7_actions': len(all_bins),
            'n_tokens_mean': float(np.mean(n_tokens_list)),
            'n_tokens_std': float(np.std(n_tokens_list)),
            'per_dim_mean': per_dim_mean,
            'per_dim_std': per_dim_std,
            'per_dim_min': per_dim_min,
            'per_dim_max': per_dim_max,
            'consistency': consistency,
            'raw_actions': [a['action_bins'][:7] for a in actions],
            'raw_texts': [a['text'][:80] for a in actions],
        }

    # Cross-category comparison
    print("\n--- Cross-Category Comparison ---", flush=True)
    dim_names = ['x_trans', 'y_trans', 'z_trans', 'x_rot', 'y_rot', 'z_rot', 'gripper']

    for cat_name, data in results.items():
        if data['per_dim_mean']:
            means_str = ", ".join(f"{d:.1f}" for d in data['per_dim_mean'])
            stds_str = ", ".join(f"{d:.1f}" for d in data['per_dim_std'])
            print(f"  {cat_name:12s}: mean=[{means_str}], std=[{stds_str}], consistency={data['consistency']:.2f}", flush=True)

    # ID vs OOD action divergence
    print("\n--- ID vs OOD Divergence ---", flush=True)
    id_actions = []
    ood_actions = {}
    for cat_name, data in results.items():
        if data['group'] == 'ID' and data['per_dim_mean']:
            id_actions.extend([a for a in data['raw_actions'] if len(a) >= 7])
        elif data['group'] == 'OOD' and data['per_dim_mean']:
            ood_actions[cat_name] = [a for a in data['raw_actions'] if len(a) >= 7]

    if id_actions:
        id_arr = np.array([a[:7] for a in id_actions])
        id_center = id_arr.mean(axis=0)

        for cat_name, ood_acts in ood_actions.items():
            if ood_acts:
                ood_arr = np.array([a[:7] for a in ood_acts])
                ood_center = ood_arr.mean(axis=0)
                divergence = np.linalg.norm(ood_center - id_center)
                per_dim_diff = (ood_center - id_center).tolist()
                print(f"  {cat_name}: divergence={divergence:.2f}, per_dim_diff={[f'{d:.1f}' for d in per_dim_diff]}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'action_under_ood',
        'experiment_number': 126,
        'timestamp': timestamp,
        'categories': {k: {kk: vv for kk, vv in v.items() if kk != 'raw_texts'} for k, v in results.items()},
        'dim_names': dim_names,
    }
    output_path = os.path.join(RESULTS_DIR, f"action_under_ood_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
