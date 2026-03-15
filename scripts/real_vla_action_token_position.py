"""
Action Token Position Analysis.

Analyzes which position in the 7-token action sequence carries the
strongest OOD signal. For each token position (x_t, y_t, z_t, x_r, y_r, z_r, grip),
we measure whether the action bin value alone can discriminate ID from OOD.

Key question: Is the OOD signal concentrated in specific action dimensions 
or distributed across all 7?

Experiment 136 in the CalibDrive series.
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
ACTION_TOKEN_BASE = 31744


def create_highway(idx):
    rng = np.random.default_rng(idx * 29001)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]
    img[SIZE[0]//2:] = [80, 80, 80]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [255, 255, 255]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_urban(idx):
    rng = np.random.default_rng(idx * 29002)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [135, 206, 235]
    img[SIZE[0]//3:SIZE[0]//2] = [139, 119, 101]
    img[SIZE[0]//2:] = [60, 60, 60]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_noise(idx):
    rng = np.random.default_rng(idx * 29003)
    return rng.integers(0, 256, (*SIZE, 3), dtype=np.uint8)

def create_indoor(idx):
    rng = np.random.default_rng(idx * 29004)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:] = [200, 180, 160]
    img[SIZE[0]//2:, :] = [100, 80, 60]
    img[:SIZE[0]//3, SIZE[1]//3:2*SIZE[1]//3] = [150, 200, 255]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_twilight(idx):
    rng = np.random.default_rng(idx * 29010)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [70, 50, 80]
    img[SIZE[0]//2:] = [60, 60, 60]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [200, 200, 100]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_snow(idx):
    rng = np.random.default_rng(idx * 29014)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [200, 200, 210]
    img[SIZE[0]//2:] = [220, 220, 230]
    img[SIZE[0]//2:, SIZE[1]//2-2:SIZE[1]//2+2] = [180, 180, 190]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def predict_action(model, processor, image, prompt):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    gen_ids = output[0, inputs['input_ids'].shape[1]:]
    action_bins = []
    for tid in gen_ids.tolist():
        if ACTION_TOKEN_BASE <= tid < ACTION_TOKEN_BASE + 256:
            action_bins.append(tid - ACTION_TOKEN_BASE)
    return action_bins


def main():
    print("=" * 70, flush=True)
    print("ACTION TOKEN POSITION ANALYSIS", flush=True)
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
    dim_names = ['x_trans', 'y_trans', 'z_trans', 'x_rot', 'y_rot', 'z_rot', 'gripper']

    categories = {
        'highway': (create_highway, 'ID'),
        'urban': (create_urban, 'ID'),
        'noise': (create_noise, 'OOD'),
        'indoor': (create_indoor, 'OOD'),
        'twilight': (create_twilight, 'OOD'),
        'snow': (create_snow, 'OOD'),
    }

    print("\n--- Collecting actions ---", flush=True)
    all_actions = []
    all_labels = []
    all_cats = []

    for cat_name, (fn, group) in categories.items():
        print(f"  {cat_name} ({group})...", flush=True)
        for i in range(15):
            img = Image.fromarray(fn(i + 3400))
            bins = predict_action(model, processor, img, prompt)
            if len(bins) >= 7:
                all_actions.append(bins[:7])
                all_labels.append(0 if group == 'ID' else 1)
                all_cats.append(cat_name)

    actions = np.array(all_actions, dtype=float)
    labels = np.array(all_labels)
    print(f"\nTotal: {len(actions)} valid actions ({sum(labels==0)} ID, {sum(labels==1)} OOD)", flush=True)

    # Per-dimension analysis
    print("\n--- Per-Dimension OOD Discrimination ---", flush=True)
    dim_results = {}
    for dim_idx in range(7):
        dim_vals = actions[:, dim_idx]
        id_vals = dim_vals[labels == 0]
        ood_vals = dim_vals[labels == 1]

        # Use absolute difference from ID mean as score
        id_mean = np.mean(id_vals)
        scores = np.abs(dim_vals - id_mean)
        auroc = float(roc_auc_score(labels, scores))
        d = float((np.mean(np.abs(ood_vals - id_mean)) - np.mean(np.abs(id_vals - id_mean))) /
                  (np.std(np.abs(id_vals - id_mean)) + 1e-10))

        dim_results[dim_names[dim_idx]] = {
            'dim_idx': dim_idx,
            'id_mean': float(np.mean(id_vals)),
            'id_std': float(np.std(id_vals)),
            'ood_mean': float(np.mean(ood_vals)),
            'ood_std': float(np.std(ood_vals)),
            'shift': float(np.mean(ood_vals) - np.mean(id_vals)),
            'auroc': auroc,
            'd_prime': d,
        }
        print(f"  {dim_names[dim_idx]:8s}: ID={np.mean(id_vals):.1f}±{np.std(id_vals):.1f}, "
              f"OOD={np.mean(ood_vals):.1f}±{np.std(ood_vals):.1f}, "
              f"shift={np.mean(ood_vals)-np.mean(id_vals):+.1f}, "
              f"AUROC={auroc:.3f}, d={d:.1f}", flush=True)

    # Combined action vector distance
    print("\n--- Combined Action Vector ---", flush=True)
    id_actions = actions[labels == 0]
    id_centroid = np.mean(id_actions, axis=0)
    scores_combined = np.array([np.linalg.norm(a - id_centroid) for a in actions])
    combined_auroc = float(roc_auc_score(labels, scores_combined))
    id_s = scores_combined[labels == 0]
    ood_s = scores_combined[labels == 1]
    combined_d = float((np.mean(ood_s) - np.mean(id_s)) / (np.std(id_s) + 1e-10))
    print(f"  Combined 7-dim L2: AUROC={combined_auroc:.4f}, d={combined_d:.1f}", flush=True)

    # Per-category action profiles
    print("\n--- Per-Category Profiles ---", flush=True)
    cat_profiles = {}
    for cat_name in set(all_cats):
        cat_mask = np.array([c == cat_name for c in all_cats])
        cat_actions = actions[cat_mask]
        cat_profiles[cat_name] = {
            'mean': cat_actions.mean(axis=0).tolist(),
            'std': cat_actions.std(axis=0).tolist(),
            'n': int(cat_mask.sum()),
        }
        print(f"  {cat_name:12s}: mean=[{', '.join(f'{v:.0f}' for v in cat_actions.mean(axis=0))}]", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'action_token_position',
        'experiment_number': 136,
        'timestamp': timestamp,
        'dim_names': dim_names,
        'n_total': len(actions),
        'n_id': int(sum(labels == 0)),
        'n_ood': int(sum(labels == 1)),
        'per_dimension': dim_results,
        'combined': {
            'auroc': combined_auroc,
            'd_prime': combined_d,
        },
        'per_category': cat_profiles,
        'id_centroid': id_centroid.tolist(),
    }
    output_path = os.path.join(RESULTS_DIR, f"action_token_position_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
