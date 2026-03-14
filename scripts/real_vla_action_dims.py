"""
Action Dimension Analysis.

Examines which of the 7 action dimensions are most affected by OOD
inputs. OpenVLA predicts 7 action tokens (x, y, z, roll, pitch, yaw,
gripper) using 256-bin tokenization. Tests whether certain dimensions
are more sensitive to OOD than others.

Experiment 97 in the CalibDrive series.
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
ACTION_DIMS = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']


def create_highway(idx):
    rng = np.random.default_rng(idx * 5001)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]
    img[SIZE[0]//2:] = [80, 80, 80]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [255, 255, 255]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_urban(idx):
    rng = np.random.default_rng(idx * 5002)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [135, 206, 235]
    img[SIZE[0]//3:SIZE[0]//2] = [139, 119, 101]
    img[SIZE[0]//2:] = [60, 60, 60]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_noise(idx):
    rng = np.random.default_rng(idx * 5003)
    return rng.integers(0, 256, (*SIZE, 3), dtype=np.uint8)

def create_indoor(idx):
    rng = np.random.default_rng(idx * 5004)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:] = [200, 180, 160]
    img[SIZE[0]//2:, :] = [100, 80, 60]
    img[:SIZE[0]//3, SIZE[1]//3:2*SIZE[1]//3] = [150, 200, 255]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_twilight_highway(idx):
    rng = np.random.default_rng(idx * 5010)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [70, 50, 80]
    img[SIZE[0]//2:] = [60, 60, 60]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [200, 200, 100]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_snow(idx):
    rng = np.random.default_rng(idx * 5014)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [200, 200, 210]
    img[SIZE[0]//2:] = [220, 220, 230]
    img[SIZE[0]//2:, SIZE[1]//2-2:SIZE[1]//2+2] = [180, 180, 190]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def generate_actions(model, processor, image, prompt, n_tokens=7):
    """Generate action tokens and their logit distributions."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=n_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

    tokens = outputs.sequences[0, -n_tokens:].cpu().tolist()
    scores = []
    for i in range(min(n_tokens, len(outputs.scores))):
        logits = outputs.scores[i][0].float().cpu()
        probs = torch.softmax(logits, dim=0).numpy()
        entropy = -float(np.sum(probs * np.log(probs + 1e-10)))
        top_prob = float(np.max(probs))
        scores.append({
            'token': tokens[i] if i < len(tokens) else -1,
            'entropy': entropy,
            'top_prob': top_prob,
        })

    return tokens, scores


def main():
    print("=" * 70, flush=True)
    print("ACTION DIMENSION ANALYSIS", flush=True)
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

    all_results = {}
    for cat_name, (fn, group) in categories.items():
        print(f"\n  {cat_name} ({group})...", flush=True)
        cat_tokens = []
        cat_scores = []
        for i in range(8):
            tokens, scores = generate_actions(model, processor,
                                               Image.fromarray(fn(i + 500)), prompt)
            cat_tokens.append(tokens)
            cat_scores.append(scores)

        # Per-dimension statistics
        dim_stats = {}
        for d in range(min(7, len(cat_tokens[0]))):
            dim_tokens = [t[d] for t in cat_tokens if d < len(t)]
            dim_entropies = [s[d]['entropy'] for s in cat_scores if d < len(s)]
            dim_top_probs = [s[d]['top_prob'] for s in cat_scores if d < len(s)]

            dim_stats[ACTION_DIMS[d]] = {
                'token_mean': float(np.mean(dim_tokens)),
                'token_std': float(np.std(dim_tokens)),
                'token_values': dim_tokens,
                'entropy_mean': float(np.mean(dim_entropies)),
                'entropy_std': float(np.std(dim_entropies)),
                'top_prob_mean': float(np.mean(dim_top_probs)),
                'top_prob_std': float(np.std(dim_top_probs)),
            }

        all_results[cat_name] = {
            'group': group,
            'n_samples': len(cat_tokens),
            'dim_stats': dim_stats,
        }

        for dim_name, stats in dim_stats.items():
            print(f"    {dim_name}: token={stats['token_mean']:.1f}±{stats['token_std']:.1f}, "
                  f"entropy={stats['entropy_mean']:.3f}, top_prob={stats['top_prob_mean']:.3f}", flush=True)

    # Cross-category analysis per dimension
    print("\n--- Per-dimension ID vs OOD ---", flush=True)
    dim_analysis = {}
    for d, dim_name in enumerate(ACTION_DIMS):
        id_tokens = []
        ood_tokens = []
        id_entropies = []
        ood_entropies = []

        for cat_name, data in all_results.items():
            if dim_name not in data['dim_stats']:
                continue
            stats = data['dim_stats'][dim_name]
            if data['group'] == 'ID':
                id_tokens.extend(stats['token_values'])
                id_entropies.extend([stats['entropy_mean']] * len(stats['token_values']))
            else:
                ood_tokens.extend(stats['token_values'])
                ood_entropies.extend([stats['entropy_mean']] * len(stats['token_values']))

        # Token variance as OOD signal
        id_var = float(np.var(id_tokens)) if id_tokens else 0
        ood_var = float(np.var(ood_tokens)) if ood_tokens else 0

        # Token value AUROC (are OOD tokens different?)
        if id_tokens and ood_tokens:
            labels = [0]*len(id_tokens) + [1]*len(ood_tokens)
            # Use absolute deviation from ID mean as score
            id_mean = np.mean(id_tokens)
            id_scores = [abs(t - id_mean) for t in id_tokens]
            ood_scores = [abs(t - id_mean) for t in ood_tokens]
            auroc = float(roc_auc_score(labels, id_scores + ood_scores))
        else:
            auroc = 0.5

        dim_analysis[dim_name] = {
            'id_token_mean': float(np.mean(id_tokens)) if id_tokens else 0,
            'id_token_std': float(np.std(id_tokens)) if id_tokens else 0,
            'ood_token_mean': float(np.mean(ood_tokens)) if ood_tokens else 0,
            'ood_token_std': float(np.std(ood_tokens)) if ood_tokens else 0,
            'id_var': id_var,
            'ood_var': ood_var,
            'var_ratio': float(ood_var / (id_var + 1e-10)),
            'deviation_auroc': auroc,
        }
        print(f"  {dim_name}: ID={np.mean(id_tokens):.1f}±{np.std(id_tokens):.1f}, "
              f"OOD={np.mean(ood_tokens):.1f}±{np.std(ood_tokens):.1f}, "
              f"var_ratio={ood_var/(id_var+1e-10):.2f}, AUROC={auroc:.3f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'action_dims',
        'experiment_number': 97,
        'timestamp': timestamp,
        'categories': all_results,
        'dim_analysis': dim_analysis,
    }
    output_path = os.path.join(RESULTS_DIR, f"action_dims_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
