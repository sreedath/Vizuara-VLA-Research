"""
Action Token Entropy and Distribution Shape Analysis.

Examines how the action token distribution (256-bin) changes
between ID and OOD inputs across all 7 action dimensions.

Tests:
1. Per-dimension entropy comparison (ID vs OOD)
2. Top-1 token concentration (peakiness)
3. Distribution shape (uniform-like vs peaked)
4. Cross-dimension correlation

Experiment 73 in the CalibDrive series.
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
ACTION_TOKEN_IDS = list(range(32000, 32256))  # OpenVLA action bins


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

def create_blackout(idx):
    return np.zeros((*SIZE, 3), dtype=np.uint8)


def extract_action_dist(model, processor, image, prompt, n_tokens=7):
    """Extract per-dimension action token distributions via autoregressive generation."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    input_ids = inputs['input_ids']
    attention_mask = inputs.get('attention_mask', None)

    distributions = []
    generated_ids = input_ids.clone()
    gen_attn = attention_mask.clone() if attention_mask is not None else None

    for dim in range(n_tokens):
        with torch.no_grad():
            outputs = model(input_ids=generated_ids, attention_mask=gen_attn)

        logits = outputs.logits[0, -1, :]  # Last token logits
        # Extract action token logits only
        action_logits = logits[ACTION_TOKEN_IDS].float().cpu().numpy()
        # Softmax over action tokens
        action_logits_shifted = action_logits - action_logits.max()
        probs = np.exp(action_logits_shifted) / np.sum(np.exp(action_logits_shifted))

        entropy = -np.sum(probs * np.log(probs + 1e-10))
        top1 = float(np.max(probs))
        top5 = float(np.sum(np.sort(probs)[-5:]))
        argmax_bin = int(np.argmax(probs))

        distributions.append({
            'entropy': float(entropy),
            'top1': top1,
            'top5': top5,
            'argmax_bin': argmax_bin,
            'probs': probs.tolist(),
        })

        # Greedy select and append
        next_token = torch.tensor([[ACTION_TOKEN_IDS[argmax_bin]]], device=model.device)
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        if gen_attn is not None:
            gen_attn = torch.cat([gen_attn, torch.ones(1, 1, device=model.device, dtype=gen_attn.dtype)], dim=1)

    return distributions


def main():
    print("=" * 70, flush=True)
    print("ACTION TOKEN ENTROPY ANALYSIS", flush=True)
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

    test_fns = {
        'highway': (create_highway, False, 8),
        'urban': (create_urban, False, 8),
        'noise': (create_noise, True, 6),
        'indoor': (create_indoor, True, 6),
        'blackout': (create_blackout, True, 4),
    }

    all_data = []
    cnt = 0
    total = sum(v[2] for v in test_fns.values())
    for scene, (fn, is_ood, n) in test_fns.items():
        for i in range(n):
            cnt += 1
            dists = extract_action_dist(model, processor,
                                        Image.fromarray(fn(i + 800)), prompt)
            record = {
                'scenario': scene,
                'is_ood': is_ood,
                'dims': dists,
            }
            all_data.append(record)
            if cnt % 5 == 0:
                print(f"  [{cnt}/{total}] {scene}", flush=True)

    print(f"\nCollected {len(all_data)} samples.", flush=True)

    # Analysis
    print("\n" + "=" * 70, flush=True)
    print("RESULTS", flush=True)
    print("=" * 70, flush=True)

    # Per-dimension entropy
    print("\n  Per-dimension entropy (ID vs OOD):", flush=True)
    dim_results = {}
    for dim in range(7):
        id_entropies = [d['dims'][dim]['entropy'] for d in all_data if not d['is_ood']]
        ood_entropies = [d['dims'][dim]['entropy'] for d in all_data if d['is_ood']]
        id_top1 = [d['dims'][dim]['top1'] for d in all_data if not d['is_ood']]
        ood_top1 = [d['dims'][dim]['top1'] for d in all_data if d['is_ood']]

        # AUROC using entropy
        labels = [0]*len(id_entropies) + [1]*len(ood_entropies)
        scores = id_entropies + ood_entropies
        auroc_ent = roc_auc_score(labels, scores)

        # AUROC using -top1 (lower top1 = more OOD)
        scores_top1 = [-t for t in id_top1] + [-t for t in ood_top1]
        auroc_top1 = roc_auc_score(labels, scores_top1)

        dim_results[dim] = {
            'id_entropy_mean': float(np.mean(id_entropies)),
            'id_entropy_std': float(np.std(id_entropies)),
            'ood_entropy_mean': float(np.mean(ood_entropies)),
            'ood_entropy_std': float(np.std(ood_entropies)),
            'id_top1_mean': float(np.mean(id_top1)),
            'ood_top1_mean': float(np.mean(ood_top1)),
            'auroc_entropy': float(auroc_ent),
            'auroc_top1': float(auroc_top1),
        }
        print(f"    Dim {dim}: ID_ent={np.mean(id_entropies):.3f} "
              f"OOD_ent={np.mean(ood_entropies):.3f} "
              f"AUROC_ent={auroc_ent:.3f} AUROC_top1={auroc_top1:.3f}", flush=True)

    # Aggregated entropy across all dims
    print("\n  Aggregated (mean across dims):", flush=True)
    id_agg = [np.mean([d['dims'][dim]['entropy'] for dim in range(7)]) for d in all_data if not d['is_ood']]
    ood_agg = [np.mean([d['dims'][dim]['entropy'] for dim in range(7)]) for d in all_data if d['is_ood']]
    labels_agg = [0]*len(id_agg) + [1]*len(ood_agg)
    auroc_agg = roc_auc_score(labels_agg, id_agg + ood_agg)
    print(f"    Mean entropy AUROC: {auroc_agg:.3f}", flush=True)
    print(f"    ID: {np.mean(id_agg):.3f}±{np.std(id_agg):.3f}", flush=True)
    print(f"    OOD: {np.mean(ood_agg):.3f}±{np.std(ood_agg):.3f}", flush=True)

    # Per-scenario
    print("\n  Per-scenario mean entropy:", flush=True)
    for scene in ['highway', 'urban', 'noise', 'indoor', 'blackout']:
        ents = [np.mean([d['dims'][dim]['entropy'] for dim in range(7)])
                for d in all_data if d['scenario'] == scene]
        print(f"    {scene:<12}: {np.mean(ents):.3f}±{np.std(ents):.3f}", flush=True)

    # Save (without full probability distributions to save space)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'action_token_entropy',
        'experiment_number': 73,
        'timestamp': timestamp,
        'n_samples': len(all_data),
        'dim_results': dim_results,
        'aggregated': {
            'auroc': float(auroc_agg),
            'id_mean': float(np.mean(id_agg)),
            'ood_mean': float(np.mean(ood_agg)),
        },
        'per_scenario': {
            scene: {
                'mean_entropy': float(np.mean([
                    np.mean([d['dims'][dim]['entropy'] for dim in range(7)])
                    for d in all_data if d['scenario'] == scene
                ])),
            }
            for scene in ['highway', 'urban', 'noise', 'indoor', 'blackout']
        },
    }
    output_path = os.path.join(RESULTS_DIR, f"action_token_entropy_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
