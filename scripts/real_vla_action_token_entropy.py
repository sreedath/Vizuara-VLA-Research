"""
Action Token Entropy and Distribution Shape Analysis.

Examines how the first action token distribution changes between ID
and OOD inputs. Uses the full vocabulary logits rather than assuming
specific action token IDs.

Tests:
1. Full vocabulary entropy comparison (ID vs OOD)
2. Top-1 token concentration (peakiness)
3. Per-scenario entropy profiles

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


def extract_logit_features(model, processor, image, prompt):
    """Extract distribution features from the output logits."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)

    with torch.no_grad():
        fwd = model(**inputs)

    logits = fwd.logits[0, -1, :].float().cpu().numpy()
    vocab_size = len(logits)

    # Full vocabulary softmax
    logits_shifted = logits - logits.max()
    probs = np.exp(logits_shifted) / np.sum(np.exp(logits_shifted))

    # Entropy
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    max_entropy = np.log(vocab_size)
    normalized_entropy = entropy / max_entropy

    # Top-k concentration
    sorted_probs = np.sort(probs)[::-1]
    top1 = float(sorted_probs[0])
    top5 = float(np.sum(sorted_probs[:5]))
    top10 = float(np.sum(sorted_probs[:10]))
    top50 = float(np.sum(sorted_probs[:50]))

    # Energy score
    energy = float(np.log(np.sum(np.exp(logits_shifted))) + logits.max())

    # Logit statistics
    logit_mean = float(np.mean(logits))
    logit_std = float(np.std(logits))
    logit_max = float(np.max(logits))
    logit_min = float(np.min(logits))

    # Top token ID
    top_id = int(np.argmax(probs))

    return {
        'entropy': float(entropy),
        'norm_entropy': float(normalized_entropy),
        'top1': top1,
        'top5': top5,
        'top10': top10,
        'top50': top50,
        'energy': energy,
        'logit_mean': logit_mean,
        'logit_std': logit_std,
        'logit_max': logit_max,
        'logit_min': logit_min,
        'top_token_id': top_id,
        'vocab_size': vocab_size,
    }


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
        'highway': (create_highway, False, 10),
        'urban': (create_urban, False, 10),
        'noise': (create_noise, True, 8),
        'indoor': (create_indoor, True, 8),
        'blackout': (create_blackout, True, 6),
    }

    all_data = []
    cnt = 0
    total = sum(v[2] for v in test_fns.values())
    for scene, (fn, is_ood, n) in test_fns.items():
        for i in range(n):
            cnt += 1
            feats = extract_logit_features(model, processor,
                                           Image.fromarray(fn(i + 800)), prompt)
            feats['scenario'] = scene
            feats['is_ood'] = is_ood
            all_data.append(feats)
            if cnt % 5 == 0:
                print(f"  [{cnt}/{total}] {scene}: entropy={feats['entropy']:.2f} "
                      f"top1={feats['top1']:.4f} top_id={feats['top_token_id']}",
                      flush=True)

    print(f"\nCollected {len(all_data)} samples.", flush=True)

    # Analysis
    print("\n" + "=" * 70, flush=True)
    print("RESULTS", flush=True)
    print("=" * 70, flush=True)

    id_data = [d for d in all_data if not d['is_ood']]
    ood_data = [d for d in all_data if d['is_ood']]

    # Per-metric AUROC
    print("\n  Detection AUROC by logit feature:", flush=True)
    results = {}
    labels = [0]*len(id_data) + [1]*len(ood_data)

    for metric in ['entropy', 'norm_entropy', 'top1', 'top5', 'top10',
                    'energy', 'logit_std', 'logit_max']:
        if metric in ['top1', 'top5', 'top10']:
            # Lower top-k = more uncertain = OOD
            scores = [-d[metric] for d in id_data] + [-d[metric] for d in ood_data]
        elif metric == 'energy':
            scores = [-d[metric] for d in id_data] + [-d[metric] for d in ood_data]
        else:
            scores = [d[metric] for d in id_data] + [d[metric] for d in ood_data]

        auroc = roc_auc_score(labels, scores)
        results[metric] = float(auroc)
        print(f"    {metric:<15}: AUROC={auroc:.3f}", flush=True)

    # Per-scenario
    print("\n  Per-scenario statistics:", flush=True)
    print(f"    {'Scene':<12} {'Entropy':>8} {'Top1':>8} {'Top5':>8} {'Energy':>8} {'LogStd':>8}",
          flush=True)
    per_scenario = {}
    for scene in ['highway', 'urban', 'noise', 'indoor', 'blackout']:
        sd = [d for d in all_data if d['scenario'] == scene]
        per_scenario[scene] = {
            'entropy': float(np.mean([d['entropy'] for d in sd])),
            'top1': float(np.mean([d['top1'] for d in sd])),
            'top5': float(np.mean([d['top5'] for d in sd])),
            'energy': float(np.mean([d['energy'] for d in sd])),
            'logit_std': float(np.mean([d['logit_std'] for d in sd])),
        }
        print(f"    {scene:<12} {per_scenario[scene]['entropy']:>8.2f} "
              f"{per_scenario[scene]['top1']:>8.4f} {per_scenario[scene]['top5']:>8.4f} "
              f"{per_scenario[scene]['energy']:>8.2f} {per_scenario[scene]['logit_std']:>8.3f}",
              flush=True)

    # Top token analysis
    print("\n  Most common top token IDs:", flush=True)
    for group_name, group in [('ID', id_data), ('OOD', ood_data)]:
        top_ids = [d['top_token_id'] for d in group]
        unique, counts = np.unique(top_ids, return_counts=True)
        sorted_idx = np.argsort(-counts)
        top3 = [(int(unique[i]), int(counts[i])) for i in sorted_idx[:3]]
        print(f"    {group_name}: {top3}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'action_token_entropy',
        'experiment_number': 73,
        'timestamp': timestamp,
        'n_samples': len(all_data),
        'auroc_by_metric': results,
        'per_scenario': per_scenario,
    }
    output_path = os.path.join(RESULTS_DIR, f"action_token_entropy_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
