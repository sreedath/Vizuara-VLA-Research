"""
Deep Action Token Distribution Analysis.

Examines how the probability distribution over 256 action bins
changes between ID and OOD inputs. Provides mechanistic insight
into WHY action mass leaks and HOW the model processes OOD inputs.

Experiment 57 in the CalibDrive series.
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
    img[SIZE[0]//2:] = [139, 90, 43]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_inverted(idx):
    return 255 - create_highway(idx + 3000)

def create_blackout(idx):
    return np.zeros((*SIZE, 3), dtype=np.uint8)


def extract_action_dist(model, processor, image, prompt):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=7, do_sample=False,
            output_scores=True, return_dict_in_generate=True,
        )
    vocab_size = outputs.scores[0].shape[-1]
    action_start = vocab_size - 256
    per_dim = []
    for score in outputs.scores[:7]:
        logits = score[0].float()
        probs = torch.softmax(logits, dim=0)
        action_probs = probs[action_start:].cpu().numpy()
        full_probs = probs.cpu().numpy()
        action_mass = float(action_probs.sum())
        entropy = float(-np.sum(action_probs * np.log(action_probs + 1e-10)))
        argmax = int(action_probs.argmax())
        sorted_p = np.sort(action_probs)[::-1]
        non_action = full_probs[:action_start]
        top_garbage = float(np.sort(non_action)[-5:].sum())
        per_dim.append({
            'mass': action_mass, 'entropy': entropy, 'argmax': argmax,
            'top1': float(sorted_p[0]), 'top5': float(sorted_p[:5].sum()),
            'top10': float(sorted_p[:10].sum()), 'garbage': top_garbage,
        })
    return per_dim


def main():
    print("=" * 70, flush=True)
    print("ACTION TOKEN DISTRIBUTION ANALYSIS", flush=True)
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
        'inverted': (create_inverted, True, 6),
        'blackout': (create_blackout, True, 6),
    }

    all_data = []
    cnt = 0
    total = sum(v[2] for v in test_fns.values())

    for scene, (fn, is_ood, n) in test_fns.items():
        for i in range(n):
            cnt += 1
            per_dim = extract_action_dist(model, processor,
                                           Image.fromarray(fn(i + 200)), prompt)
            all_data.append({'scenario': scene, 'is_ood': is_ood, 'per_dim': per_dim})
            if cnt % 10 == 0:
                print(f"  [{cnt}/{total}] {scene}_{i}", flush=True)

    easy = [d for d in all_data if not d['is_ood']]
    ood = [d for d in all_data if d['is_ood']]

    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    # Per-dimension stats
    for dim in range(7):
        id_mass = [d['per_dim'][dim]['mass'] for d in easy]
        ood_mass = [d['per_dim'][dim]['mass'] for d in ood]
        id_ent = [d['per_dim'][dim]['entropy'] for d in easy]
        ood_ent = [d['per_dim'][dim]['entropy'] for d in ood]
        id_top1 = [d['per_dim'][dim]['top1'] for d in easy]
        ood_top1 = [d['per_dim'][dim]['top1'] for d in ood]
        print(f"\n  Dim {dim}:", flush=True)
        print(f"    Mass: ID={np.mean(id_mass):.3f}±{np.std(id_mass):.3f}, "
              f"OOD={np.mean(ood_mass):.3f}±{np.std(ood_mass):.3f}", flush=True)
        print(f"    Entropy: ID={np.mean(id_ent):.3f}±{np.std(id_ent):.3f}, "
              f"OOD={np.mean(ood_ent):.3f}±{np.std(ood_ent):.3f}", flush=True)
        print(f"    Top-1: ID={np.mean(id_top1):.3f}±{np.std(id_top1):.3f}, "
              f"OOD={np.mean(ood_top1):.3f}±{np.std(ood_top1):.3f}", flush=True)

    # Aggregate AUROC per metric
    print("\n  Aggregate AUROC (mean over 7 dims):", flush=True)
    metrics = ['mass', 'entropy', 'top1', 'top5', 'top10', 'garbage']
    labels = [0]*len(easy) + [1]*len(ood)
    for metric in metrics:
        id_v = [np.mean([d['per_dim'][dim][metric] for dim in range(7)]) for d in easy]
        ood_v = [np.mean([d['per_dim'][dim][metric] for dim in range(7)]) for d in ood]
        all_v = id_v + ood_v
        # Higher mass/top1 = more ID, lower = more OOD
        if metric in ['mass', 'top1', 'top5', 'top10']:
            scores = [-v for v in all_v]
        else:
            scores = all_v
        auroc = roc_auc_score(labels, scores)
        print(f"    {metric}: ID={np.mean(id_v):.4f}, OOD={np.mean(ood_v):.4f}, AUROC={auroc:.3f}",
              flush=True)

    # Argmax consistency
    print("\n  Argmax consistency:", flush=True)
    for name, group in [('ID', easy), ('OOD', ood)]:
        argmaxes = np.array([[d['per_dim'][dim]['argmax'] for dim in range(7)] for d in group])
        std_per_dim = np.std(argmaxes, axis=0)
        print(f"    {name} argmax std per dim: {std_per_dim.tolist()}", flush=True)
        print(f"    {name} mean std: {np.mean(std_per_dim):.1f}", flush=True)

    # Per-scenario summary
    print("\n  Per-scenario action mass:", flush=True)
    for scene in sorted(set(d['scenario'] for d in all_data)):
        s_data = [d for d in all_data if d['scenario'] == scene]
        mass = np.mean([np.mean([d['per_dim'][dim]['mass'] for dim in range(7)]) for d in s_data])
        garb = np.mean([np.mean([d['per_dim'][dim]['garbage'] for dim in range(7)]) for d in s_data])
        print(f"    {scene}: mass={mass:.3f}, garbage={garb:.4f}", flush=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    scenario_stats = {}
    for scene in sorted(set(d['scenario'] for d in all_data)):
        s_data = [d for d in all_data if d['scenario'] == scene]
        stats = {}
        for m in metrics:
            vals = [np.mean([d['per_dim'][dim][m] for dim in range(7)]) for d in s_data]
            stats[m] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
        scenario_stats[scene] = stats
    output = {
        'experiment': 'action_dist_deep', 'experiment_number': 57,
        'timestamp': timestamp, 'n_id': len(easy), 'n_ood': len(ood),
        'scenario_stats': scenario_stats,
    }
    output_path = os.path.join(RESULTS_DIR, f"action_dist_deep_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
