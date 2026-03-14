"""
Multi-Head Attention Analysis.

Which attention heads are most diagnostic for OOD detection?
Tests per-head AUROC in the last layer (32 heads) to identify
which heads drive the aggregate attention max signal.

Experiment 74 in the CalibDrive series.
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


def main():
    print("=" * 70, flush=True)
    print("MULTI-HEAD ATTENTION ANALYSIS", flush=True)
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

    # Collect per-head attention for last layer
    all_data = []
    cnt = 0
    total = sum(v[2] for v in test_fns.values())

    for scene, (fn, is_ood, n) in test_fns.items():
        for i in range(n):
            cnt += 1
            img = Image.fromarray(fn(i + 900))
            inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)
            with torch.no_grad():
                fwd = model(**inputs, output_attentions=True)

            if hasattr(fwd, 'attentions') and fwd.attentions:
                attn = fwd.attentions[-1][0].float().cpu().numpy()  # [n_heads, seq, seq]
                n_heads = attn.shape[0]
                last_attn = attn[:, -1, :]  # [n_heads, seq]

                per_head_max = [float(np.max(last_attn[h])) for h in range(n_heads)]
                per_head_entropy = [float(-np.sum(
                    (last_attn[h]+1e-10) * np.log(last_attn[h]+1e-10)
                )) for h in range(n_heads)]

                all_data.append({
                    'scenario': scene,
                    'is_ood': is_ood,
                    'per_head_max': per_head_max,
                    'per_head_entropy': per_head_entropy,
                    'n_heads': n_heads,
                })

            if cnt % 10 == 0:
                print(f"  [{cnt}/{total}] {scene}", flush=True)

    print(f"\nCollected {len(all_data)} samples with {all_data[0]['n_heads']} heads.", flush=True)
    n_heads = all_data[0]['n_heads']

    # Per-head AUROC
    print("\n" + "=" * 70, flush=True)
    print("PER-HEAD RESULTS", flush=True)
    print("=" * 70, flush=True)

    id_data = [d for d in all_data if not d['is_ood']]
    ood_data = [d for d in all_data if d['is_ood']]
    labels = [0]*len(id_data) + [1]*len(ood_data)

    head_results = {}
    for h in range(n_heads):
        # Attn max for this head
        max_scores = [d['per_head_max'][h] for d in id_data] + [d['per_head_max'][h] for d in ood_data]
        ent_scores = [-d['per_head_entropy'][h] for d in id_data] + [-d['per_head_entropy'][h] for d in ood_data]

        auroc_max = roc_auc_score(labels, max_scores)
        auroc_ent = roc_auc_score(labels, ent_scores)

        id_max_mean = float(np.mean([d['per_head_max'][h] for d in id_data]))
        ood_max_mean = float(np.mean([d['per_head_max'][h] for d in ood_data]))

        head_results[h] = {
            'auroc_max': float(auroc_max),
            'auroc_entropy': float(auroc_ent),
            'id_max_mean': id_max_mean,
            'ood_max_mean': ood_max_mean,
        }

        if h % 4 == 0 or auroc_max > 0.9:
            print(f"  Head {h:>2}: max_AUROC={auroc_max:.3f}  ent_AUROC={auroc_ent:.3f}  "
                  f"ID={id_max_mean:.4f}  OOD={ood_max_mean:.4f}", flush=True)

    # Best and worst heads
    aurocs = [(h, head_results[h]['auroc_max']) for h in range(n_heads)]
    aurocs_sorted = sorted(aurocs, key=lambda x: -x[1])
    print(f"\n  Top-5 heads (by max AUROC):", flush=True)
    for h, a in aurocs_sorted[:5]:
        print(f"    Head {h}: AUROC={a:.3f}", flush=True)
    print(f"\n  Bottom-5 heads:", flush=True)
    for h, a in aurocs_sorted[-5:]:
        print(f"    Head {h}: AUROC={a:.3f}", flush=True)

    # Ensemble of top-k heads vs all heads
    print(f"\n  Ensemble analysis:", flush=True)
    for k in [1, 3, 5, 10, n_heads]:
        top_heads = [h for h, _ in aurocs_sorted[:k]]
        ensemble_scores = []
        for d in id_data + list(ood_data):
            score = np.mean([d['per_head_max'][h] for h in top_heads])
            ensemble_scores.append(score)
        auroc = roc_auc_score(labels, ensemble_scores)
        print(f"    Top-{k} heads: AUROC={auroc:.3f}", flush=True)

    # Head consistency across scenarios
    print(f"\n  Per-scenario head consistency:", flush=True)
    for scene in ['noise', 'indoor', 'blackout']:
        scene_data = [d for d in all_data if d['scenario'] == scene]
        scene_labels = [0]*len(id_data) + [1]*len(scene_data)
        best_h, best_auroc = -1, 0
        for h in range(n_heads):
            scores = [d['per_head_max'][h] for d in id_data] + [d['per_head_max'][h] for d in scene_data]
            auroc = roc_auc_score(scene_labels, scores)
            if auroc > best_auroc:
                best_h, best_auroc = h, auroc
        print(f"    {scene}: best_head={best_h}, AUROC={best_auroc:.3f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'head_analysis',
        'experiment_number': 74,
        'timestamp': timestamp,
        'n_heads': n_heads,
        'n_samples': len(all_data),
        'head_results': head_results,
        'top5_heads': [h for h, _ in aurocs_sorted[:5]],
        'bottom5_heads': [h for h, _ in aurocs_sorted[-5:]],
    }
    output_path = os.path.join(RESULTS_DIR, f"head_analysis_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
