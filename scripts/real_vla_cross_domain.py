"""
Cross-Domain Generalization.

Tests whether a centroid calibrated on one ID domain (highway-only or
urban-only) transfers to detect OOD when tested against a different ID
domain. This tests the assumption that "driving" forms a single cluster.

Experiment 106 in the CalibDrive series.
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


def extract_hidden(model, processor, image, prompt):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    if not hasattr(fwd, 'hidden_states') or not fwd.hidden_states:
        return None
    return fwd.hidden_states[-1][0, -1, :].float().cpu().numpy()


def cosine_dist(a, b):
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def evaluate_scenario(cal_embeds, test_embeds_dict):
    centroid = np.mean(cal_embeds, axis=0)
    results = {}
    all_scores = []
    all_labels = []
    for name, (embeds, label) in test_embeds_dict.items():
        scores = [cosine_dist(e, centroid) for e in embeds]
        results[name] = {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'scores': [float(s) for s in scores],
            'label': label,
        }
        all_scores.extend(scores)
        all_labels.extend([label] * len(scores))
    auroc = roc_auc_score(all_labels, all_scores)
    id_scores = [s for s, l in zip(all_scores, all_labels) if l == 0]
    ood_scores = [s for s, l in zip(all_scores, all_labels) if l == 1]
    d = (np.mean(ood_scores) - np.mean(id_scores)) / (np.std(id_scores) + 1e-10)
    return auroc, d, results


def main():
    print("=" * 70, flush=True)
    print("CROSS-DOMAIN GENERALIZATION", flush=True)
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

    print("\n--- Collecting embeddings ---", flush=True)
    embeddings = {}
    for cat_name, (fn, group) in categories.items():
        print(f"  {cat_name} ({group})...", flush=True)
        embeds = []
        for i in range(20):
            h = extract_hidden(model, processor, Image.fromarray(fn(i + 900)), prompt)
            if h is not None:
                embeds.append(h)
        embeddings[cat_name] = {'embeds': np.array(embeds), 'group': group}
        print(f"    Collected {len(embeds)} embeddings", flush=True)

    ood_cats = ['noise', 'indoor', 'twilight', 'snow']

    print("\n--- Scenario 1: Highway-Calibrated ---", flush=True)
    hw_cal = embeddings['highway']['embeds'][:10]
    test_dict_hw = {
        'highway': (embeddings['highway']['embeds'][10:], 0),
        'urban': (embeddings['urban']['embeds'], 0),
    }
    for c in ood_cats:
        test_dict_hw[c] = (embeddings[c]['embeds'], 1)
    auroc_hw, d_hw, res_hw = evaluate_scenario(hw_cal, test_dict_hw)
    print(f"  AUROC={auroc_hw:.4f}, d={d_hw:.2f}", flush=True)
    for name, r in res_hw.items():
        print(f"    {name}: {r['mean']:.4f}+/-{r['std']:.4f}", flush=True)

    print("\n--- Scenario 2: Urban-Calibrated ---", flush=True)
    urb_cal = embeddings['urban']['embeds'][:10]
    test_dict_urb = {
        'highway': (embeddings['highway']['embeds'], 0),
        'urban': (embeddings['urban']['embeds'][10:], 0),
    }
    for c in ood_cats:
        test_dict_urb[c] = (embeddings[c]['embeds'], 1)
    auroc_urb, d_urb, res_urb = evaluate_scenario(urb_cal, test_dict_urb)
    print(f"  AUROC={auroc_urb:.4f}, d={d_urb:.2f}", flush=True)
    for name, r in res_urb.items():
        print(f"    {name}: {r['mean']:.4f}+/-{r['std']:.4f}", flush=True)

    print("\n--- Scenario 3: Mixed-Calibrated (baseline) ---", flush=True)
    mix_cal = np.concatenate([embeddings['highway']['embeds'][:10],
                               embeddings['urban']['embeds'][:10]], axis=0)
    test_dict_mix = {
        'highway': (embeddings['highway']['embeds'][10:], 0),
        'urban': (embeddings['urban']['embeds'][10:], 0),
    }
    for c in ood_cats:
        test_dict_mix[c] = (embeddings[c]['embeds'], 1)
    auroc_mix, d_mix, res_mix = evaluate_scenario(mix_cal, test_dict_mix)
    print(f"  AUROC={auroc_mix:.4f}, d={d_mix:.2f}", flush=True)
    for name, r in res_mix.items():
        print(f"    {name}: {r['mean']:.4f}+/-{r['std']:.4f}", flush=True)

    print("\n--- Scenario 4: Highway->Urban Transfer ---", flush=True)
    test_dict_transfer = {
        'urban_as_id': (embeddings['urban']['embeds'], 0),
    }
    for c in ood_cats:
        test_dict_transfer[c] = (embeddings[c]['embeds'], 1)
    auroc_t1, d_t1, res_t1 = evaluate_scenario(hw_cal, test_dict_transfer)
    print(f"  AUROC={auroc_t1:.4f}, d={d_t1:.2f}", flush=True)
    for name, r in res_t1.items():
        print(f"    {name}: {r['mean']:.4f}+/-{r['std']:.4f}", flush=True)

    print("\n--- Scenario 5: Urban->Highway Transfer ---", flush=True)
    test_dict_transfer2 = {
        'highway_as_id': (embeddings['highway']['embeds'], 0),
    }
    for c in ood_cats:
        test_dict_transfer2[c] = (embeddings[c]['embeds'], 1)
    auroc_t2, d_t2, res_t2 = evaluate_scenario(urb_cal, test_dict_transfer2)
    print(f"  AUROC={auroc_t2:.4f}, d={d_t2:.2f}", flush=True)
    for name, r in res_t2.items():
        print(f"    {name}: {r['mean']:.4f}+/-{r['std']:.4f}", flush=True)

    hw_centroid = np.mean(embeddings['highway']['embeds'], axis=0)
    urb_centroid = np.mean(embeddings['urban']['embeds'], axis=0)
    inter_domain_dist = cosine_dist(hw_centroid, urb_centroid)
    print(f"\nInter-domain distance (highway<->urban): {inter_domain_dist:.4f}", flush=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'cross_domain',
        'experiment_number': 106,
        'timestamp': timestamp,
        'inter_domain_distance': inter_domain_dist,
        'scenarios': {
            'highway_calibrated': {
                'auroc': float(auroc_hw), 'd': float(d_hw),
                'per_category': {k: {'mean': v['mean'], 'std': v['std'], 'label': v['label']}
                                 for k, v in res_hw.items()},
            },
            'urban_calibrated': {
                'auroc': float(auroc_urb), 'd': float(d_urb),
                'per_category': {k: {'mean': v['mean'], 'std': v['std'], 'label': v['label']}
                                 for k, v in res_urb.items()},
            },
            'mixed_calibrated': {
                'auroc': float(auroc_mix), 'd': float(d_mix),
                'per_category': {k: {'mean': v['mean'], 'std': v['std'], 'label': v['label']}
                                 for k, v in res_mix.items()},
            },
            'highway_to_urban': {
                'auroc': float(auroc_t1), 'd': float(d_t1),
                'per_category': {k: {'mean': v['mean'], 'std': v['std'], 'label': v['label']}
                                 for k, v in res_t1.items()},
            },
            'urban_to_highway': {
                'auroc': float(auroc_t2), 'd': float(d_t2),
                'per_category': {k: {'mean': v['mean'], 'std': v['std'], 'label': v['label']}
                                 for k, v in res_t2.items()},
            },
        },
    }
    output_path = os.path.join(RESULTS_DIR, f"cross_domain_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
