"""
Multi-Centroid and Prototype-Based OOD Detection on Real OpenVLA-7B.

Tests whether using multiple centroids (one per driving scenario) improves
OOD detection compared to a single global centroid.

Methods tested:
1. Single centroid (baseline): min cosine distance to global mean
2. Per-scene centroids: min cosine distance to nearest scene centroid
3. Cluster prototypes: k-means with k=2,3,5 on calibration set
4. Max cosine similarity: use max similarity to any calibration sample
5. Quantile-based: use 5th percentile of similarities to calibration set

Also tests on expanded OOD set including checker pattern.

Experiment 36 in the CalibDrive series.
"""
import os
import json
import time
import datetime
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

RESULTS_DIR = "/workspace/Vizuara-VLA-Research/experiments"
os.makedirs(RESULTS_DIR, exist_ok=True)

SCENARIOS = {
    'highway': {'n': 25, 'speed': '30', 'difficulty': 'easy'},
    'urban': {'n': 25, 'speed': '15', 'difficulty': 'easy'},
    'ood_noise': {'n': 12, 'speed': '25', 'difficulty': 'ood'},
    'ood_blank': {'n': 12, 'speed': '25', 'difficulty': 'ood'},
    'ood_indoor': {'n': 12, 'speed': '25', 'difficulty': 'ood'},
    'ood_inverted': {'n': 12, 'speed': '30', 'difficulty': 'ood'},
    'ood_checker': {'n': 12, 'speed': '25', 'difficulty': 'ood'},
    'ood_blackout': {'n': 12, 'speed': '25', 'difficulty': 'ood'},
}


def create_scene_image(scenario, idx, size=(256, 256)):
    np.random.seed(idx * 3600 + hash(scenario) % 36000)
    if scenario == 'highway':
        img = np.zeros((*size, 3), dtype=np.uint8)
        img[:size[0]//2] = [135, 206, 235]
        img[size[0]//2:] = [80, 80, 80]
    elif scenario == 'urban':
        img = np.zeros((*size, 3), dtype=np.uint8)
        img[:size[0]//3] = [135, 206, 235]
        img[size[0]//3:size[0]//2] = [139, 119, 101]
        img[size[0]//2:] = [80, 80, 80]
    elif scenario == 'ood_noise':
        img = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
    elif scenario == 'ood_blank':
        img = np.full((*size, 3), 128, dtype=np.uint8)
    elif scenario == 'ood_indoor':
        img = np.zeros((*size, 3), dtype=np.uint8)
        img[:size[0]//3] = [210, 180, 140]
        img[size[0]//3:2*size[0]//3] = [180, 120, 80]
        img[2*size[0]//3:] = [100, 70, 50]
    elif scenario == 'ood_inverted':
        img = np.zeros((*size, 3), dtype=np.uint8)
        img[:size[0]//2] = [135, 206, 235]
        img[size[0]//2:] = [80, 80, 80]
        img = 255 - img
    elif scenario == 'ood_checker':
        img = np.zeros((*size, 3), dtype=np.uint8)
        block = 32
        for y in range(0, size[0], block):
            for x in range(0, size[1], block):
                if (y // block + x // block) % 2 == 0:
                    img[y:y+block, x:x+block] = [255, 255, 255]
    elif scenario == 'ood_blackout':
        img = np.full((*size, 3), 5, dtype=np.uint8)
    else:
        img = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
    noise = np.random.randint(-3, 3, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def cosine_dist(a, b):
    a_n = a / (np.linalg.norm(a) + 1e-10)
    b_n = b / (np.linalg.norm(b) + 1e-10)
    return 1.0 - float(np.dot(a_n, b_n))


def main():
    print("=" * 70, flush=True)
    print("MULTI-CENTROID OOD DETECTION", flush=True)
    print("=" * 70, flush=True)

    from transformers import AutoModelForVision2Seq, AutoProcessor
    print("Loading OpenVLA-7B...", flush=True)
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        "openvla/openvla-7b",
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.", flush=True)

    prompt = "In: What action should the robot take to drive forward at {speed} m/s safely?\nOut:"
    total = sum(s['n'] for s in SCENARIOS.values())
    print(f"Total samples: {total}", flush=True)

    all_samples = []
    all_hidden = []
    sample_idx = 0

    for scenario, config in SCENARIOS.items():
        for i in range(config['n']):
            sample_idx += 1
            image = create_scene_image(scenario, i)
            p = prompt.format(speed=config['speed'])
            inputs = processor(p, image).to(model.device, dtype=torch.bfloat16)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=7,
                    do_sample=False,
                    output_scores=True,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                )

            # Hidden state
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                last_step = outputs.hidden_states[-1]
                if isinstance(last_step, tuple):
                    hidden = last_step[-1][0, -1, :].float().cpu().numpy()
                else:
                    hidden = last_step[0, -1, :].float().cpu().numpy()
            else:
                hidden = np.zeros(4096)

            # Action mass
            vocab_size = outputs.scores[0].shape[-1]
            action_start = vocab_size - 256
            dim_masses = []
            for score in outputs.scores[:7]:
                full_logits = score[0].float()
                full_probs = torch.softmax(full_logits, dim=0).cpu().numpy()
                action_probs = full_probs[action_start:]
                dim_masses.append(float(action_probs.sum()))

            sample = {
                'scenario': scenario,
                'difficulty': config['difficulty'],
                'idx': i,
                'action_mass': float(np.mean(dim_masses)),
            }
            all_samples.append(sample)
            all_hidden.append(hidden)

            if sample_idx % 20 == 0 or sample_idx == total:
                print(f"  [{sample_idx}/{total}] {scenario}_{i}: "
                      f"mass={sample['action_mass']:.4f}", flush=True)

    hidden_arr = np.array(all_hidden)

    # Split: first half easy = calibration, second half = test
    easy_idxs = [i for i, s in enumerate(all_samples) if s['difficulty'] == 'easy']
    ood_idxs = [i for i, s in enumerate(all_samples) if s['difficulty'] == 'ood']

    np.random.seed(42)
    np.random.shuffle(easy_idxs)
    cal_easy = easy_idxs[:len(easy_idxs)//2]
    test_easy = easy_idxs[len(easy_idxs)//2:]

    highway_cal = [i for i in cal_easy if all_samples[i]['scenario'] == 'highway']
    urban_cal = [i for i in cal_easy if all_samples[i]['scenario'] == 'urban']

    print(f"\nCalibration: {len(cal_easy)} ({len(highway_cal)} hwy, {len(urban_cal)} urban)")
    print(f"Test easy: {len(test_easy)}, OOD: {len(ood_idxs)}", flush=True)

    # ===================================================================
    # Method 1: Single global centroid (baseline)
    # ===================================================================
    print("\n" + "=" * 70, flush=True)
    print("METHOD COMPARISON", flush=True)
    print("=" * 70, flush=True)

    # 1. Global centroid
    global_mean = np.mean(hidden_arr[cal_easy], axis=0)
    global_norm = global_mean / (np.linalg.norm(global_mean) + 1e-10)
    global_cos = [cosine_dist(hidden_arr[i], global_mean) for i in range(len(all_samples))]

    # 2. Per-scene centroids (min distance to nearest)
    hwy_mean = np.mean(hidden_arr[highway_cal], axis=0)
    urb_mean = np.mean(hidden_arr[urban_cal], axis=0)
    scene_cos = [min(cosine_dist(hidden_arr[i], hwy_mean),
                     cosine_dist(hidden_arr[i], urb_mean))
                 for i in range(len(all_samples))]

    # 3. K-means prototypes (k=2, 3, 5)
    kmeans_cos = {}
    for k in [2, 3, 5]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(hidden_arr[cal_easy])
        centroids = km.cluster_centers_
        k_cos = [min(cosine_dist(hidden_arr[i], c) for c in centroids)
                 for i in range(len(all_samples))]
        kmeans_cos[k] = k_cos

    # 4. Max similarity to any calibration sample
    max_sim = []
    for i in range(len(all_samples)):
        sims = [1.0 - cosine_dist(hidden_arr[i], hidden_arr[j]) for j in cal_easy]
        max_sim.append(1.0 - max(sims))  # convert to distance

    # 5. 5th percentile similarity to calibration set
    quantile_dist = []
    for i in range(len(all_samples)):
        dists = [cosine_dist(hidden_arr[i], hidden_arr[j]) for j in cal_easy]
        quantile_dist.append(np.percentile(dists, 5))  # 5th pctl = closest

    # 6. Mean of top-3 nearest (kNN k=3)
    knn3_dist = []
    for i in range(len(all_samples)):
        dists = sorted([cosine_dist(hidden_arr[i], hidden_arr[j]) for j in cal_easy])
        knn3_dist.append(np.mean(dists[:3]))

    # Evaluate all methods
    methods = {
        'Global centroid': global_cos,
        'Per-scene (2 centroids)': scene_cos,
        'KMeans k=2': kmeans_cos[2],
        'KMeans k=3': kmeans_cos[3],
        'KMeans k=5': kmeans_cos[5],
        'Max similarity': max_sim,
        '5th pctl distance': quantile_dist,
        'kNN k=3 mean': knn3_dist,
    }

    # Labels: 0 = easy (test), 1 = OOD
    eval_idxs = test_easy + ood_idxs
    labels = [0] * len(test_easy) + [1] * len(ood_idxs)

    print(f"\n{'Method':<25} | {'AUROC':>8} | {'Easy mean':>10} | {'OOD mean':>10} | {'Gap':>8}", flush=True)
    print("-" * 75, flush=True)

    method_aurocs = {}
    for name, scores in methods.items():
        eval_scores = [scores[i] for i in eval_idxs]
        auroc = roc_auc_score(labels, eval_scores)
        easy_mean = np.mean([scores[i] for i in test_easy])
        ood_mean = np.mean([scores[i] for i in ood_idxs])
        gap = ood_mean - easy_mean
        method_aurocs[name] = auroc
        print(f"  {name:<23} | {auroc:>8.3f} | {easy_mean:>10.4f} | {ood_mean:>10.4f} | {gap:>+8.4f}", flush=True)

    # Per-OOD-type breakdown for top methods
    print(f"\nPer-OOD-Type AUROC", flush=True)
    print("-" * 90, flush=True)
    ood_types = [s for s in SCENARIOS if s.startswith('ood_')]
    header_parts = [f"{'Method':<25}"]
    for t in ood_types:
        header_parts.append(f"{t[4:]:>10}")
    print("  " + " | ".join(header_parts), flush=True)
    print("  " + "-" * 85, flush=True)

    for name, scores in methods.items():
        parts = [f"{name:<23}"]
        for ood_type in ood_types:
            type_idxs = [i for i in ood_idxs if all_samples[i]['scenario'] == ood_type]
            type_labels = [0] * len(test_easy) + [1] * len(type_idxs)
            type_scores = [scores[i] for i in test_easy] + [scores[i] for i in type_idxs]
            if len(set(type_labels)) > 1:
                auroc = roc_auc_score(type_labels, type_scores)
                parts.append(f"{auroc:>10.3f}")
            else:
                parts.append(f"{'N/A':>10}")
        print("  " + " | ".join(parts), flush=True)

    # Bootstrap confidence intervals for top 3 methods
    print(f"\nBootstrap AUROC (20 iterations)", flush=True)
    print("-" * 60, flush=True)
    top_methods = sorted(method_aurocs.items(), key=lambda x: -x[1])[:4]

    for name, _ in top_methods:
        scores = methods[name]
        bootstrap_aurocs = []
        for b in range(20):
            rng = np.random.default_rng(b)
            b_easy = rng.choice(test_easy, len(test_easy), replace=True).tolist()
            b_ood = rng.choice(ood_idxs, len(ood_idxs), replace=True).tolist()
            b_idxs = b_easy + b_ood
            b_labels = [0] * len(b_easy) + [1] * len(b_ood)
            b_scores = [scores[i] for i in b_idxs]
            bootstrap_aurocs.append(roc_auc_score(b_labels, b_scores))
        mean_auroc = np.mean(bootstrap_aurocs)
        std_auroc = np.std(bootstrap_aurocs)
        print(f"  {name:<23}: {mean_auroc:.3f} ± {std_auroc:.3f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'multi_centroid',
        'experiment_number': 36,
        'timestamp': timestamp,
        'n_calibration': len(cal_easy),
        'n_test_easy': len(test_easy),
        'n_ood': len(ood_idxs),
        'samples': [{k: v for k, v in s.items()} for s in all_samples],
        'method_aurocs': method_aurocs,
    }

    output_path = os.path.join(RESULTS_DIR, f"multi_centroid_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
