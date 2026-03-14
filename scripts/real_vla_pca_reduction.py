"""
PCA Dimensionality Reduction for OOD Detection on Real OpenVLA-7B.

Tests whether PCA-compressed hidden states maintain cosine distance
OOD detection performance. If low-dimensional projections work,
this dramatically reduces storage and compute.

Dimensions tested: 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096 (full)

Also tests:
- Random projection as a baseline
- Which PCA components carry the OOD signal

Experiment 38 in the CalibDrive series.
"""
import os
import json
import time
import datetime
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
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
    np.random.seed(idx * 3800 + hash(scenario) % 38000)
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
    print("PCA DIMENSIONALITY REDUCTION FOR OOD DETECTION", flush=True)
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

    # Collect hidden states
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
                    **inputs, max_new_tokens=7, do_sample=False,
                    output_hidden_states=True, return_dict_in_generate=True,
                )

            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                last_step = outputs.hidden_states[-1]
                if isinstance(last_step, tuple):
                    hidden = last_step[-1][0, -1, :].float().cpu().numpy()
                else:
                    hidden = last_step[0, -1, :].float().cpu().numpy()
            else:
                hidden = np.zeros(4096)

            all_samples.append({
                'scenario': scenario,
                'difficulty': config['difficulty'],
                'idx': i,
            })
            all_hidden.append(hidden)

            if sample_idx % 20 == 0 or sample_idx == total:
                print(f"  [{sample_idx}/{total}] {scenario}_{i}", flush=True)

    hidden_arr = np.array(all_hidden)  # (122, 4096)

    # Split
    easy_idxs = [i for i, s in enumerate(all_samples) if s['difficulty'] == 'easy']
    ood_idxs = [i for i, s in enumerate(all_samples) if s['difficulty'] == 'ood']
    np.random.seed(42)
    np.random.shuffle(easy_idxs)
    cal_easy = easy_idxs[:len(easy_idxs)//2]
    test_easy = easy_idxs[len(easy_idxs)//2:]

    eval_idxs = test_easy + ood_idxs
    labels = [0] * len(test_easy) + [1] * len(ood_idxs)

    # ===================================================================
    # Analysis
    # ===================================================================
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    dims = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    # Limit dims to <= available calibration samples
    max_dim = min(len(cal_easy), 4096)

    # 1. PCA reduction
    print("\n1. PCA Reduction: AUROC by Dimension", flush=True)
    print("-" * 80, flush=True)

    pca_results = {}
    for d in dims:
        if d > max_dim and d < 4096:
            continue
        if d == 4096:
            # Full dimension - no PCA needed
            centroid = np.mean(hidden_arr[cal_easy], axis=0)
            scores = [cosine_dist(hidden_arr[i], centroid) for i in eval_idxs]
        else:
            pca = PCA(n_components=d, random_state=42)
            pca.fit(hidden_arr[cal_easy])
            reduced = pca.transform(hidden_arr)
            centroid = np.mean(reduced[cal_easy], axis=0)
            scores = [cosine_dist(reduced[i], centroid) for i in eval_idxs]

        auroc = roc_auc_score(labels, scores)
        pca_results[d] = auroc

        # Explained variance
        if d < 4096:
            var_explained = pca.explained_variance_ratio_.sum()
            storage_kb = d * 4 / 1024  # float32
            print(f"  d={d:>5}: AUROC={auroc:.3f}, var_explained={var_explained:.3f}, "
                  f"storage={storage_kb:.1f}KB", flush=True)
        else:
            print(f"  d={d:>5}: AUROC={auroc:.3f} (full dimension), "
                  f"storage=16.0KB", flush=True)

    # 2. Random projection comparison
    print("\n2. Random Projection: AUROC by Dimension", flush=True)
    print("-" * 80, flush=True)

    rp_results = {}
    for d in [8, 32, 128, 512, 2048]:
        rp = GaussianRandomProjection(n_components=d, random_state=42)
        reduced = rp.fit_transform(hidden_arr)
        centroid = np.mean(reduced[cal_easy], axis=0)
        scores = [cosine_dist(reduced[i], centroid) for i in eval_idxs]
        auroc = roc_auc_score(labels, scores)
        rp_results[d] = auroc
        print(f"  d={d:>5}: AUROC={auroc:.3f}", flush=True)

    # 3. Which PCA components carry OOD signal?
    print("\n3. Component Analysis: Which PCA Components Detect OOD?", flush=True)
    print("-" * 80, flush=True)

    # Fit PCA with all components up to max_dim
    pca_full = PCA(n_components=max_dim, random_state=42)
    pca_full.fit(hidden_arr[cal_easy])
    reduced_full = pca_full.transform(hidden_arr)

    # Test individual components
    print("  Individual component AUROC (top 10):", flush=True)
    component_aurocs = []
    for c in range(min(50, max_dim)):
        comp_scores = [abs(reduced_full[i, c] - np.mean(reduced_full[cal_easy, c]))
                      for i in eval_idxs]
        try:
            auroc = roc_auc_score(labels, comp_scores)
            component_aurocs.append((c, auroc))
        except ValueError:
            pass

    component_aurocs.sort(key=lambda x: -x[1])
    for rank, (c, auroc) in enumerate(component_aurocs[:10], 1):
        var_ratio = pca_full.explained_variance_ratio_[c]
        print(f"    #{rank}: PC{c} AUROC={auroc:.3f} (var={var_ratio:.4f})", flush=True)

    # 4. First k vs last k components
    print("\n4. First k vs Last k Components", flush=True)
    print("-" * 80, flush=True)

    for k in [4, 8, 16]:
        if k > max_dim:
            continue
        # First k
        centroid_first = np.mean(reduced_full[cal_easy, :k], axis=0)
        scores_first = [cosine_dist(reduced_full[i, :k], centroid_first) for i in eval_idxs]
        auroc_first = roc_auc_score(labels, scores_first)

        # Last k
        centroid_last = np.mean(reduced_full[cal_easy, -k:], axis=0)
        scores_last = [cosine_dist(reduced_full[i, -k:], centroid_last) for i in eval_idxs]
        auroc_last = roc_auc_score(labels, scores_last)

        print(f"  k={k:>3}: First {k} AUROC={auroc_first:.3f}, "
              f"Last {k} AUROC={auroc_last:.3f}", flush=True)

    # 5. Per-OOD-type at key dimensions
    print("\n5. Per-OOD-Type AUROC at Key Dimensions", flush=True)
    print("-" * 80, flush=True)
    ood_types = [s for s in SCENARIOS if s.startswith('ood_')]
    key_dims = [d for d in [8, 16, 4096] if d <= max_dim or d == 4096]

    header_parts = [f"{'Dim':>6}"]
    for t in ood_types:
        header_parts.append(f"{t[4:]:>10}")
    header_parts.append(f"{'Overall':>10}")
    print("  " + " | ".join(header_parts), flush=True)

    for d in key_dims:
        if d == 4096:
            centroid = np.mean(hidden_arr[cal_easy], axis=0)
            all_cos = [cosine_dist(hidden_arr[i], centroid) for i in range(len(all_samples))]
        else:
            pca = PCA(n_components=d, random_state=42)
            pca.fit(hidden_arr[cal_easy])
            reduced = pca.transform(hidden_arr)
            centroid = np.mean(reduced[cal_easy], axis=0)
            all_cos = [cosine_dist(reduced[i], centroid) for i in range(len(all_samples))]

        parts = [f"{d:>6}"]
        for ood_type in ood_types:
            type_idxs = [i for i in ood_idxs if all_samples[i]['scenario'] == ood_type]
            type_labels = [0] * len(test_easy) + [1] * len(type_idxs)
            type_scores = [all_cos[i] for i in test_easy] + [all_cos[i] for i in type_idxs]
            auroc = roc_auc_score(type_labels, type_scores)
            parts.append(f"{auroc:>10.3f}")
        overall_scores = [all_cos[i] for i in eval_idxs]
        overall_auroc = roc_auc_score(labels, overall_scores)
        parts.append(f"{overall_auroc:>10.3f}")
        print("  " + " | ".join(parts), flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'pca_reduction',
        'experiment_number': 38,
        'timestamp': timestamp,
        'pca_results': {str(k): v for k, v in pca_results.items()},
        'rp_results': {str(k): v for k, v in rp_results.items()},
        'top_components': component_aurocs[:20],
        'samples': [{k: v for k, v in s.items()} for s in all_samples],
    }

    output_path = os.path.join(RESULTS_DIR, f"pca_reduction_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
