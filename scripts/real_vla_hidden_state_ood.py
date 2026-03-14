"""
Hidden State OOD Detection on Real OpenVLA-7B.

Tests whether the model's internal hidden states can detect semantic OOD
that action mass misses:
1. Extract hidden states from the last LLM layer
2. Compute Mahalanobis distance from easy-scenario distribution
3. Use cosine similarity between hidden states
4. Test on diverse OOD types including semantic OOD (indoor, inverted)

Experiment 27 in the CalibDrive series.
"""
import os
import json
import time
import datetime
import numpy as np
import torch
from PIL import Image, ImageDraw

RESULTS_DIR = "/workspace/Vizuara-VLA-Research/experiments"
os.makedirs(RESULTS_DIR, exist_ok=True)

SCENARIOS = {
    'highway': {'n': 20, 'speed': '30', 'difficulty': 'easy'},
    'urban': {'n': 20, 'speed': '15', 'difficulty': 'easy'},
    'ood_noise': {'n': 10, 'speed': '25', 'difficulty': 'ood'},
    'ood_blank': {'n': 10, 'speed': '25', 'difficulty': 'ood'},
    'ood_indoor': {'n': 10, 'speed': '25', 'difficulty': 'ood'},
    'ood_inverted': {'n': 10, 'speed': '30', 'difficulty': 'ood'},
    'ood_checker': {'n': 10, 'speed': '25', 'difficulty': 'ood'},
    'ood_blackout': {'n': 10, 'speed': '25', 'difficulty': 'ood'},
}


def create_scene_image(scenario, idx, size=(256, 256)):
    np.random.seed(idx * 2200 + hash(scenario) % 22000)
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


def compute_auroc(pos_scores, neg_scores):
    n_correct = sum(1 for p in pos_scores for n in neg_scores if p > n)
    n_ties = sum(0.5 for p in pos_scores for n in neg_scores if p == n)
    n_total = len(pos_scores) * len(neg_scores)
    return (n_correct + n_ties) / n_total if n_total > 0 else 0.5


def main():
    print("=" * 70, flush=True)
    print("HIDDEN STATE OOD DETECTION ON REAL OpenVLA-7B", flush=True)
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

    total = sum(s['n'] for s in SCENARIOS.values())
    print(f"Total samples: {total}", flush=True)
    print(flush=True)

    prompt = "In: What action should the robot take to drive forward at {speed} m/s safely?\nOut:"

    all_samples = []
    all_hidden_states = []
    sample_idx = 0

    for scenario, config in SCENARIOS.items():
        for i in range(config['n']):
            sample_idx += 1
            image = create_scene_image(scenario, i)
            p = prompt.format(speed=config['speed'])

            inputs = processor(p, image).to(model.device, dtype=torch.bfloat16)

            t0 = time.time()
            with torch.no_grad():
                # Get hidden states from last layer before generation
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=7,
                    do_sample=False,
                    output_scores=True,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                )

            # Extract hidden state from the last step of generation
            # outputs.hidden_states is a tuple of (step, layer, batch, seq, hidden)
            # Get last generated token's hidden state from last layer
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                # Each step has hidden states from all layers
                # Take last generated step, last layer, last token
                last_step_hidden = outputs.hidden_states[-1]  # Last step
                if isinstance(last_step_hidden, tuple):
                    last_layer = last_step_hidden[-1]  # Last layer
                    hidden = last_layer[0, -1, :].float().cpu().numpy()  # Last token
                else:
                    hidden = last_step_hidden[0, -1, :].float().cpu().numpy()
            else:
                hidden = np.zeros(4096)  # Fallback

            # Also get action mass and entropy
            vocab_size = outputs.scores[0].shape[-1]
            action_start = vocab_size - 256
            dim_masses = []
            dim_entropies = []
            for score in outputs.scores[:7]:
                full_logits = score[0].float()
                full_probs = torch.softmax(full_logits, dim=0).cpu().numpy()
                action_probs = full_probs[action_start:]
                dim_masses.append(float(action_probs.sum()))
                action_norm = action_probs / (action_probs.sum() + 1e-10)
                dim_entropies.append(float(-(action_norm * np.log(action_norm + 1e-10)).sum()))

            elapsed = time.time() - t0

            # Reduce hidden state dimensionality for storage
            hidden_norm = float(np.linalg.norm(hidden))
            hidden_mean = float(np.mean(hidden))
            hidden_std = float(np.std(hidden))
            hidden_abs_max = float(np.max(np.abs(hidden)))

            sample = {
                'scenario': scenario,
                'difficulty': config['difficulty'],
                'idx': i,
                'action_mass': float(np.mean(dim_masses)),
                'entropy': float(np.mean(dim_entropies)),
                'hidden_norm': hidden_norm,
                'hidden_mean': hidden_mean,
                'hidden_std': hidden_std,
                'hidden_abs_max': hidden_abs_max,
            }
            all_samples.append(sample)
            all_hidden_states.append(hidden)

            if i % 5 == 0 or i == config['n'] - 1:
                print(f"  [{sample_idx}/{total}] {scenario}_{i}: "
                      f"mass={sample['action_mass']:.4f}, ent={sample['entropy']:.3f}, "
                      f"h_norm={hidden_norm:.1f} ({elapsed:.1f}s)", flush=True)

    # ===================================================================
    # Analysis
    # ===================================================================
    print("\n" + "=" * 70, flush=True)
    print("HIDDEN STATE ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    hidden_arr = np.array(all_hidden_states)

    # 1. Per-scenario hidden state statistics
    print("\n1. Per-Scenario Hidden State Statistics", flush=True)
    print("-" * 80, flush=True)
    print(f"{'Scenario':<15} {'H_norm':>8} {'H_mean':>8} {'H_std':>8} {'H_max':>8} "
          f"{'Mass':>7} {'Ent':>7}", flush=True)
    print("-" * 80, flush=True)

    for scenario in SCENARIOS:
        idxs = [i for i, s in enumerate(all_samples) if s['scenario'] == scenario]
        norms = [all_samples[i]['hidden_norm'] for i in idxs]
        means = [all_samples[i]['hidden_mean'] for i in idxs]
        stds = [all_samples[i]['hidden_std'] for i in idxs]
        maxes = [all_samples[i]['hidden_abs_max'] for i in idxs]
        masses = [all_samples[i]['action_mass'] for i in idxs]
        ents = [all_samples[i]['entropy'] for i in idxs]

        print(f"{scenario:<15} {np.mean(norms):>8.1f} {np.mean(means):>8.4f} "
              f"{np.mean(stds):>8.4f} {np.mean(maxes):>8.2f} "
              f"{np.mean(masses):>7.4f} {np.mean(ents):>7.3f}", flush=True)

    # 2. Cosine similarity between scenario centroids
    print("\n2. Cosine Similarity Between Scenario Centroids", flush=True)
    print("-" * 60, flush=True)

    centroids = {}
    for scenario in SCENARIOS:
        idxs = [i for i, s in enumerate(all_samples) if s['scenario'] == scenario]
        centroids[scenario] = np.mean(hidden_arr[idxs], axis=0)

    for s1 in ['highway', 'urban']:
        for s2 in SCENARIOS:
            if s2 == s1:
                continue
            cos_sim = np.dot(centroids[s1], centroids[s2]) / (
                np.linalg.norm(centroids[s1]) * np.linalg.norm(centroids[s2]) + 1e-10)
            print(f"  {s1:<12} ↔ {s2:<15}: cos_sim = {cos_sim:.4f}", flush=True)

    # 3. Mahalanobis-like distance using easy distribution
    print("\n3. Distance-Based OOD Detection", flush=True)
    print("-" * 60, flush=True)

    easy_idxs = [i for i, s in enumerate(all_samples) if s['difficulty'] == 'easy']
    easy_hidden = hidden_arr[easy_idxs]
    easy_mean = np.mean(easy_hidden, axis=0)

    # Use L2 distance to centroid as simple signal
    for s_idx, s in enumerate(all_samples):
        dist = float(np.linalg.norm(hidden_arr[s_idx] - easy_mean))
        s['centroid_dist'] = dist

    easy_dist = [s['centroid_dist'] for s in all_samples if s['difficulty'] == 'easy']
    for ood_type in [s for s in SCENARIOS if s.startswith('ood_')]:
        ood_s = [s for s in all_samples if s['scenario'] == ood_type]
        ood_dist = [s['centroid_dist'] for s in ood_s]
        auroc = compute_auroc(ood_dist, easy_dist)
        print(f"  {ood_type:<15}: L2 dist AUROC = {auroc:.3f} "
              f"(mean dist: easy={np.mean(easy_dist):.1f}, ood={np.mean(ood_dist):.1f})", flush=True)

    # Overall
    all_ood_dist = [s['centroid_dist'] for s in all_samples if s['difficulty'] == 'ood']
    auroc_dist_all = compute_auroc(all_ood_dist, easy_dist)
    print(f"\n  Overall L2 dist AUROC (all OOD): {auroc_dist_all:.3f}", flush=True)

    # 4. Hidden norm as signal
    print("\n4. Hidden State Norm as OOD Signal", flush=True)
    print("-" * 60, flush=True)

    easy_norm = [s['hidden_norm'] for s in all_samples if s['difficulty'] == 'easy']
    for ood_type in [s for s in SCENARIOS if s.startswith('ood_')]:
        ood_s = [s for s in all_samples if s['scenario'] == ood_type]
        ood_norm = [s['hidden_norm'] for s in ood_s]
        # Try both directions
        auroc_high = compute_auroc(ood_norm, easy_norm)
        auroc_low = compute_auroc(easy_norm, ood_norm)
        auroc = max(auroc_high, auroc_low)
        direction = "higher" if auroc_high > auroc_low else "lower"
        print(f"  {ood_type:<15}: norm AUROC = {auroc:.3f} (OOD norm {direction})", flush=True)

    # 5. Comparison: combined hidden state + action mass
    print("\n5. Combined Signal: Centroid Distance + Action Mass", flush=True)
    print("-" * 60, flush=True)

    # Normalize both signals to [0,1] range
    all_dists = [s['centroid_dist'] for s in all_samples]
    all_masses = [s['action_mass'] for s in all_samples]
    d_min, d_max = min(all_dists), max(all_dists)
    m_min, m_max = min(all_masses), max(all_masses)

    for w_dist in [0.0, 0.25, 0.5, 0.75, 1.0]:
        w_mass = 1.0 - w_dist
        combined_easy = []
        combined_ood = []
        for s in all_samples:
            d_norm = (s['centroid_dist'] - d_min) / (d_max - d_min + 1e-10)
            m_norm = 1.0 - (s['action_mass'] - m_min) / (m_max - m_min + 1e-10)
            combined = w_dist * d_norm + w_mass * m_norm
            if s['difficulty'] == 'easy':
                combined_easy.append(combined)
            elif s['difficulty'] == 'ood':
                combined_ood.append(combined)
        auroc = compute_auroc(combined_ood, combined_easy)
        print(f"  w_dist={w_dist:.2f}, w_mass={w_mass:.2f}: AUROC = {auroc:.3f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'timestamp': timestamp,
        'samples': [{k: v for k, v in s.items()} for s in all_samples],
    }

    output_path = os.path.join(RESULTS_DIR, f"hidden_state_ood_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
