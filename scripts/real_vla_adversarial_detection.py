#!/usr/bin/env python3
"""Experiment 435: Adversarial-Style Perturbation Detection

Tests detection against subtle, targeted perturbations that mimic
adversarial attacks. These are designed to change model behavior
while being visually imperceptible.

Tests:
1. Uniform random noise at varying epsilon levels
2. Structured perturbations (patches, stripes, gradients)
3. Channel-specific attacks (R, G, B individually)
4. Frequency-domain perturbations (high-freq, low-freq)
5. Detection boundary: minimum perturbation detected
"""

import json, time, os, sys
import numpy as np
import torch
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor

def apply_adversarial(image, atype, epsilon=0.1):
    """Apply adversarial-style perturbations."""
    arr = np.array(image).astype(np.float32) / 255.0
    rng = np.random.RandomState(42)

    if atype == 'uniform_noise':
        arr = arr + rng.uniform(-epsilon, epsilon, arr.shape)
    elif atype == 'gaussian_noise':
        arr = arr + rng.randn(*arr.shape) * epsilon
    elif atype == 'patch':
        # Random patch in center
        h, w = arr.shape[:2]
        ps = max(1, int(h * epsilon * 2))  # patch size proportional to epsilon
        y0, x0 = h//2 - ps//2, w//2 - ps//2
        arr[y0:y0+ps, x0:x0+ps] = rng.rand(ps, ps, 3)
    elif atype == 'horizontal_stripes':
        for y in range(0, arr.shape[0], max(1, int(2 / epsilon))):
            arr[y] = arr[y] + rng.randn(arr.shape[1], 3) * epsilon
    elif atype == 'gradient':
        grad = np.linspace(-epsilon, epsilon, arr.shape[1])[None, :, None]
        arr = arr + grad
    elif atype == 'red_channel':
        arr[:, :, 0] = arr[:, :, 0] + rng.randn(*arr.shape[:2]) * epsilon
    elif atype == 'green_channel':
        arr[:, :, 1] = arr[:, :, 1] + rng.randn(*arr.shape[:2]) * epsilon
    elif atype == 'blue_channel':
        arr[:, :, 2] = arr[:, :, 2] + rng.randn(*arr.shape[:2]) * epsilon
    elif atype == 'high_freq':
        # Checkerboard pattern
        x, y = np.meshgrid(np.arange(arr.shape[1]), np.arange(arr.shape[0]))
        checker = ((x + y) % 2).astype(np.float32)[:, :, None] * 2 - 1
        arr = arr + checker * epsilon
    elif atype == 'low_freq':
        # Smooth sinusoidal perturbation
        x = np.linspace(0, 2 * np.pi, arr.shape[1])
        y = np.linspace(0, 2 * np.pi, arr.shape[0])
        xx, yy = np.meshgrid(x, y)
        wave = np.sin(xx) * np.sin(yy)
        arr = arr + wave[:, :, None] * epsilon
    elif atype == 'salt_pepper':
        mask = rng.random(arr.shape[:2]) < epsilon
        arr[mask] = rng.choice([0.0, 1.0], size=(mask.sum(), 3))

    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))

def extract_hidden(model, processor, image, prompt, layer=3):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return fwd.hidden_states[layer][0, -1, :].float().cpu().numpy()

def cosine_dist(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    return 1.0 - np.dot(a, b) / (na * nb)

def compute_auroc(id_scores, ood_scores):
    id_s = np.asarray(id_scores, dtype=np.float64)
    ood_s = np.asarray(ood_scores, dtype=np.float64)
    n_id, n_ood = len(id_s), len(ood_s)
    if n_id == 0 or n_ood == 0:
        return 0.5
    count = sum(float(np.sum(o > id_s) + 0.5 * np.sum(o == id_s)) for o in ood_s)
    return count / (n_id * n_ood)

def main():
    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    seeds = [42, 123, 456, 789, 999, 1234, 5678, 9999]
    scenes = [Image.fromarray(np.random.RandomState(s).randint(0, 255, (224, 224, 3), dtype=np.uint8)) for s in seeds]

    # Extract clean embeddings
    print("Extracting clean embeddings...")
    clean_embs = [extract_hidden(model, processor, s, prompt) for s in scenes]
    centroid = np.mean(clean_embs, axis=0)
    clean_dists = [cosine_dist(e, centroid) for e in clean_embs]

    results = {"n_scenes": len(scenes)}

    # === Test 1: Epsilon sweep for different perturbation types ===
    print("\n=== Epsilon Sweep ===")
    attack_types = ['uniform_noise', 'gaussian_noise', 'patch', 'horizontal_stripes',
                    'gradient', 'high_freq', 'low_freq', 'salt_pepper']
    epsilons = [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5]

    epsilon_results = {}
    for atype in attack_types:
        per_eps = {}
        for eps in epsilons:
            ood_dists = []
            for s in scenes:
                emb = extract_hidden(model, processor, apply_adversarial(s, atype, eps), prompt)
                ood_dists.append(float(cosine_dist(emb, centroid)))
            auroc = float(compute_auroc(clean_dists, ood_dists))
            per_eps[str(eps)] = {
                "auroc": auroc,
                "mean_dist": float(np.mean(ood_dists)),
            }
        epsilon_results[atype] = per_eps
        # Print detection boundary
        first_detect = None
        for eps in epsilons:
            if per_eps[str(eps)]["auroc"] >= 0.9:
                first_detect = eps
                break
        print(f"  {atype}: detectable at ε≥{first_detect}, ε=0.1 AUROC={per_eps['0.1']['auroc']:.4f}")
    results["epsilon_sweep"] = epsilon_results

    # === Test 2: Channel-specific attacks ===
    print("\n=== Channel-Specific Attacks ===")
    channel_results = {}
    for channel in ['red_channel', 'green_channel', 'blue_channel']:
        per_eps = {}
        for eps in [0.01, 0.05, 0.1, 0.3]:
            ood_dists = []
            for s in scenes:
                emb = extract_hidden(model, processor, apply_adversarial(s, channel, eps), prompt)
                ood_dists.append(float(cosine_dist(emb, centroid)))
            auroc = float(compute_auroc(clean_dists, ood_dists))
            per_eps[str(eps)] = {"auroc": auroc, "mean_dist": float(np.mean(ood_dists))}
        channel_results[channel] = per_eps
        print(f"  {channel}: ε=0.1 AUROC={per_eps['0.1']['auroc']:.4f}, ε=0.3 AUROC={per_eps['0.3']['auroc']:.4f}")
    results["channel_attacks"] = channel_results

    # === Test 3: Detection boundary (fine-grained) ===
    print("\n=== Detection Boundary (Fine-Grained) ===")
    boundary_results = {}
    fine_epsilons = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.008, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1]
    for atype in ['gaussian_noise', 'uniform_noise']:
        per_eps = {}
        for eps in fine_epsilons:
            ood_dists = []
            for s in scenes:
                emb = extract_hidden(model, processor, apply_adversarial(s, atype, eps), prompt)
                ood_dists.append(float(cosine_dist(emb, centroid)))
            auroc = float(compute_auroc(clean_dists, ood_dists))
            per_eps[str(eps)] = {"auroc": auroc}
        boundary_results[atype] = per_eps
        # Find exact boundary
        for eps in fine_epsilons:
            if per_eps[str(eps)]["auroc"] >= 0.9:
                print(f"  {atype}: detection boundary at ε={eps}")
                break
    results["detection_boundary"] = boundary_results

    # === Test 4: Perturbation L2 norm vs detection ===
    print("\n=== Perturbation L2 Norm vs Detection ===")
    norm_results = {}
    for atype in ['gaussian_noise', 'patch', 'high_freq']:
        per_eps = {}
        for eps in [0.01, 0.05, 0.1, 0.3]:
            # Compute actual L2 perturbation norm
            l2_norms = []
            ood_dists = []
            for s in scenes:
                arr_clean = np.array(s).astype(np.float32) / 255.0
                arr_perturbed = np.array(apply_adversarial(s, atype, eps)).astype(np.float32) / 255.0
                l2 = float(np.linalg.norm(arr_perturbed - arr_clean))
                l2_norms.append(l2)
                emb = extract_hidden(model, processor, apply_adversarial(s, atype, eps), prompt)
                ood_dists.append(float(cosine_dist(emb, centroid)))
            auroc = float(compute_auroc(clean_dists, ood_dists))
            per_eps[str(eps)] = {
                "auroc": auroc,
                "mean_l2_norm": float(np.mean(l2_norms)),
                "mean_cosine_dist": float(np.mean(ood_dists)),
            }
        norm_results[atype] = per_eps
        print(f"  {atype}: ε=0.1 → L2={per_eps['0.1']['mean_l2_norm']:.4f}, AUROC={per_eps['0.1']['auroc']:.4f}")
    results["l2_vs_detection"] = norm_results

    out_path = "/workspace/Vizuara-VLA-Research/experiments/adversarial_detection_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
