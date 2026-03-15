#!/usr/bin/env python3
"""Experiment 385: Prompt Engineering for Detection

Does the detection prompt affect detector performance?
1. Multiple prompt styles and their detection distances
2. Prompt length effect on detection
3. Task-specific vs generic prompts
4. Adversarial prompt testing (confusing instructions)
5. Empty/minimal prompt behavior
"""

import json, time, os, sys
import numpy as np
import torch
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor

def extract_hidden(model, processor, image, prompt, layer=3):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return fwd.hidden_states[layer][0, -1, :].float().cpu().numpy()

def apply_corruption(image, ctype, severity=1.0):
    arr = np.array(image).astype(np.float32) / 255.0
    if ctype == 'fog':
        arr = arr * (1 - 0.6 * severity) + 0.6 * severity
    elif ctype == 'night':
        arr = arr * max(0.01, 1.0 - 0.95 * severity)
    elif ctype == 'noise':
        arr = arr + np.random.RandomState(42).randn(*arr.shape) * 0.3 * severity
        arr = np.clip(arr, 0, 1)
    elif ctype == 'blur':
        return image.filter(ImageFilter.GaussianBlur(radius=10 * severity))
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))

def cosine_dist(a, b):
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return 1.0 - dot / (na * nb)

def compute_auroc(id_scores, ood_scores):
    id_s, ood_s = np.asarray(id_scores), np.asarray(ood_scores)
    n_id, n_ood = len(id_s), len(ood_s)
    if n_id == 0 or n_ood == 0: return 0.5
    count = sum(float(np.sum(o > id_s) + 0.5 * np.sum(o == id_s)) for o in ood_s)
    return count / (n_id * n_ood)

def main():
    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    ctypes = ['fog', 'night', 'noise', 'blur']
    results = {}

    # Define prompts
    prompts = {
        'standard': "In: What action should the robot take to pick up the object?\nOut:",
        'simple': "In: Pick up the object.\nOut:",
        'detailed': "In: What action should the robot take to carefully pick up the red object on the table?\nOut:",
        'navigation': "In: What action should the robot take to move forward?\nOut:",
        'generic': "In: Describe this image.\nOut:",
        'minimal': "In: Act.\nOut:",
        'empty_task': "In: \nOut:",
        'reversed': "Out: What action should the robot take to pick up the object?\nIn:",
        'long': "In: Given the current visual observation from the robot's camera, determine the optimal action vector that the robot should execute in order to successfully grasp and pick up the target object visible in the scene.\nOut:",
        'nonsense': "In: Banana purple seventeen clouds.\nOut:",
    }

    print("Generating images...")
    seeds = list(range(0, 1000, 100))[:10]
    images = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        px = rng.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        images[seed] = Image.fromarray(px)

    # ========== Per-Prompt Analysis ==========
    for pname, prompt in prompts.items():
        print(f"\n=== Prompt: {pname} ===")

        # Get clean embeddings
        clean_embs = {}
        for seed in seeds:
            clean_embs[seed] = extract_hidden(model, processor, images[seed], prompt)

        centroid = np.mean(list(clean_embs.values()), axis=0)
        clean_dists = [cosine_dist(centroid, clean_embs[s]) for s in seeds]
        threshold = max(clean_dists)

        # Get corrupt embeddings
        prompt_results = {
            'threshold': float(threshold),
            'mean_clean_dist': float(np.mean(clean_dists)),
            'prompt_length': len(prompt),
        }

        for ct in ctypes:
            ood_dists = []
            for seed in seeds[:5]:
                corrupt_img = apply_corruption(images[seed], ct, 0.5)
                emb = extract_hidden(model, processor, corrupt_img, prompt)
                d = cosine_dist(emb, centroid)
                ood_dists.append(d)

            auroc = compute_auroc(clean_dists, ood_dists)
            det_rate = sum(1 for d in ood_dists if d > threshold) / len(ood_dists)
            separation = float(np.mean(ood_dists)) / max(threshold, 1e-10)

            prompt_results[ct] = {
                'mean_dist': float(np.mean(ood_dists)),
                'auroc': float(auroc),
                'detection_rate': float(det_rate),
                'separation_ratio': float(separation),
            }

        print(f"  threshold={threshold:.6f}, aurocs=" +
              ", ".join(f"{ct}={prompt_results[ct]['auroc']:.2f}" for ct in ctypes))

        results[pname] = prompt_results

    # Cross-prompt centroid comparison
    print("\n=== Cross-Prompt Centroid Similarity ===")
    cross_prompt = {}
    centroids = {}
    for pname, prompt in prompts.items():
        embs = [extract_hidden(model, processor, images[seeds[0]], prompt)]
        centroids[pname] = embs[0]

    for pn1 in list(prompts.keys())[:5]:
        for pn2 in list(prompts.keys())[:5]:
            if pn1 < pn2:
                sim = float(1.0 - cosine_dist(centroids[pn1], centroids[pn2]))
                cross_prompt[f"{pn1}_vs_{pn2}"] = sim

    results['cross_prompt_similarity'] = cross_prompt
    print(f"  {len(cross_prompt)} pairs computed")

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/prompt_engineering_{ts}.json"
    def convert(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)): return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return obj
    def recursive_convert(d):
        if isinstance(d, dict): return {k: recursive_convert(v) for k, v in d.items()}
        if isinstance(d, list): return [recursive_convert(x) for x in d]
        return convert(d)
    results = recursive_convert(results)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
