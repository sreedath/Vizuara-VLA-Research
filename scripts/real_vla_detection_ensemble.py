#!/usr/bin/env python3
"""Experiment 386: Detection Ensemble Methods

Can we improve noise detection by combining multiple detection signals?
1. Multi-layer ensemble (L1, L3, L8, L16, L24)
2. Multi-prompt ensemble (5 different prompts)
3. Multi-metric ensemble (cosine, euclidean, mahalanobis)
4. Voting vs averaging vs max-of-scores
5. Per-corruption optimal ensemble
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

def extract_multi_layer(model, processor, image, prompt, layers=[1, 3, 8, 16, 24]):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return {L: fwd.hidden_states[L][0, -1, :].float().cpu().numpy() for L in layers}

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
    if na < 1e-10 or nb < 1e-10: return 0.0
    return 1.0 - dot / (na * nb)

def euclidean_dist(a, b):
    return float(np.linalg.norm(a - b))

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
    layers = [1, 3, 8, 16, 24]
    prompts = [
        "In: What action should the robot take to pick up the object?\nOut:",
        "In: Pick up the object.\nOut:",
        "In: What action should the robot take to move forward?\nOut:",
        "In: Act.\nOut:",
        "In: Describe this image.\nOut:",
    ]

    print("Generating images...")
    seeds = list(range(0, 1000, 100))[:10]
    images = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        px = rng.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        images[seed] = Image.fromarray(px)

    # ========== 1. Multi-Layer Ensemble ==========
    print("\n=== Multi-Layer Ensemble ===")
    
    prompt = prompts[0]
    clean_multi = {}
    for seed in seeds:
        clean_multi[seed] = extract_multi_layer(model, processor, images[seed], prompt, layers)

    layer_centroids = {L: np.mean([clean_multi[s][L] for s in seeds], axis=0) for L in layers}
    
    for ct in ctypes:
        # Per-layer scores
        layer_id_scores = {L: [] for L in layers}
        layer_ood_scores = {L: [] for L in layers}
        
        for seed in seeds:
            for L in layers:
                layer_id_scores[L].append(cosine_dist(clean_multi[seed][L], layer_centroids[L]))
        
        for seed in seeds[:5]:
            corrupt_embs = extract_multi_layer(model, processor,
                apply_corruption(images[seed], ct, 0.5), prompt, layers)
            for L in layers:
                layer_ood_scores[L].append(cosine_dist(corrupt_embs[L], layer_centroids[L]))

        # Individual layer AUROCs
        individual = {}
        for L in layers:
            individual[f'L{L}'] = compute_auroc(layer_id_scores[L], layer_ood_scores[L])

        # Ensemble methods
        # Normalize scores to [0, 1] using min-max of ID scores
        def normalize_scores(id_s, ood_s):
            all_s = id_s + ood_s
            mn, mx = min(all_s), max(all_s)
            if mx - mn < 1e-15: return id_s, ood_s
            return [(x - mn) / (mx - mn) for x in id_s], [(x - mn) / (mx - mn) for x in ood_s]

        # Average ensemble
        avg_id = [np.mean([layer_id_scores[L][i] / max(max(layer_id_scores[L]), 1e-10) 
                           for L in layers]) for i in range(len(seeds))]
        avg_ood = [np.mean([layer_ood_scores[L][i] / max(max(layer_id_scores[L]), 1e-10) 
                            for L in layers]) for i in range(5)]
        avg_auroc = compute_auroc(avg_id, avg_ood)

        # Max ensemble
        max_id = [max(layer_id_scores[L][i] / max(max(layer_id_scores[L]), 1e-10) 
                      for L in layers) for i in range(len(seeds))]
        max_ood = [max(layer_ood_scores[L][i] / max(max(layer_id_scores[L]), 1e-10) 
                       for L in layers) for i in range(5)]
        max_auroc = compute_auroc(max_id, max_ood)

        # Vote ensemble (majority vote > threshold)
        vote_id = [sum(1 for L in layers if layer_id_scores[L][i] > 
                       np.percentile(layer_id_scores[L], 90)) for i in range(len(seeds))]
        vote_ood = [sum(1 for L in layers if layer_ood_scores[L][i] > 
                        np.percentile(layer_id_scores[L], 90)) for i in range(5)]
        vote_auroc = compute_auroc(vote_id, vote_ood)

        results[f'multi_layer_{ct}'] = {
            'individual': individual,
            'avg_auroc': float(avg_auroc),
            'max_auroc': float(max_auroc),
            'vote_auroc': float(vote_auroc),
        }
        print(f"  {ct}: individual={individual}, avg={avg_auroc:.3f}, "
              f"max={max_auroc:.3f}, vote={vote_auroc:.3f}")

    # ========== 2. Multi-Prompt Ensemble ==========
    print("\n=== Multi-Prompt Ensemble ===")

    prompt_embs = {}
    prompt_centroids = {}
    for pi, p in enumerate(prompts):
        prompt_embs[pi] = {}
        for seed in seeds:
            prompt_embs[pi][seed] = extract_hidden(model, processor, images[seed], p)
        prompt_centroids[pi] = np.mean([prompt_embs[pi][s] for s in seeds], axis=0)

    for ct in ctypes:
        prompt_id = {pi: [] for pi in range(len(prompts))}
        prompt_ood = {pi: [] for pi in range(len(prompts))}

        for seed in seeds:
            for pi in range(len(prompts)):
                prompt_id[pi].append(cosine_dist(prompt_embs[pi][seed], prompt_centroids[pi]))

        for seed in seeds[:5]:
            for pi, p in enumerate(prompts):
                corrupt_emb = extract_hidden(model, processor,
                    apply_corruption(images[seed], ct, 0.5), p)
                prompt_ood[pi].append(cosine_dist(corrupt_emb, prompt_centroids[pi]))

        individual = {f'p{pi}': compute_auroc(prompt_id[pi], prompt_ood[pi]) 
                      for pi in range(len(prompts))}

        # Average
        avg_id = [np.mean([prompt_id[pi][i] / max(max(prompt_id[pi]), 1e-10) 
                           for pi in range(len(prompts))]) for i in range(len(seeds))]
        avg_ood = [np.mean([prompt_ood[pi][i] / max(max(prompt_id[pi]), 1e-10)
                            for pi in range(len(prompts))]) for i in range(5)]
        avg_auroc = compute_auroc(avg_id, avg_ood)

        results[f'multi_prompt_{ct}'] = {
            'individual': individual,
            'avg_auroc': float(avg_auroc),
        }
        print(f"  {ct}: individual={individual}, avg={avg_auroc:.3f}")

    # ========== 3. Multi-Metric Ensemble ==========
    print("\n=== Multi-Metric Ensemble ===")

    for ct in ctypes:
        cos_id = [cosine_dist(clean_multi[s][3], layer_centroids[3]) for s in seeds]
        euc_id = [euclidean_dist(clean_multi[s][3], layer_centroids[3]) for s in seeds]

        cos_ood, euc_ood = [], []
        for seed in seeds[:5]:
            emb = extract_hidden(model, processor,
                apply_corruption(images[seed], ct, 0.5), prompt)
            cos_ood.append(cosine_dist(emb, layer_centroids[3]))
            euc_ood.append(euclidean_dist(emb, layer_centroids[3]))

        cos_auroc = compute_auroc(cos_id, cos_ood)
        euc_auroc = compute_auroc(euc_id, euc_ood)

        # Average normalized
        avg_id = [np.mean([cos_id[i]/max(max(cos_id),1e-10), euc_id[i]/max(max(euc_id),1e-10)])
                  for i in range(len(seeds))]
        avg_ood = [np.mean([cos_ood[i]/max(max(cos_id),1e-10), euc_ood[i]/max(max(euc_id),1e-10)])
                   for i in range(5)]
        avg_auroc = compute_auroc(avg_id, avg_ood)

        results[f'multi_metric_{ct}'] = {
            'cosine_auroc': float(cos_auroc),
            'euclidean_auroc': float(euc_auroc),
            'avg_auroc': float(avg_auroc),
        }
        print(f"  {ct}: cos={cos_auroc:.3f}, euc={euc_auroc:.3f}, avg={avg_auroc:.3f}")

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/detection_ensemble_{ts}.json"
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
