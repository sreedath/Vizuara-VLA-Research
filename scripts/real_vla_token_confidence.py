#!/usr/bin/env python3
"""Experiment 365: Token Prediction Confidence Under Corruption

How does model confidence change when inputs are corrupted?
1. Softmax entropy of next-token prediction
2. Top-1 probability under each corruption
3. Rank of clean-predicted token under corruption
4. Confidence-based OOD detection (AUROC)
5. Per-action-dimension confidence analysis
"""

import json, time, os, sys
import numpy as np
import torch
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor

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

def get_logits_and_tokens(model, processor, image, prompt, n_actions=7):
    """Get logits and generated tokens for action prediction."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    input_len = inputs['input_ids'].shape[1]

    # Generate with logits
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=n_actions + 5,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

    gen_tokens = output.sequences[0, input_len:].cpu().tolist()
    # scores is a tuple of (n_generated_tokens,) each of shape (1, vocab_size)
    scores = [s[0].float().cpu() for s in output.scores]

    return gen_tokens, scores

def compute_auroc(id_scores, ood_scores):
    id_s = np.asarray(id_scores)
    ood_s = np.asarray(ood_scores)
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
    results = {}
    ctypes = ['fog', 'night', 'noise', 'blur']

    # Generate test images
    print("Generating images...")
    seeds = list(range(0, 1500, 100))[:15]
    images = {}
    clean_tokens = {}
    clean_scores = {}

    for seed in seeds:
        rng = np.random.RandomState(seed)
        px = rng.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        images[seed] = Image.fromarray(px)
        tokens, scores = get_logits_and_tokens(model, processor, images[seed], prompt)
        clean_tokens[seed] = tokens
        clean_scores[seed] = scores

    print(f"  Generated {len(seeds)} scenes")

    # ========== 1. Softmax Entropy Under Corruption ==========
    print("\n=== Softmax Entropy ===")

    entropy_results = {}
    for ct in ctypes:
        clean_entropies = []
        corrupt_entropies = []

        for seed in seeds:
            # Clean entropy
            for score in clean_scores[seed][:7]:
                probs = torch.softmax(score, dim=-1).numpy()
                ent = -float(np.sum(probs * np.log2(probs + 1e-12)))
                clean_entropies.append(ent)

            # Corrupt entropy
            corrupt_img = apply_corruption(images[seed], ct, 0.5)
            _, c_scores = get_logits_and_tokens(model, processor, corrupt_img, prompt)
            for score in c_scores[:7]:
                probs = torch.softmax(score, dim=-1).numpy()
                ent = -float(np.sum(probs * np.log2(probs + 1e-12)))
                corrupt_entropies.append(ent)

        entropy_results[ct] = {
            'clean_mean': float(np.mean(clean_entropies)),
            'clean_std': float(np.std(clean_entropies)),
            'corrupt_mean': float(np.mean(corrupt_entropies)),
            'corrupt_std': float(np.std(corrupt_entropies)),
            'entropy_increase': float(np.mean(corrupt_entropies) - np.mean(clean_entropies)),
            'auroc': float(compute_auroc(clean_entropies, corrupt_entropies)),
        }
        print(f"  {ct}: clean={np.mean(clean_entropies):.2f}, "
              f"corrupt={np.mean(corrupt_entropies):.2f}, "
              f"AUROC={entropy_results[ct]['auroc']:.4f}")

    results['softmax_entropy'] = entropy_results

    # ========== 2. Top-1 Probability ==========
    print("\n=== Top-1 Probability ===")

    top1_results = {}
    for ct in ctypes:
        clean_top1 = []
        corrupt_top1 = []

        for seed in seeds:
            for score in clean_scores[seed][:7]:
                probs = torch.softmax(score, dim=-1)
                clean_top1.append(float(probs.max()))

            corrupt_img = apply_corruption(images[seed], ct, 0.5)
            _, c_scores = get_logits_and_tokens(model, processor, corrupt_img, prompt)
            for score in c_scores[:7]:
                probs = torch.softmax(score, dim=-1)
                corrupt_top1.append(float(probs.max()))

        top1_results[ct] = {
            'clean_mean': float(np.mean(clean_top1)),
            'corrupt_mean': float(np.mean(corrupt_top1)),
            'decrease': float(np.mean(clean_top1) - np.mean(corrupt_top1)),
            'auroc': float(compute_auroc(
                [-x for x in clean_top1], [-x for x in corrupt_top1]
            )),
        }
        print(f"  {ct}: clean_top1={np.mean(clean_top1):.4f}, "
              f"corrupt_top1={np.mean(corrupt_top1):.4f}, "
              f"AUROC={top1_results[ct]['auroc']:.4f}")

    results['top1_probability'] = top1_results

    # ========== 3. Rank of Clean Token Under Corruption ==========
    print("\n=== Clean Token Rank Under Corruption ===")

    rank_results = {}
    for ct in ctypes:
        ranks = []
        for seed in seeds:
            corrupt_img = apply_corruption(images[seed], ct, 0.5)
            _, c_scores = get_logits_and_tokens(model, processor, corrupt_img, prompt)

            min_len = min(len(clean_tokens[seed]), len(c_scores))
            for i in range(min(min_len, 7)):
                clean_tok = clean_tokens[seed][i]
                # Sort corrupt logits to find rank of clean token
                sorted_indices = torch.argsort(c_scores[i], descending=True).numpy()
                rank = int(np.where(sorted_indices == clean_tok)[0][0]) + 1
                ranks.append(rank)

        rank_results[ct] = {
            'mean_rank': float(np.mean(ranks)),
            'median_rank': float(np.median(ranks)),
            'max_rank': int(max(ranks)),
            'fraction_top1': float(np.mean([1 if r == 1 else 0 for r in ranks])),
            'fraction_top5': float(np.mean([1 if r <= 5 else 0 for r in ranks])),
            'fraction_top10': float(np.mean([1 if r <= 10 else 0 for r in ranks])),
        }
        print(f"  {ct}: mean_rank={np.mean(ranks):.1f}, "
              f"top1={rank_results[ct]['fraction_top1']*100:.0f}%, "
              f"top5={rank_results[ct]['fraction_top5']*100:.0f}%")

    results['clean_token_rank'] = rank_results

    # ========== 4. Severity-Dependent Confidence ==========
    print("\n=== Severity vs Confidence ===")

    sev_conf = {}
    for ct in ctypes:
        per_sev = {}
        for sev in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
            top1_probs = []
            entropies = []
            for seed in seeds[:5]:
                corrupt_img = apply_corruption(images[seed], ct, sev)
                _, c_scores = get_logits_and_tokens(model, processor, corrupt_img, prompt)
                for score in c_scores[:7]:
                    probs = torch.softmax(score, dim=-1)
                    top1_probs.append(float(probs.max()))
                    p_np = probs.numpy()
                    entropies.append(-float(np.sum(p_np * np.log2(p_np + 1e-12))))

            per_sev[str(sev)] = {
                'mean_top1': float(np.mean(top1_probs)),
                'mean_entropy': float(np.mean(entropies)),
            }

        sev_conf[ct] = per_sev
        ents = [per_sev[str(s)]['mean_entropy'] for s in [0.05, 0.1, 0.5, 1.0]]
        print(f"  {ct}: entropy@[0.05,0.1,0.5,1.0] = [{', '.join(f'{e:.2f}' for e in ents)}]")

    results['severity_confidence'] = sev_conf

    # ========== 5. Entropy-Based OOD Detection ==========
    print("\n=== Entropy-Based Detection (per scene) ===")

    detection = {}
    for ct in ctypes:
        clean_ents = []
        corrupt_ents = []
        for seed in seeds:
            # Mean entropy across action tokens
            ents = []
            for score in clean_scores[seed][:7]:
                probs = torch.softmax(score, dim=-1).numpy()
                ents.append(-float(np.sum(probs * np.log2(probs + 1e-12))))
            clean_ents.append(np.mean(ents))

            corrupt_img = apply_corruption(images[seed], ct, 0.5)
            _, c_scores = get_logits_and_tokens(model, processor, corrupt_img, prompt)
            c_ents = []
            for score in c_scores[:7]:
                probs = torch.softmax(score, dim=-1).numpy()
                c_ents.append(-float(np.sum(probs * np.log2(probs + 1e-12))))
            corrupt_ents.append(np.mean(c_ents))

        auroc = compute_auroc(clean_ents, corrupt_ents)
        detection[ct] = {
            'auroc': float(auroc),
            'clean_mean': float(np.mean(clean_ents)),
            'corrupt_mean': float(np.mean(corrupt_ents)),
            'gap': float(np.mean(corrupt_ents) - np.mean(clean_ents)),
            'overlap': float(max(0, min(max(clean_ents), max(corrupt_ents)) -
                               max(min(clean_ents), min(corrupt_ents)))),
        }
        print(f"  {ct}: AUROC={auroc:.4f}, gap={detection[ct]['gap']:.2f} bits")

    results['entropy_detection'] = detection

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/token_confidence_{ts}.json"
    def convert(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)): return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, torch.Tensor): return obj.tolist()
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
