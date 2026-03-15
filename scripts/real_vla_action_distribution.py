#!/usr/bin/env python3
"""Experiment 302: Action Token Distribution Analysis
Analyzes how corruption affects the output token distribution:
1. Full logit distribution comparison (clean vs corrupted)
2. Top-K token changes across corruption types
3. Entropy of action token distributions
4. Confidence calibration under corruption
5. Token probability mass shift
"""

import torch
import numpy as np
import json
from datetime import datetime
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor
from scipy.spatial.distance import cosine

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

def main():
    print("Loading OpenVLA-7B...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    np.random.seed(42)
    pixels = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    base_img = Image.fromarray(pixels)

    results = {
        "experiment": "action_distribution",
        "experiment_number": 302,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    corruptions = ['fog', 'night', 'blur', 'noise']
    ACTION_TOKEN_START = 31744
    ACTION_TOKEN_END = 31999

    # Part 1: Get clean logit distribution
    print("=== Part 1: Clean Logits ===")
    inputs = processor(prompt, base_img).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        outputs = model(**inputs)

    clean_logits = outputs.logits[0, -1, :].float().cpu().numpy()
    clean_action_logits = clean_logits[ACTION_TOKEN_START:ACTION_TOKEN_END+1]

    # Softmax
    clean_probs = np.exp(clean_action_logits - clean_action_logits.max())
    clean_probs = clean_probs / clean_probs.sum()

    clean_token = int(np.argmax(clean_action_logits)) + ACTION_TOKEN_START
    clean_entropy = -np.sum(clean_probs * np.log(clean_probs + 1e-30))
    clean_top1_prob = float(clean_probs.max())

    # Top 5
    top5_idx = np.argsort(clean_action_logits)[-5:][::-1]
    clean_top5 = [(int(i + ACTION_TOKEN_START), float(clean_probs[i])) for i in top5_idx]

    results["clean"] = {
        "token": clean_token,
        "entropy": float(clean_entropy),
        "top1_prob": clean_top1_prob,
        "top5": clean_top5,
        "action_bin": clean_token - ACTION_TOKEN_START
    }
    print(f"  Token={clean_token}, bin={clean_token-ACTION_TOKEN_START}, "
          f"entropy={clean_entropy:.4f}, top1_prob={clean_top1_prob:.4f}")

    # Part 2: Corrupted distributions
    print("\n=== Part 2: Corrupted Distributions ===")
    corrupted_results = {}

    for c in corruptions:
        corrupted_results[c] = {}
        for sev in [0.3, 0.5, 1.0]:
            corrupted = apply_corruption(base_img, c, sev)
            inputs = processor(prompt, corrupted).to(model.device, dtype=torch.bfloat16)
            with torch.no_grad():
                outputs = model(**inputs)

            logits = outputs.logits[0, -1, :].float().cpu().numpy()
            action_logits = logits[ACTION_TOKEN_START:ACTION_TOKEN_END+1]

            probs = np.exp(action_logits - action_logits.max())
            probs = probs / probs.sum()

            token = int(np.argmax(action_logits)) + ACTION_TOKEN_START
            entropy = -np.sum(probs * np.log(probs + 1e-30))
            top1_prob = float(probs.max())

            # KL divergence from clean
            kl = np.sum(clean_probs * np.log((clean_probs + 1e-30) / (probs + 1e-30)))

            # Total variation distance
            tv = 0.5 * np.sum(np.abs(clean_probs - probs))

            # Top 5
            top5_idx = np.argsort(action_logits)[-5:][::-1]
            top5 = [(int(i + ACTION_TOKEN_START), float(probs[i])) for i in top5_idx]

            # Token shift
            token_shift = abs(token - clean_token)

            corrupted_results[c][f"sev_{sev}"] = {
                "token": token,
                "action_bin": token - ACTION_TOKEN_START,
                "entropy": float(entropy),
                "top1_prob": top1_prob,
                "kl_from_clean": float(kl),
                "tv_from_clean": float(tv),
                "token_shift": token_shift,
                "top5": top5,
                "same_token_as_clean": token == clean_token
            }
            print(f"  {c} sev={sev}: token={token} (shift={token_shift}), "
                  f"entropy={entropy:.4f}, KL={kl:.4f}, TV={tv:.4f}")

    results["corrupted"] = corrupted_results

    # Part 3: Confidence comparison
    print("\n=== Part 3: Confidence Analysis ===")
    confidence = {}
    for c in corruptions:
        sev1 = corrupted_results[c]["sev_1.0"]
        confidence[c] = {
            "clean_confidence": clean_top1_prob,
            "corrupted_confidence": sev1["top1_prob"],
            "confidence_ratio": sev1["top1_prob"] / (clean_top1_prob + 1e-30),
            "entropy_ratio": sev1["entropy"] / (clean_entropy + 1e-30),
            "wrong_but_confident": sev1["token_shift"] > 0 and sev1["top1_prob"] > 0.5
        }
        print(f"  {c}: clean_conf={clean_top1_prob:.4f}, corrupt_conf={sev1['top1_prob']:.4f}, "
              f"wrong_confident={confidence[c]['wrong_but_confident']}")

    results["confidence"] = confidence

    # Part 4: Multi-step action analysis (7 action tokens)
    print("\n=== Part 4: Multi-Step Actions ===")
    multi_step = {}

    for condition in ['clean'] + [f"{c}_1.0" for c in corruptions]:
        if condition == 'clean':
            img = base_img
        else:
            c, sev = condition.rsplit('_', 1)
            img = apply_corruption(base_img, c, float(sev))

        inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=7,
                do_sample=False
            )

        # Get the generated tokens
        input_len = inputs['input_ids'].shape[1]
        gen_tokens = generated[0, input_len:].cpu().numpy()
        action_bins = [int(t - ACTION_TOKEN_START) if ACTION_TOKEN_START <= t <= ACTION_TOKEN_END else -1
                       for t in gen_tokens]

        multi_step[condition] = {
            "tokens": gen_tokens.tolist(),
            "action_bins": action_bins,
            "n_action_tokens": sum(1 for b in action_bins if b >= 0)
        }
        print(f"  {condition}: bins={action_bins[:7]}")

    results["multi_step"] = multi_step

    # Compare actions
    clean_bins = multi_step['clean']['action_bins']
    for c in corruptions:
        corr_bins = multi_step[f"{c}_1.0"]["action_bins"]
        n_different = sum(1 for a, b in zip(clean_bins, corr_bins) if a != b)
        total_shift = sum(abs(a - b) for a, b in zip(clean_bins, corr_bins) if a >= 0 and b >= 0)
        print(f"  {c}: {n_different}/7 dims changed, total_shift={total_shift}")

    # Save (convert numpy types)
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(v) for v in obj]
        return obj

    ts = results["timestamp"]
    out_path = f"experiments/action_dist_{ts}.json"
    with open(out_path, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
