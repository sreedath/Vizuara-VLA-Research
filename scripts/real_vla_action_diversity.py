#!/usr/bin/env python3
"""Experiment 418: Action Token Diversity Under Corruption

Studies how corruption affects the DIVERSITY of action token predictions
across scenes. Does corruption collapse actions to a single default
or scatter them? What is the relationship between embedding distance
and action token diversity?

Tests:
1. Action token distribution entropy across scenes for each corruption
2. Per-action-dimension diversity (7 dims: x, y, z, roll, pitch, yaw, gripper)
3. Unique action token counts vs corruption severity
4. Scene-to-scene action token agreement under corruption
5. Action token distance from clean baseline vs embedding distance
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

def extract_hidden_and_action(model, processor, image, prompt, layer=3):
    """Extract hidden state and full 7-dim action token sequence."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    hidden = fwd.hidden_states[layer][0, -1, :].float().cpu().numpy()

    # Get action token (first predicted token)
    logits = fwd.logits[0, -1, :].float().cpu()
    action_logits = logits[31744:32000]
    first_token = int(action_logits.argmax())

    # Generate full 7-token action sequence
    action_tokens = []
    generated = model.generate(
        **inputs,
        max_new_tokens=7,
        do_sample=False,
    )
    gen_tokens = generated[0, inputs['input_ids'].shape[1]:].cpu().tolist()
    for t in gen_tokens[:7]:
        if 31744 <= t <= 31999:
            action_tokens.append(t - 31744)  # 0-255 range
        else:
            action_tokens.append(-1)  # non-action token

    # Pad to 7 if needed
    while len(action_tokens) < 7:
        action_tokens.append(-1)

    return hidden, action_tokens[:7]

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
    corruptions = ['fog', 'night', 'noise', 'blur']
    severities = [0.2, 0.4, 0.6, 0.8, 1.0]

    seeds = [42, 123, 456, 789, 999]
    scenes = [Image.fromarray(np.random.RandomState(s).randint(0, 255, (224, 224, 3), dtype=np.uint8)) for s in seeds]

    # Clean baseline
    print("Extracting clean actions...")
    clean_data = []
    for s in scenes:
        h, tokens = extract_hidden_and_action(model, processor, s, prompt)
        clean_data.append({"hidden": h, "tokens": tokens})
        print(f"  Scene: tokens={tokens}")

    centroid = np.mean([d["hidden"] for d in clean_data], axis=0)
    clean_dists = [cosine_dist(d["hidden"], centroid) for d in clean_data]
    clean_tokens_all = [d["tokens"] for d in clean_data]

    results = {}

    # === Test 1: Per-condition action diversity ===
    print("\n=== Action Diversity per Condition ===")
    condition_actions = {"clean": clean_tokens_all}
    condition_dists = {"clean": clean_dists}

    for c in corruptions:
        for sev in severities:
            cname = f"{c}_{sev}"
            print(f"  {cname}:")
            actions = []
            dists = []
            for i, s in enumerate(scenes):
                corrupted = apply_corruption(s, c, sev)
                h, tokens = extract_hidden_and_action(model, processor, corrupted, prompt)
                actions.append(tokens)
                dists.append(cosine_dist(h, centroid))
                print(f"    Scene {i}: tokens={tokens}")
            condition_actions[cname] = actions
            condition_dists[cname] = dists

    # Analyze diversity
    diversity_results = {}
    for cname, actions in condition_actions.items():
        # Per-dimension diversity
        per_dim_unique = []
        per_dim_range = []
        for dim in range(7):
            dim_vals = [a[dim] for a in actions if a[dim] >= 0]
            if dim_vals:
                per_dim_unique.append(len(set(dim_vals)))
                per_dim_range.append(max(dim_vals) - min(dim_vals))
            else:
                per_dim_unique.append(0)
                per_dim_range.append(0)

        # Overall diversity: unique complete action vectors
        unique_actions = len(set(tuple(a) for a in actions))

        # Agreement: fraction of scenes producing same first token
        first_tokens = [a[0] for a in actions if a[0] >= 0]
        if first_tokens:
            most_common = max(set(first_tokens), key=first_tokens.count)
            agreement = first_tokens.count(most_common) / len(first_tokens)
        else:
            agreement = 0

        # Action token distance from clean
        if cname != "clean":
            token_dists = []
            for i in range(len(actions)):
                if i < len(clean_tokens_all):
                    diff = sum(abs(actions[i][d] - clean_tokens_all[i][d])
                               for d in range(7)
                               if actions[i][d] >= 0 and clean_tokens_all[i][d] >= 0)
                    token_dists.append(diff)
            mean_token_dist = float(np.mean(token_dists)) if token_dists else 0
        else:
            mean_token_dist = 0

        diversity_results[cname] = {
            "unique_actions": unique_actions,
            "per_dim_unique": per_dim_unique,
            "per_dim_range": per_dim_range,
            "first_token_agreement": float(agreement),
            "mean_token_dist_from_clean": mean_token_dist,
            "mean_embedding_dist": float(np.mean(condition_dists.get(cname, [0]))),
            "all_tokens": [a for a in actions],
        }
    results["diversity"] = diversity_results

    # === Test 2: Collapse detection ===
    print("\n=== Collapse Detection ===")
    collapse = {}
    for cname, actions in condition_actions.items():
        # Check if all scenes produce identical actions
        all_same = all(actions[i] == actions[0] for i in range(1, len(actions)))
        collapse[cname] = {
            "all_identical": all_same,
            "unique_count": len(set(tuple(a) for a in actions)),
        }
        if all_same:
            print(f"  {cname}: COLLAPSED to {actions[0]}")
    results["collapse"] = collapse

    # === Test 3: Embedding distance vs action token distance correlation ===
    print("\n=== Distance-Action Correlation ===")
    all_emb_dists = []
    all_tok_dists = []
    for cname in condition_actions:
        if cname == "clean":
            continue
        for i in range(len(condition_actions[cname])):
            emb_d = condition_dists[cname][i]
            tok_d = sum(abs(condition_actions[cname][i][d] - clean_tokens_all[i][d])
                        for d in range(7)
                        if condition_actions[cname][i][d] >= 0 and clean_tokens_all[i][d] >= 0)
            all_emb_dists.append(emb_d)
            all_tok_dists.append(tok_d)

    if np.std(all_emb_dists) > 1e-10 and np.std(all_tok_dists) > 1e-10:
        corr = float(np.corrcoef(all_emb_dists, all_tok_dists)[0, 1])
    else:
        corr = 0.0
    results["dist_action_correlation"] = corr
    print(f"  Embedding dist vs action token dist: r={corr:.4f}")

    # Save
    out_path = "/workspace/Vizuara-VLA-Research/experiments/action_diversity_" + \
               time.strftime("%Y%m%d_%H%M%S") + ".json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
