#!/usr/bin/env python3
"""Experiment 307: Decision Boundary & Embedding Interpolation Analysis
Studies the geometry of clean/corrupted decision boundaries:
1. Image-space interpolation (alpha-blend clean→corrupted)
2. Embedding-space interpolation characteristics
3. Decision boundary sharpness (how fast distance grows)
4. Cross-corruption interpolation (blend between corruption types)
5. Action change threshold vs detection threshold
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

def extract_hidden(model, processor, image, prompt, layer=3):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return fwd.hidden_states[layer][0, -1, :].float().cpu().numpy()

def get_action_tokens(model, processor, image, prompt):
    ACTION_TOKEN_START = 31744
    ACTION_TOKEN_END = 31999
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=7, do_sample=False)
    input_len = inputs['input_ids'].shape[1]
    gen_tokens = generated[0, input_len:].cpu().numpy()
    return [int(t - ACTION_TOKEN_START) if ACTION_TOKEN_START <= t <= ACTION_TOKEN_END else -1
            for t in gen_tokens]

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
        "experiment": "decision_boundary",
        "experiment_number": 307,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    corruptions = ['fog', 'night', 'blur', 'noise']
    clean_emb = extract_hidden(model, processor, base_img, prompt)
    clean_actions = get_action_tokens(model, processor, base_img, prompt)
    print(f"Clean actions: {clean_actions}")

    # Part 1: Fine-grained image-space interpolation
    print("=== Part 1: Image-Space Interpolation ===")
    interpolation = {}

    alphas = [0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    clean_arr = np.array(base_img).astype(np.float32) / 255.0

    for c in corruptions:
        print(f"  {c}...")
        if c == 'blur':
            # For blur, use severity parameter directly
            interp_data = []
            for alpha in alphas:
                if alpha == 0:
                    img = base_img
                else:
                    img = apply_corruption(base_img, c, alpha)
                emb = extract_hidden(model, processor, img, prompt)
                d = float(cosine(clean_emb, emb))
                actions = get_action_tokens(model, processor, img, prompt)
                n_changed = sum(1 for a, b in zip(clean_actions, actions) if a != b)
                interp_data.append({
                    "alpha": alpha,
                    "distance": d,
                    "actions": actions,
                    "n_actions_changed": n_changed,
                })
        else:
            corrupted_arr = np.array(apply_corruption(base_img, c, 1.0)).astype(np.float32) / 255.0
            interp_data = []
            for alpha in alphas:
                blended = (1 - alpha) * clean_arr + alpha * corrupted_arr
                blended = np.clip(blended, 0, 1)
                img = Image.fromarray((blended * 255).astype(np.uint8))
                emb = extract_hidden(model, processor, img, prompt)
                d = float(cosine(clean_emb, emb))
                actions = get_action_tokens(model, processor, img, prompt)
                n_changed = sum(1 for a, b in zip(clean_actions, actions) if a != b)
                interp_data.append({
                    "alpha": alpha,
                    "distance": d,
                    "actions": actions,
                    "n_actions_changed": n_changed,
                })

        interpolation[c] = interp_data

        # Find detection and action thresholds
        first_detect = next((d for d in interp_data if d["distance"] > 0), None)
        first_action = next((d for d in interp_data if d["n_actions_changed"] > 0), None)
        detect_alpha = first_detect["alpha"] if first_detect else "never"
        action_alpha = first_action["alpha"] if first_action else "never"
        print(f"    Detection at alpha={detect_alpha}, action change at alpha={action_alpha}")

    results["interpolation"] = interpolation

    # Part 2: Embedding trajectory analysis
    print("\n=== Part 2: Embedding Trajectory ===")
    trajectory = {}

    for c in corruptions:
        print(f"  {c}...")
        embeddings = []
        for alpha in [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
            if c == 'blur':
                img = apply_corruption(base_img, c, alpha) if alpha > 0 else base_img
            else:
                corrupted_arr = np.array(apply_corruption(base_img, c, 1.0)).astype(np.float32) / 255.0
                blended = (1 - alpha) * clean_arr + alpha * corrupted_arr
                img = Image.fromarray((np.clip(blended, 0, 1) * 255).astype(np.uint8))
            emb = extract_hidden(model, processor, img, prompt)
            embeddings.append(emb)

        # Compute pairwise distances and linearity
        distances_from_clean = [float(cosine(embeddings[0], e)) for e in embeddings]
        distances_sequential = [float(cosine(embeddings[i], embeddings[i+1]))
                                for i in range(len(embeddings)-1)]

        # Path length vs straight-line distance (curvature)
        path_length = sum(float(np.linalg.norm(embeddings[i+1] - embeddings[i]))
                          for i in range(len(embeddings)-1))
        straight_line = float(np.linalg.norm(embeddings[-1] - embeddings[0]))
        curvature = path_length / (straight_line + 1e-30)

        # Direction consistency
        directions = []
        for i in range(len(embeddings)-1):
            diff = embeddings[i+1] - embeddings[i]
            diff_norm = np.linalg.norm(diff)
            if diff_norm > 0:
                directions.append(diff / diff_norm)
        dir_sims = []
        if len(directions) > 1:
            for i in range(len(directions)-1):
                dir_sims.append(float(np.dot(directions[i], directions[i+1])))

        trajectory[c] = {
            "distances_from_clean": distances_from_clean,
            "distances_sequential": distances_sequential,
            "path_length": path_length,
            "straight_line": straight_line,
            "curvature_ratio": curvature,
            "direction_consistency": dir_sims,
            "mean_direction_sim": float(np.mean(dir_sims)) if dir_sims else 0,
        }
        print(f"    curvature={curvature:.3f}, mean_dir_sim={trajectory[c]['mean_direction_sim']:.4f}")

    results["trajectory"] = trajectory

    # Part 3: Cross-corruption interpolation
    print("\n=== Part 3: Cross-Corruption Interpolation ===")
    cross_interp = {}

    pairs = [('fog', 'night'), ('fog', 'noise'), ('night', 'blur'), ('blur', 'noise')]
    for c1, c2 in pairs:
        print(f"  {c1}→{c2}...")
        arr1 = np.array(apply_corruption(base_img, c1, 1.0)).astype(np.float32) / 255.0
        arr2 = np.array(apply_corruption(base_img, c2, 1.0)).astype(np.float32) / 255.0

        cross_data = []
        for alpha in [0, 0.25, 0.5, 0.75, 1.0]:
            blended = (1 - alpha) * arr1 + alpha * arr2
            img = Image.fromarray((np.clip(blended, 0, 1) * 255).astype(np.uint8))
            emb = extract_hidden(model, processor, img, prompt)
            d = float(cosine(clean_emb, emb))
            cross_data.append({"alpha": alpha, "distance": d})

        cross_interp[f"{c1}_to_{c2}"] = cross_data
        dists = [x["distance"] for x in cross_data]
        print(f"    distances: {[f'{d:.6f}' for d in dists]}")

    results["cross_interpolation"] = cross_interp

    # Part 4: Decision boundary sharpness
    print("\n=== Part 4: Decision Boundary Sharpness ===")
    sharpness = {}

    for c in corruptions:
        print(f"  {c}...")
        # Very fine-grained around the detection threshold
        fine_alphas = [0, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.05]
        fine_data = []

        for alpha in fine_alphas:
            if c == 'blur':
                img = apply_corruption(base_img, c, alpha) if alpha > 0 else base_img
            else:
                corrupted_arr = np.array(apply_corruption(base_img, c, 1.0)).astype(np.float32) / 255.0
                blended = (1 - alpha) * clean_arr + alpha * corrupted_arr
                img = Image.fromarray((np.clip(blended, 0, 1) * 255).astype(np.uint8))
            emb = extract_hidden(model, processor, img, prompt)
            d = float(cosine(clean_emb, emb))
            fine_data.append({"alpha": alpha, "distance": d})

        sharpness[c] = fine_data
        # Find transition point
        nonzero = [(x["alpha"], x["distance"]) for x in fine_data if x["distance"] > 0]
        if nonzero:
            print(f"    first nonzero at alpha={nonzero[0][0]}, d={nonzero[0][1]:.8f}")
        else:
            print(f"    no detection in fine range")

    results["boundary_sharpness"] = sharpness

    # Part 5: Action change threshold analysis
    print("\n=== Part 5: Action Change Threshold ===")
    action_threshold = {}

    for c in corruptions:
        print(f"  {c}...")
        # Find the alpha where actions first change
        test_alphas = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
        action_data = []

        for alpha in test_alphas:
            if c == 'blur':
                img = apply_corruption(base_img, c, alpha) if alpha > 0 else base_img
            else:
                corrupted_arr = np.array(apply_corruption(base_img, c, 1.0)).astype(np.float32) / 255.0
                blended = (1 - alpha) * clean_arr + alpha * corrupted_arr
                img = Image.fromarray((np.clip(blended, 0, 1) * 255).astype(np.uint8))

            emb = extract_hidden(model, processor, img, prompt)
            d = float(cosine(clean_emb, emb))
            actions = get_action_tokens(model, processor, img, prompt)
            n_changed = sum(1 for a, b in zip(clean_actions, actions) if a != b)
            total_shift = sum(abs(a - b) for a, b in zip(clean_actions, actions) if a >= 0 and b >= 0)

            action_data.append({
                "alpha": alpha,
                "distance": d,
                "actions": actions,
                "n_changed": n_changed,
                "total_shift": total_shift,
            })

        action_threshold[c] = action_data
        first_change = next((d for d in action_data if d["n_changed"] > 0), None)
        if first_change:
            print(f"    first action change at alpha={first_change['alpha']}, "
                  f"d={first_change['distance']:.6f}, {first_change['n_changed']}/7 dims")

    results["action_threshold"] = action_threshold

    # Compute detection-before-action margin
    print("\n=== Detection-Before-Action Margins ===")
    margins = {}
    for c in corruptions:
        interp = interpolation[c]
        first_detect = next((d["alpha"] for d in interp if d["distance"] > 0), None)
        first_action = next((d["alpha"] for d in interp if d["n_actions_changed"] > 0), None)
        if first_detect is not None and first_action is not None and first_action > 0:
            margin = first_action / first_detect
            margins[c] = {
                "detect_alpha": first_detect,
                "action_alpha": first_action,
                "margin_ratio": margin,
            }
            print(f"  {c}: detect@{first_detect}, action@{first_action}, margin={margin:.1f}x")

    results["detection_action_margins"] = margins

    # Save
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(v) for v in obj]
        return obj

    ts = results["timestamp"]
    out_path = f"experiments/boundary_{ts}.json"
    with open(out_path, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
