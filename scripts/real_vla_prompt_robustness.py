"""
Prompt Robustness Analysis.

Tests OOD detection performance across 10 different prompt formulations
to measure sensitivity to prompt wording. Includes driving-specific,
generic robot, adversarial, and minimal prompts.

Experiment 115 in the CalibDrive series.
"""
import os
import json
import datetime
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

RESULTS_DIR = "/workspace/Vizuara-VLA-Research/experiments"
os.makedirs(RESULTS_DIR, exist_ok=True)
SIZE = (256, 256)


def create_highway(idx):
    rng = np.random.default_rng(idx * 9001)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]
    img[SIZE[0]//2:] = [80, 80, 80]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [255, 255, 255]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_urban(idx):
    rng = np.random.default_rng(idx * 9002)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [135, 206, 235]
    img[SIZE[0]//3:SIZE[0]//2] = [139, 119, 101]
    img[SIZE[0]//2:] = [60, 60, 60]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_noise(idx):
    rng = np.random.default_rng(idx * 9003)
    return rng.integers(0, 256, (*SIZE, 3), dtype=np.uint8)

def create_indoor(idx):
    rng = np.random.default_rng(idx * 9004)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:] = [200, 180, 160]
    img[SIZE[0]//2:, :] = [100, 80, 60]
    img[:SIZE[0]//3, SIZE[1]//3:2*SIZE[1]//3] = [150, 200, 255]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_twilight_highway(idx):
    rng = np.random.default_rng(idx * 9010)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [70, 50, 80]
    img[SIZE[0]//2:] = [60, 60, 60]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [200, 200, 100]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_snow(idx):
    rng = np.random.default_rng(idx * 9014)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [200, 200, 210]
    img[SIZE[0]//2:] = [220, 220, 230]
    img[SIZE[0]//2:, SIZE[1]//2-2:SIZE[1]//2+2] = [180, 180, 190]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def extract_hidden(model, processor, image, prompt):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    if not hasattr(fwd, 'hidden_states') or not fwd.hidden_states:
        return None
    return fwd.hidden_states[-1][0, -1, :].float().cpu().numpy()


def cosine_dist(a, b):
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def main():
    print("=" * 70, flush=True)
    print("PROMPT ROBUSTNESS ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    from transformers import AutoModelForVision2Seq, AutoProcessor
    print("Loading OpenVLA-7B...", flush=True)
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        "openvla/openvla-7b", trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.", flush=True)

    prompts = {
        'driving_standard': "In: What action should the robot take to drive forward at 25 m/s safely?\nOut:",
        'driving_simple': "In: What action should the robot take to drive?\nOut:",
        'driving_speed': "In: What action should the robot take to drive at 60 mph on the highway?\nOut:",
        'driving_stop': "In: What action should the robot take to stop the vehicle?\nOut:",
        'robot_generic': "In: What action should the robot take to pick up the red block?\nOut:",
        'robot_navigate': "In: What action should the robot take to navigate to the kitchen?\nOut:",
        'minimal': "In: What action should the robot take?\nOut:",
        'empty_task': "In: What action should the robot take to complete the task?\nOut:",
        'adversarial_long': "In: What action should the robot take to carefully and precisely drive forward at exactly 25.0 m/s while maintaining perfect lane centering and optimal following distance?\nOut:",
        'adversarial_unrelated': "In: What action should the robot take to solve the math equation x^2 + 3x - 4 = 0?\nOut:",
    }

    categories = {
        'highway': (create_highway, 'ID'),
        'urban': (create_urban, 'ID'),
        'noise': (create_noise, 'OOD'),
        'indoor': (create_indoor, 'OOD'),
        'twilight': (create_twilight_highway, 'OOD'),
        'snow': (create_snow, 'OOD'),
    }

    # For each prompt, collect embeddings and evaluate
    results = {}
    all_centroids = {}  # to compare centroids across prompts

    for prompt_name, prompt_text in prompts.items():
        print(f"\n--- Prompt: {prompt_name} ---", flush=True)
        print(f"  \"{prompt_text[:60]}...\"", flush=True)

        embeddings = {}
        for cat_name, (fn, group) in categories.items():
            embeds = []
            for i in range(10):
                h = extract_hidden(model, processor, Image.fromarray(fn(i + 1600)), prompt_text)
                if h is not None:
                    embeds.append(h)
            embeddings[cat_name] = {'embeds': np.array(embeds), 'group': group}

        # Cal: first 5 of each ID, Test: rest
        cal_embeds = []
        test_embeds = []
        test_labels = []

        for cat_name, data in embeddings.items():
            if data['group'] == 'ID':
                cal_embeds.extend(data['embeds'][:5])
                for e in data['embeds'][5:]:
                    test_embeds.append(e)
                    test_labels.append(0)
            else:
                for e in data['embeds']:
                    test_embeds.append(e)
                    test_labels.append(1)

        cal_embeds = np.array(cal_embeds)
        test_embeds = np.array(test_embeds)
        test_labels = np.array(test_labels)

        centroid = np.mean(cal_embeds, axis=0)
        all_centroids[prompt_name] = centroid

        scores = np.array([cosine_dist(e, centroid) for e in test_embeds])
        id_scores = scores[test_labels == 0]
        ood_scores = scores[test_labels == 1]

        auroc = float(roc_auc_score(test_labels, scores))
        d = float((np.mean(ood_scores) - np.mean(id_scores)) / (np.std(id_scores) + 1e-10))

        # Per-category scores
        per_cat = {}
        for cat_name, data in embeddings.items():
            cat_scores = [cosine_dist(e, centroid) for e in data['embeds']]
            per_cat[cat_name] = {
                'mean': float(np.mean(cat_scores)),
                'std': float(np.std(cat_scores)),
                'group': data['group'],
            }

        results[prompt_name] = {
            'prompt': prompt_text,
            'auroc': auroc,
            'd': d,
            'id_score_mean': float(np.mean(id_scores)),
            'ood_score_mean': float(np.mean(ood_scores)),
            'per_category': per_cat,
        }
        print(f"  AUROC={auroc:.4f}, d={d:.2f}", flush=True)

    # Cross-prompt centroid similarity
    print("\n--- Cross-prompt centroid similarity ---", flush=True)
    prompt_names = list(all_centroids.keys())
    centroid_sims = {}
    for i, p1 in enumerate(prompt_names):
        for j, p2 in enumerate(prompt_names):
            if j <= i:
                continue
            sim = float(np.dot(all_centroids[p1], all_centroids[p2]) /
                       (np.linalg.norm(all_centroids[p1]) * np.linalg.norm(all_centroids[p2]) + 1e-10))
            centroid_sims[f"{p1}_vs_{p2}"] = sim

    # Summary stats
    all_sims = list(centroid_sims.values())
    print(f"  Mean similarity: {np.mean(all_sims):.6f}", flush=True)
    print(f"  Min similarity: {np.min(all_sims):.6f}", flush=True)
    print(f"  Max similarity: {np.max(all_sims):.6f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'prompt_robustness',
        'experiment_number': 115,
        'timestamp': timestamp,
        'n_prompts': len(prompts),
        'results': results,
        'centroid_similarity': centroid_sims,
        'centroid_sim_stats': {
            'mean': float(np.mean(all_sims)),
            'min': float(np.min(all_sims)),
            'max': float(np.max(all_sims)),
            'std': float(np.std(all_sims)),
        },
    }
    output_path = os.path.join(RESULTS_DIR, f"prompt_robustness_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
