"""
Prompt-Conditioned Embedding Geometry.

Analyzes how different text prompts shape the embedding space geometry.
For each prompt, we measure:
1. ID cluster radius (intra-class spread)  
2. OOD-to-ID centroid distance
3. Separation ratio (gap / radius)
4. Whether prompts create qualitatively different embedding landscapes

Key question: Do different prompts map inputs to different regions of 
embedding space, or the same region?

Experiment 135 in the CalibDrive series.
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
    rng = np.random.default_rng(idx * 28001)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]
    img[SIZE[0]//2:] = [80, 80, 80]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [255, 255, 255]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_urban(idx):
    rng = np.random.default_rng(idx * 28002)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [135, 206, 235]
    img[SIZE[0]//3:SIZE[0]//2] = [139, 119, 101]
    img[SIZE[0]//2:] = [60, 60, 60]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_noise(idx):
    rng = np.random.default_rng(idx * 28003)
    return rng.integers(0, 256, (*SIZE, 3), dtype=np.uint8)

def create_indoor(idx):
    rng = np.random.default_rng(idx * 28004)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:] = [200, 180, 160]
    img[SIZE[0]//2:, :] = [100, 80, 60]
    img[:SIZE[0]//3, SIZE[1]//3:2*SIZE[1]//3] = [150, 200, 255]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def extract_hidden(model, processor, image, prompt):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    if not hasattr(fwd, 'hidden_states') or not fwd.hidden_states:
        return None
    return fwd.hidden_states[-1][0, -1, :].float().cpu().numpy()


def cosine_distance(a, b):
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def main():
    print("=" * 70, flush=True)
    print("PROMPT-CONDITIONED EMBEDDING GEOMETRY", flush=True)
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
        'drive_forward': "In: What action should the robot take to drive forward at 25 m/s safely?\nOut:",
        'navigate': "In: What action should the robot take to navigate this road?\nOut:",
        'follow_lane': "In: What action should the robot take to follow the lane markings?\nOut:",
        'stop': "In: What action should the robot take to stop the vehicle?\nOut:",
        'turn_left': "In: What action should the robot take to turn left at the intersection?\nOut:",
        'park': "In: What action should the robot take to park the vehicle?\nOut:",
        'avoid_obstacle': "In: What action should the robot take to avoid the obstacle ahead?\nOut:",
        'reverse': "In: What action should the robot take to reverse the vehicle?\nOut:",
    }

    # Collect embeddings for each prompt
    categories = {
        'highway': (create_highway, 'ID'),
        'urban': (create_urban, 'ID'),
        'noise': (create_noise, 'OOD'),
        'indoor': (create_indoor, 'OOD'),
    }

    results = {}
    all_centroids = {}

    for prompt_name, prompt_text in prompts.items():
        print(f"\n--- Prompt: {prompt_name} ---", flush=True)
        id_embeds = []
        ood_embeds = []

        for cat_name, (fn, group) in categories.items():
            for i in range(8):
                h = extract_hidden(model, processor, Image.fromarray(fn(i + 3300)), prompt_text)
                if h is not None:
                    if group == 'ID':
                        id_embeds.append(h)
                    else:
                        ood_embeds.append(h)

        id_arr = np.array(id_embeds)
        ood_arr = np.array(ood_embeds)
        centroid = np.mean(id_arr, axis=0)
        all_centroids[prompt_name] = centroid

        # ID radius
        id_dists = [cosine_distance(e, centroid) for e in id_arr]
        id_radius = float(np.mean(id_dists))
        id_max = float(np.max(id_dists))

        # OOD distances
        ood_dists = [cosine_distance(e, centroid) for e in ood_arr]
        ood_mean = float(np.mean(ood_dists))
        ood_min = float(np.min(ood_dists))

        # Separation ratio
        gap = ood_min - id_max
        ratio = gap / (id_radius + 1e-10)

        # AUROC
        labels = np.array([0]*len(id_dists) + [1]*len(ood_dists))
        scores = np.array(id_dists + ood_dists)
        auroc = float(roc_auc_score(labels, scores))
        d = float((np.mean(ood_dists) - np.mean(id_dists)) / (np.std(id_dists) + 1e-10))

        results[prompt_name] = {
            'n_id': len(id_embeds),
            'n_ood': len(ood_embeds),
            'id_radius': id_radius,
            'id_max': id_max,
            'ood_mean': ood_mean,
            'ood_min': ood_min,
            'gap': gap,
            'separation_ratio': ratio,
            'auroc': auroc,
            'd_prime': d,
        }
        print(f"  ID radius={id_radius:.4f}, OOD mean={ood_mean:.4f}, gap={gap:.4f}, ratio={ratio:.1f}, AUROC={auroc:.4f}, d={d:.1f}", flush=True)

    # Cross-prompt centroid similarity
    print("\n--- Cross-Prompt Centroid Similarity ---", flush=True)
    prompt_names = list(all_centroids.keys())
    centroid_sim = np.zeros((len(prompt_names), len(prompt_names)))
    for i, p1 in enumerate(prompt_names):
        for j, p2 in enumerate(prompt_names):
            sim = float(np.dot(all_centroids[p1], all_centroids[p2]) /
                       (np.linalg.norm(all_centroids[p1]) * np.linalg.norm(all_centroids[p2]) + 1e-10))
            centroid_sim[i, j] = sim
    print(f"  Mean off-diagonal similarity: {float(np.mean(centroid_sim[np.triu_indices(len(prompt_names), k=1)])):.4f}", flush=True)
    print(f"  Min off-diagonal similarity: {float(np.min(centroid_sim[np.triu_indices(len(prompt_names), k=1)])):.4f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'prompt_embedding_geometry',
        'experiment_number': 135,
        'timestamp': timestamp,
        'prompts': {k: v for k, v in prompts.items()},
        'results': results,
        'centroid_similarity': centroid_sim.tolist(),
        'prompt_names': prompt_names,
    }
    output_path = os.path.join(RESULTS_DIR, f"prompt_embedding_geometry_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
