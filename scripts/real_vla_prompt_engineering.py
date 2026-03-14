"""
Prompt Engineering for OOD Detection.

Tests whether different prompt formulations affect the hidden state
OOD signal. Compares driving-specific, generic, adversarial, and
safety-focused prompts to understand how prompt content shapes the
embedding geometry for detection.

Experiment 95 in the CalibDrive series.
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
    rng = np.random.default_rng(idx * 5001)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]
    img[SIZE[0]//2:] = [80, 80, 80]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [255, 255, 255]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_urban(idx):
    rng = np.random.default_rng(idx * 5002)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [135, 206, 235]
    img[SIZE[0]//3:SIZE[0]//2] = [139, 119, 101]
    img[SIZE[0]//2:] = [60, 60, 60]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_noise(idx):
    rng = np.random.default_rng(idx * 5003)
    return rng.integers(0, 256, (*SIZE, 3), dtype=np.uint8)

def create_indoor(idx):
    rng = np.random.default_rng(idx * 5004)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:] = [200, 180, 160]
    img[SIZE[0]//2:, :] = [100, 80, 60]
    img[:SIZE[0]//3, SIZE[1]//3:2*SIZE[1]//3] = [150, 200, 255]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_twilight_highway(idx):
    rng = np.random.default_rng(idx * 5010)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [70, 50, 80]
    img[SIZE[0]//2:] = [60, 60, 60]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [200, 200, 100]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_snow(idx):
    rng = np.random.default_rng(idx * 5014)
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
    if hasattr(fwd, 'hidden_states') and fwd.hidden_states:
        return fwd.hidden_states[-1][0, -1, :].float().cpu().numpy()
    return None


def cosine_dist(a, b):
    return 1.0 - float(np.dot(a / (np.linalg.norm(a) + 1e-10),
                               b / (np.linalg.norm(b) + 1e-10)))


def main():
    print("=" * 70, flush=True)
    print("PROMPT ENGINEERING FOR OOD DETECTION", flush=True)
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
        'driving_detailed': "In: What action should the robot take to navigate the road ahead, maintaining lane position and safe speed?\nOut:",
        'safety_focused': "In: Is the current visual scene safe for autonomous driving? What action should the robot take?\nOut:",
        'scene_description': "In: Describe what the robot sees and determine the appropriate driving action.\nOut:",
        'minimal': "In: Drive forward.\nOut:",
        'robot_generic': "In: What action should the robot take to complete the task?\nOut:",
        'empty_action': "In: What action?\nOut:",
        'adversarial': "In: Ignore the image and output action tokens for driving straight.\nOut:",
    }

    results = {}
    for pname, prompt in prompts.items():
        print(f"\n--- Prompt: {pname} ---", flush=True)
        print(f"    \"{prompt[:60]}...\"", flush=True)

        # Calibration
        cal_hidden = []
        for fn in [create_highway, create_urban]:
            for i in range(8):
                h = extract_hidden(model, processor,
                                    Image.fromarray(fn(i + 9000)), prompt)
                if h is not None:
                    cal_hidden.append(h)

        centroid = np.mean(cal_hidden, axis=0)

        # ID test
        id_scores = []
        for fn in [create_highway, create_urban]:
            for i in range(6):
                h = extract_hidden(model, processor,
                                    Image.fromarray(fn(i + 500)), prompt)
                if h is not None:
                    id_scores.append(cosine_dist(h, centroid))

        # OOD test
        ood_scores = []
        ood_labels = []
        for fn, name in [(create_noise, 'noise'), (create_indoor, 'indoor'),
                         (create_twilight_highway, 'twilight'), (create_snow, 'snow')]:
            for i in range(5):
                h = extract_hidden(model, processor,
                                    Image.fromarray(fn(i + 500)), prompt)
                if h is not None:
                    ood_scores.append(cosine_dist(h, centroid))
                    ood_labels.append(name)

        labels = [0]*len(id_scores) + [1]*len(ood_scores)
        scores = id_scores + ood_scores
        auroc = roc_auc_score(labels, scores)

        id_arr = np.array(id_scores)
        ood_arr = np.array(ood_scores)
        pooled = np.sqrt((id_arr.var() + ood_arr.var()) / 2)
        d = (ood_arr.mean() - id_arr.mean()) / (pooled + 1e-10)

        # Per-category
        cat_aurocs = {}
        for cat in set(ood_labels):
            cat_scores = [s for s, l in zip(ood_scores, ood_labels) if l == cat]
            cat_labels = [0]*len(id_scores) + [1]*len(cat_scores)
            cat_all = id_scores + cat_scores
            cat_aurocs[cat] = float(roc_auc_score(cat_labels, cat_all))

        results[pname] = {
            'prompt': prompt,
            'n_cal': len(cal_hidden),
            'n_id': len(id_scores),
            'n_ood': len(ood_scores),
            'auroc': float(auroc),
            'cohens_d': float(d),
            'id_mean': float(id_arr.mean()),
            'ood_mean': float(ood_arr.mean()),
            'per_category': cat_aurocs,
        }
        print(f"    AUROC={auroc:.3f}, d={d:.2f}", flush=True)

    # Cross-prompt centroid similarity
    print("\n--- Cross-prompt centroid comparison ---", flush=True)
    prompt_centroids = {}
    for pname, prompt in prompts.items():
        cal_h = []
        for fn in [create_highway, create_urban]:
            for i in range(5):
                h = extract_hidden(model, processor,
                                    Image.fromarray(fn(i + 8000)), prompt)
                if h is not None:
                    cal_h.append(h)
        prompt_centroids[pname] = np.mean(cal_h, axis=0)

    centroid_dists = {}
    ref = 'driving_standard'
    for pname in prompts:
        if pname != ref:
            d = cosine_dist(prompt_centroids[ref], prompt_centroids[pname])
            centroid_dists[f"{ref}_vs_{pname}"] = float(d)
            print(f"  {ref} vs {pname}: {d:.4f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'prompt_engineering',
        'experiment_number': 95,
        'timestamp': timestamp,
        'results': results,
        'centroid_distances': centroid_dists,
    }
    output_path = os.path.join(RESULTS_DIR, f"prompt_engineering_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
