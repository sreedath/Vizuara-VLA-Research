"""
Cross-Prompt Calibration Transfer.

Tests whether a centroid calibrated with one prompt still detects OOD
when test images are processed with a different prompt.  Experiment 115
showed all prompts achieve AUROC=1.000 individually — but does the
centroid transfer across prompts?

Experiment 122 in the CalibDrive series.
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
    rng = np.random.default_rng(idx * 15001)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]
    img[SIZE[0]//2:] = [80, 80, 80]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [255, 255, 255]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_urban(idx):
    rng = np.random.default_rng(idx * 15002)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [135, 206, 235]
    img[SIZE[0]//3:SIZE[0]//2] = [139, 119, 101]
    img[SIZE[0]//2:] = [60, 60, 60]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_noise(idx):
    rng = np.random.default_rng(idx * 15003)
    return rng.integers(0, 256, (*SIZE, 3), dtype=np.uint8)

def create_indoor(idx):
    rng = np.random.default_rng(idx * 15004)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:] = [200, 180, 160]
    img[SIZE[0]//2:, :] = [100, 80, 60]
    img[:SIZE[0]//3, SIZE[1]//3:2*SIZE[1]//3] = [150, 200, 255]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_twilight_highway(idx):
    rng = np.random.default_rng(idx * 15010)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [70, 50, 80]
    img[SIZE[0]//2:] = [60, 60, 60]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [200, 200, 100]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_snow(idx):
    rng = np.random.default_rng(idx * 15014)
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
    print("CROSS-PROMPT CALIBRATION TRANSFER", flush=True)
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
        'lane_keep': "In: What action should the robot take to stay in the current lane?\nOut:",
        'slow_down': "In: What action should the robot take to slow down to 10 m/s?\nOut:",
        'navigate': "In: What action should the robot take to navigate this road?\nOut:",
        'avoid_obstacle': "In: What action should the robot take to avoid obstacles ahead?\nOut:",
    }

    categories = {
        'highway': (create_highway, 'ID'),
        'urban': (create_urban, 'ID'),
        'noise': (create_noise, 'OOD'),
        'indoor': (create_indoor, 'OOD'),
        'twilight': (create_twilight_highway, 'OOD'),
        'snow': (create_snow, 'OOD'),
    }

    N_CAL = 8   # ID images for calibration per category
    N_TEST = 5  # ID images for testing per category
    N_OOD = 8   # OOD images per category

    # Pre-generate images
    id_cal_images = []
    id_test_images = []
    ood_images = []

    for cat_name, (fn, group) in categories.items():
        if group == 'ID':
            for i in range(N_CAL):
                id_cal_images.append((cat_name, Image.fromarray(fn(i + 2200))))
            for i in range(N_TEST):
                id_test_images.append((cat_name, Image.fromarray(fn(i + 2200 + N_CAL))))
        else:
            for i in range(N_OOD):
                ood_images.append((cat_name, Image.fromarray(fn(i + 2200))))

    print(f"\nID cal: {len(id_cal_images)}, ID test: {len(id_test_images)}, OOD: {len(ood_images)}", flush=True)
    print(f"Prompts: {len(prompts)}", flush=True)
    total_inferences = len(prompts) * (len(id_cal_images) + len(id_test_images) + len(ood_images))
    print(f"Total inferences: {total_inferences}", flush=True)

    # Collect embeddings for every (image, prompt) pair
    print("\n--- Collecting embeddings per prompt ---", flush=True)
    prompt_embeddings = {}

    for p_name, p_text in prompts.items():
        print(f"\n  Prompt: {p_name}", flush=True)

        cal_embeds = []
        for cat_name, img in id_cal_images:
            h = extract_hidden(model, processor, img, p_text)
            if h is not None:
                cal_embeds.append(h)
        cal_embeds = np.array(cal_embeds)

        test_id_embeds = []
        for cat_name, img in id_test_images:
            h = extract_hidden(model, processor, img, p_text)
            if h is not None:
                test_id_embeds.append(h)
        test_id_embeds = np.array(test_id_embeds)

        test_ood_embeds = []
        for cat_name, img in ood_images:
            h = extract_hidden(model, processor, img, p_text)
            if h is not None:
                test_ood_embeds.append(h)
        test_ood_embeds = np.array(test_ood_embeds)

        centroid = np.mean(cal_embeds, axis=0)

        prompt_embeddings[p_name] = {
            'centroid': centroid,
            'cal_embeds': cal_embeds,
            'test_id_embeds': test_id_embeds,
            'test_ood_embeds': test_ood_embeds,
        }
        print(f"    cal={len(cal_embeds)}, test_id={len(test_id_embeds)}, test_ood={len(test_ood_embeds)}", flush=True)

    # Cross-prompt transfer matrix
    print("\n--- Cross-Prompt Transfer Matrix ---", flush=True)
    prompt_names = list(prompts.keys())
    n_prompts = len(prompt_names)
    auroc_matrix = np.zeros((n_prompts, n_prompts))
    d_matrix = np.zeros((n_prompts, n_prompts))

    for i, cal_prompt in enumerate(prompt_names):
        centroid = prompt_embeddings[cal_prompt]['centroid']
        for j, test_prompt in enumerate(prompt_names):
            id_embeds = prompt_embeddings[test_prompt]['test_id_embeds']
            ood_embeds = prompt_embeddings[test_prompt]['test_ood_embeds']

            id_scores = np.array([cosine_dist(e, centroid) for e in id_embeds])
            ood_scores = np.array([cosine_dist(e, centroid) for e in ood_embeds])

            all_scores = np.concatenate([id_scores, ood_scores])
            all_labels = np.array([0]*len(id_scores) + [1]*len(ood_scores))

            auroc = float(roc_auc_score(all_labels, all_scores))
            d = float((np.mean(ood_scores) - np.mean(id_scores)) / (np.std(id_scores) + 1e-10))

            auroc_matrix[i, j] = auroc
            d_matrix[i, j] = d

    print("\nAUROC Matrix (rows=cal_prompt, cols=test_prompt):", flush=True)
    header = f"{'':20s}" + "".join(f"{p:>14s}" for p in prompt_names)
    print(header, flush=True)
    for i, cal_p in enumerate(prompt_names):
        row = f"{cal_p:20s}" + "".join(f"{auroc_matrix[i,j]:14.4f}" for j in range(n_prompts))
        print(row, flush=True)

    print("\nD-prime Matrix:", flush=True)
    print(header, flush=True)
    for i, cal_p in enumerate(prompt_names):
        row = f"{cal_p:20s}" + "".join(f"{d_matrix[i,j]:14.2f}" for j in range(n_prompts))
        print(row, flush=True)

    # Centroid similarity across prompts
    print("\n--- Centroid Similarity ---", flush=True)
    centroid_sim = np.zeros((n_prompts, n_prompts))
    for i in range(n_prompts):
        for j in range(n_prompts):
            ci = prompt_embeddings[prompt_names[i]]['centroid']
            cj = prompt_embeddings[prompt_names[j]]['centroid']
            sim = float(np.dot(ci, cj) / (np.linalg.norm(ci) * np.linalg.norm(cj) + 1e-10))
            centroid_sim[i, j] = sim

    print("Centroid Cosine Similarity Matrix:", flush=True)
    print(header, flush=True)
    for i, p in enumerate(prompt_names):
        row = f"{p:20s}" + "".join(f"{centroid_sim[i,j]:14.4f}" for j in range(n_prompts))
        print(row, flush=True)

    # Summary statistics
    off_diag_auroc = []
    on_diag_auroc = []
    off_diag_d = []
    on_diag_d = []
    off_diag_sim = []

    for i in range(n_prompts):
        for j in range(n_prompts):
            if i == j:
                on_diag_auroc.append(auroc_matrix[i, j])
                on_diag_d.append(d_matrix[i, j])
            else:
                off_diag_auroc.append(auroc_matrix[i, j])
                off_diag_d.append(d_matrix[i, j])
                off_diag_sim.append(centroid_sim[i, j])

    print(f"\nSame-prompt AUROC:  mean={np.mean(on_diag_auroc):.4f}, min={np.min(on_diag_auroc):.4f}", flush=True)
    print(f"Cross-prompt AUROC: mean={np.mean(off_diag_auroc):.4f}, min={np.min(off_diag_auroc):.4f}", flush=True)
    print(f"Same-prompt d:      mean={np.mean(on_diag_d):.2f}", flush=True)
    print(f"Cross-prompt d:     mean={np.mean(off_diag_d):.2f}", flush=True)
    print(f"Cross-prompt centroid sim: mean={np.mean(off_diag_sim):.4f}, min={np.min(off_diag_sim):.4f}", flush=True)

    # AUROC degradation
    degradation = np.mean(on_diag_auroc) - np.mean(off_diag_auroc)
    print(f"\nAUROC degradation (same - cross): {degradation:.4f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'cross_prompt_calibration',
        'experiment_number': 122,
        'timestamp': timestamp,
        'n_prompts': n_prompts,
        'prompt_names': prompt_names,
        'n_cal_id': len(id_cal_images),
        'n_test_id': len(id_test_images),
        'n_ood': len(ood_images),
        'total_inferences': total_inferences,
        'auroc_matrix': auroc_matrix.tolist(),
        'd_matrix': d_matrix.tolist(),
        'centroid_similarity': centroid_sim.tolist(),
        'summary': {
            'same_prompt_auroc_mean': float(np.mean(on_diag_auroc)),
            'same_prompt_auroc_min': float(np.min(on_diag_auroc)),
            'cross_prompt_auroc_mean': float(np.mean(off_diag_auroc)),
            'cross_prompt_auroc_min': float(np.min(off_diag_auroc)),
            'same_prompt_d_mean': float(np.mean(on_diag_d)),
            'cross_prompt_d_mean': float(np.mean(off_diag_d)),
            'centroid_sim_mean': float(np.mean(off_diag_sim)),
            'centroid_sim_min': float(np.min(off_diag_sim)),
            'auroc_degradation': float(degradation),
        },
    }
    output_path = os.path.join(RESULTS_DIR, f"cross_prompt_calibration_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
