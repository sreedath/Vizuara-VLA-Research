"""
Image Resolution Sensitivity Analysis.

Tests OOD detection at different input image resolutions to determine
how detection degrades when images are downsized before processing.
Tests resolutions from 32x32 to 512x512.

Experiment 118 in the CalibDrive series.
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
    rng = np.random.default_rng(idx * 12001)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]
    img[SIZE[0]//2:] = [80, 80, 80]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [255, 255, 255]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_urban(idx):
    rng = np.random.default_rng(idx * 12002)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [135, 206, 235]
    img[SIZE[0]//3:SIZE[0]//2] = [139, 119, 101]
    img[SIZE[0]//2:] = [60, 60, 60]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_noise(idx):
    rng = np.random.default_rng(idx * 12003)
    return rng.integers(0, 256, (*SIZE, 3), dtype=np.uint8)

def create_indoor(idx):
    rng = np.random.default_rng(idx * 12004)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:] = [200, 180, 160]
    img[SIZE[0]//2:, :] = [100, 80, 60]
    img[:SIZE[0]//3, SIZE[1]//3:2*SIZE[1]//3] = [150, 200, 255]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_twilight_highway(idx):
    rng = np.random.default_rng(idx * 12010)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [70, 50, 80]
    img[SIZE[0]//2:] = [60, 60, 60]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [200, 200, 100]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_snow(idx):
    rng = np.random.default_rng(idx * 12014)
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
    print("IMAGE RESOLUTION SENSITIVITY", flush=True)
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

    prompt = "In: What action should the robot take to drive forward at 25 m/s safely?\nOut:"

    categories = {
        'highway': (create_highway, 'ID'),
        'urban': (create_urban, 'ID'),
        'noise': (create_noise, 'OOD'),
        'indoor': (create_indoor, 'OOD'),
        'twilight': (create_twilight_highway, 'OOD'),
        'snow': (create_snow, 'OOD'),
    }

    resolutions = [32, 64, 128, 256, 384, 512]

    print(f"\n--- Testing {len(resolutions)} resolutions ---", flush=True)
    results = {}

    for res in resolutions:
        print(f"\n  Resolution: {res}x{res}", flush=True)

        cal_embeds = []
        test_embeds = []
        test_labels = []
        per_cat = {}

        for cat_name, (fn, group) in categories.items():
            embeds = []
            for i in range(10):
                # Create at native size, then resize
                img_arr = fn(i + 1900)
                pil = Image.fromarray(img_arr)
                # Downscale then upscale to simulate resolution loss
                pil_down = pil.resize((res, res), Image.BILINEAR)
                pil_up = pil_down.resize(SIZE, Image.BILINEAR)

                h = extract_hidden(model, processor, pil_up, prompt)
                if h is not None:
                    embeds.append(h)

            embeds = np.array(embeds)
            per_cat[cat_name] = {'group': group, 'n_samples': len(embeds)}

            if group == 'ID':
                cal_embeds.extend(embeds[:5])
                for e in embeds[5:]:
                    test_embeds.append(e)
                    test_labels.append(0)
            else:
                for e in embeds:
                    test_embeds.append(e)
                    test_labels.append(1)

        cal_embeds = np.array(cal_embeds)
        test_embeds = np.array(test_embeds)
        test_labels = np.array(test_labels)

        centroid = np.mean(cal_embeds, axis=0)
        scores = np.array([cosine_dist(e, centroid) for e in test_embeds])
        id_scores = scores[test_labels == 0]
        ood_scores = scores[test_labels == 1]

        auroc = float(roc_auc_score(test_labels, scores))
        d = float((np.mean(ood_scores) - np.mean(id_scores)) / (np.std(id_scores) + 1e-10))

        # Cross-resolution centroid similarity with native (256)
        results[str(res)] = {
            'resolution': res,
            'auroc': auroc,
            'd': d,
            'id_score_mean': float(np.mean(id_scores)),
            'id_score_std': float(np.std(id_scores)),
            'ood_score_mean': float(np.mean(ood_scores)),
            'ood_score_std': float(np.std(ood_scores)),
        }
        print(f"    AUROC={auroc:.4f}, d={d:.2f}", flush=True)

    # Cross-resolution centroid comparison
    print("\n--- Cross-resolution centroid similarity ---", flush=True)
    # Re-extract at each resolution for centroid comparison
    centroids = {}
    for res_key, res_data in results.items():
        # Already computed — use results
        pass

    # Re-do just for centroid extraction at base resolution
    base_embeds = []
    for cat_name, (fn, group) in categories.items():
        if group == 'ID':
            for i in range(10):
                img = Image.fromarray(fn(i + 1900))
                h = extract_hidden(model, processor, img, prompt)
                if h is not None:
                    base_embeds.append(h)
    base_centroid = np.mean(base_embeds[:10], axis=0)

    # Compare centroids at each resolution
    centroid_sims = {}
    for res in resolutions:
        res_embeds = []
        for cat_name, (fn, group) in categories.items():
            if group == 'ID':
                for i in range(5):
                    img_arr = fn(i + 1900)
                    pil = Image.fromarray(img_arr)
                    pil_down = pil.resize((res, res), Image.BILINEAR)
                    pil_up = pil_down.resize(SIZE, Image.BILINEAR)
                    h = extract_hidden(model, processor, pil_up, prompt)
                    if h is not None:
                        res_embeds.append(h)
        if res_embeds:
            res_centroid = np.mean(res_embeds, axis=0)
            sim = float(np.dot(res_centroid, base_centroid) /
                       (np.linalg.norm(res_centroid) * np.linalg.norm(base_centroid) + 1e-10))
            centroid_sims[str(res)] = sim
            print(f"  {res}x{res} vs 256x256: sim={sim:.6f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'resolution_sensitivity',
        'experiment_number': 118,
        'timestamp': timestamp,
        'resolutions': resolutions,
        'results': results,
        'centroid_similarity_to_native': centroid_sims,
    }
    output_path = os.path.join(RESULTS_DIR, f"resolution_sensitivity_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
