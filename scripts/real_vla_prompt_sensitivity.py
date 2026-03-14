"""
Prompt Sensitivity Analysis.

Tests whether OOD detection signals depend on the text prompt used.
If detection is prompt-invariant, it's more practical for deployment.

Tests 5 different prompts:
1. Original driving prompt
2. Simplified prompt
3. Different speed
4. Lane change task
5. Emergency braking task

Experiment 71 in the CalibDrive series.
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

def create_blackout(idx):
    return np.zeros((*SIZE, 3), dtype=np.uint8)


PROMPTS = {
    'original': "In: What action should the robot take to drive forward at 25 m/s safely?\nOut:",
    'simplified': "In: What action should the robot take to drive forward?\nOut:",
    'fast': "In: What action should the robot take to drive forward at 60 m/s safely?\nOut:",
    'lane_change': "In: What action should the robot take to change lanes to the left?\nOut:",
    'brake': "In: What action should the robot take to emergency brake?\nOut:",
}


def extract_signals(model, processor, image, prompt):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    result = {}
    with torch.no_grad():
        fwd = model(**inputs, output_attentions=True, output_hidden_states=True)

    if hasattr(fwd, 'attentions') and fwd.attentions:
        attn = fwd.attentions[-1][0].float().cpu().numpy()
        n_heads = attn.shape[0]
        last_attn = attn[:, -1, :]
        result['attn_max'] = float(np.mean([np.max(last_attn[h]) for h in range(n_heads)]))

    if hasattr(fwd, 'hidden_states') and fwd.hidden_states:
        result['hidden'] = fwd.hidden_states[-1][0, -1, :].float().cpu().numpy()

    return result


def cosine_dist(a, b):
    return 1.0 - float(np.dot(a / (np.linalg.norm(a) + 1e-10),
                               b / (np.linalg.norm(b) + 1e-10)))


def main():
    print("=" * 70, flush=True)
    print("PROMPT SENSITIVITY ANALYSIS", flush=True)
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

    test_images = {
        'highway': [(create_highway, i) for i in range(600, 608)],
        'urban': [(create_urban, i) for i in range(600, 608)],
        'noise': [(create_noise, i) for i in range(600, 606)],
        'indoor': [(create_indoor, i) for i in range(600, 606)],
        'blackout': [(create_blackout, i) for i in range(600, 604)],
    }

    results = {}
    total_infs = sum(len(imgs) for imgs in test_images.values()) * len(PROMPTS)
    cnt = 0

    for prompt_name, prompt_text in PROMPTS.items():
        print(f"\n  Prompt: {prompt_name}", flush=True)
        results[prompt_name] = {}

        # Calibrate with this prompt
        cal_hidden = []
        for fn in [create_highway, create_urban]:
            for i in range(10):
                sig = extract_signals(model, processor,
                                      Image.fromarray(fn(i + 9000)), prompt_text)
                if 'hidden' in sig:
                    cal_hidden.append(sig['hidden'])
        centroid = np.mean(cal_hidden, axis=0)
        print(f"    Calibrated with {len(cal_hidden)} samples", flush=True)

        # Test
        all_labels = []
        all_cosines = []
        all_attns = []
        per_scene = {}

        for scene, img_specs in test_images.items():
            scene_cosines = []
            scene_attns = []
            is_ood = scene not in ['highway', 'urban']

            for fn, idx in img_specs:
                cnt += 1
                sig = extract_signals(model, processor,
                                      Image.fromarray(fn(idx)), prompt_text)

                cos_d = cosine_dist(sig['hidden'], centroid) if 'hidden' in sig else 0
                attn_m = sig.get('attn_max', 0)

                scene_cosines.append(cos_d)
                scene_attns.append(attn_m)
                all_labels.append(1 if is_ood else 0)
                all_cosines.append(cos_d)
                all_attns.append(attn_m)

                if cnt % 20 == 0:
                    print(f"    [{cnt}/{total_infs}] {prompt_name}/{scene}", flush=True)

            per_scene[scene] = {
                'cosine_mean': float(np.mean(scene_cosines)),
                'cosine_std': float(np.std(scene_cosines)),
                'attn_max_mean': float(np.mean(scene_attns)),
                'attn_max_std': float(np.std(scene_attns)),
                'is_ood': is_ood,
            }

        cos_auroc = roc_auc_score(all_labels, all_cosines)
        attn_auroc = roc_auc_score(all_labels, all_attns)

        results[prompt_name] = {
            'cosine_auroc': float(cos_auroc),
            'attn_auroc': float(attn_auroc),
            'per_scene': per_scene,
        }
        print(f"    Cosine AUROC: {cos_auroc:.3f}, Attn AUROC: {attn_auroc:.3f}", flush=True)

    # Cross-prompt centroid analysis
    print("\n  Cross-prompt centroid distances:", flush=True)
    centroids = {}
    for prompt_name, prompt_text in PROMPTS.items():
        cal_hidden = []
        for fn in [create_highway, create_urban]:
            for i in range(10):
                sig = extract_signals(model, processor,
                                      Image.fromarray(fn(i + 9000)), prompt_text)
                if 'hidden' in sig:
                    cal_hidden.append(sig['hidden'])
        centroids[prompt_name] = np.mean(cal_hidden, axis=0)

    centroid_dists = {}
    for p1 in PROMPTS:
        for p2 in PROMPTS:
            if p1 < p2:
                d = cosine_dist(centroids[p1], centroids[p2])
                centroid_dists[f"{p1}_vs_{p2}"] = float(d)
                print(f"    {p1} vs {p2}: {d:.4f}", flush=True)

    # Summary
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 70, flush=True)
    for pn in PROMPTS:
        r = results[pn]
        print(f"  {pn:<15}: cos={r['cosine_auroc']:.3f}, attn={r['attn_auroc']:.3f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'prompt_sensitivity',
        'experiment_number': 71,
        'timestamp': timestamp,
        'prompts': PROMPTS,
        'results': results,
        'centroid_distances': centroid_dists,
    }
    output_path = os.path.join(RESULTS_DIR, f"prompt_sensitivity_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
