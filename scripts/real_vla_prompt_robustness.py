"""
Prompt Robustness for OOD Detection.

Tests whether the cosine distance OOD signal is robust across
different instruction prompts. If the signal depends heavily on
the prompt wording, it limits practical deployability.

Tests 5 different prompts:
1. Original driving prompt
2. Modified speed prompt
3. Cautious driving prompt
4. Simple action prompt
5. Adversarial/unusual prompt

Experiment 54 in the CalibDrive series.
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
    img[SIZE[0]//2:] = [139, 90, 43]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_inverted(idx):
    return 255 - create_highway(idx + 3000)

def create_blackout(idx):
    return np.zeros((*SIZE, 3), dtype=np.uint8)


def extract_hidden(model, processor, image, prompt):
    """Extract hidden state only (for speed)."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=7, do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
    if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
        last_step = outputs.hidden_states[-1]
        if isinstance(last_step, tuple):
            hidden = last_step[-1][0, -1, :].float().cpu().numpy()
        else:
            hidden = last_step[0, -1, :].float().cpu().numpy()
    else:
        hidden = np.zeros(4096)
    return hidden


def cosine_dist(a, b):
    return 1.0 - float(np.dot(a / (np.linalg.norm(a) + 1e-10),
                               b / (np.linalg.norm(b) + 1e-10)))


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

    # Define prompts
    prompts = {
        'original': "In: What action should the robot take to drive forward at 25 m/s safely?\nOut:",
        'speed_50': "In: What action should the robot take to drive forward at 50 m/s safely?\nOut:",
        'cautious': "In: You are driving carefully. What is the safe driving action?\nOut:",
        'simple': "In: Navigate safely. What action to take?\nOut:",
        'different': "In: Predict the driving action to maintain safe driving at 25 m/s.\nOut:",
    }

    # Test images
    test_fns = {
        'highway': (create_highway, False, 8),
        'urban': (create_urban, False, 8),
        'noise': (create_noise, True, 6),
        'indoor': (create_indoor, True, 6),
        'inverted': (create_inverted, True, 6),
        'blackout': (create_blackout, True, 6),
    }

    # For each prompt: calibrate separately (using that prompt's hidden states)
    # then evaluate
    prompt_results = {}
    total_inferences = 0

    for prompt_name, prompt_text in prompts.items():
        print(f"\n{'='*60}", flush=True)
        print(f"Prompt: {prompt_name}", flush=True)
        print(f"  '{prompt_text[:60]}...'", flush=True)

        # Calibration with this prompt
        cal_hidden = []
        for fn in [create_highway, create_urban]:
            for i in range(10):
                h = extract_hidden(model, processor,
                                   Image.fromarray(fn(i + 9000)), prompt_text)
                cal_hidden.append(h)
                total_inferences += 1

        centroid = np.mean(cal_hidden, axis=0)
        cal_cos = [cosine_dist(h, centroid) for h in cal_hidden]
        print(f"  Cal: mean={np.mean(cal_cos):.4f}, max={np.max(cal_cos):.4f}", flush=True)

        # Test
        test_data = []
        for scene, (fn, is_ood, n) in test_fns.items():
            for i in range(n):
                h = extract_hidden(model, processor,
                                   Image.fromarray(fn(i + 200)), prompt_text)
                cos = cosine_dist(h, centroid)
                test_data.append({
                    'scenario': scene,
                    'is_ood': is_ood,
                    'cos_dist': cos,
                })
                total_inferences += 1

        easy = [r for r in test_data if not r['is_ood']]
        ood = [r for r in test_data if r['is_ood']]
        labels = [0]*len(easy) + [1]*len(ood)
        scores = [r['cos_dist'] for r in easy + ood]
        auroc = roc_auc_score(labels, scores)

        id_mean = np.mean([r['cos_dist'] for r in easy])
        ood_mean = np.mean([r['cos_dist'] for r in ood])

        print(f"  AUROC: {auroc:.3f}", flush=True)
        print(f"  ID cos: {id_mean:.4f}, OOD cos: {ood_mean:.4f}, "
              f"sep={ood_mean-id_mean:.4f}", flush=True)

        # Per-scenario
        per_scene = {}
        for scene in sorted(set(r['scenario'] for r in test_data)):
            s_r = [r for r in test_data if r['scenario'] == scene]
            s_cos = np.mean([r['cos_dist'] for r in s_r])
            per_scene[scene] = float(s_cos)
            print(f"    {scene}: {s_cos:.4f}", flush=True)

        prompt_results[prompt_name] = {
            'auroc': auroc,
            'id_cos_mean': float(id_mean),
            'ood_cos_mean': float(ood_mean),
            'separation': float(ood_mean - id_mean),
            'per_scene': per_scene,
        }

    # Cross-prompt analysis
    print("\n" + "=" * 70, flush=True)
    print("CROSS-PROMPT ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    print(f"\n  {'Prompt':<15} {'AUROC':>8} {'ID cos':>10} {'OOD cos':>10} {'Sep':>8}",
          flush=True)
    print("  " + "-" * 55, flush=True)
    aurocs = []
    for name, result in prompt_results.items():
        print(f"  {name:<15} {result['auroc']:>8.3f} {result['id_cos_mean']:>10.4f} "
              f"{result['ood_cos_mean']:>10.4f} {result['separation']:>+8.4f}", flush=True)
        aurocs.append(result['auroc'])

    print(f"\n  Mean AUROC: {np.mean(aurocs):.3f} ± {np.std(aurocs):.3f}", flush=True)
    print(f"  Min AUROC: {min(aurocs):.3f}", flush=True)
    print(f"  Max AUROC: {max(aurocs):.3f}", flush=True)
    print(f"  Range: {max(aurocs) - min(aurocs):.3f}", flush=True)

    # Cross-prompt centroid similarity
    print("\n  Cross-prompt centroid analysis:", flush=True)
    print(f"  Testing whether calibration centroid from one prompt works for another...",
          flush=True)

    # Use original prompt's centroid for all prompts
    print("\n  Using 'original' calibration for all prompts:", flush=True)
    original_cal = []
    for fn in [create_highway, create_urban]:
        for i in range(10):
            h = extract_hidden(model, processor,
                               Image.fromarray(fn(i + 9000)), prompts['original'])
            original_cal.append(h)
    original_centroid = np.mean(original_cal, axis=0)

    cross_results = {}
    for prompt_name, prompt_text in prompts.items():
        if prompt_name == 'original':
            continue

        # Evaluate using original centroid but different inference prompt
        test_data = []
        for scene, (fn, is_ood, n) in test_fns.items():
            for i in range(n):
                h = extract_hidden(model, processor,
                                   Image.fromarray(fn(i + 200)), prompt_text)
                cos = cosine_dist(h, original_centroid)
                test_data.append({
                    'is_ood': is_ood,
                    'cos_dist': cos,
                })
                total_inferences += 1

        easy = [r for r in test_data if not r['is_ood']]
        ood = [r for r in test_data if r['is_ood']]
        labels = [0]*len(easy) + [1]*len(ood)
        scores = [r['cos_dist'] for r in easy + ood]
        auroc = roc_auc_score(labels, scores)
        cross_results[prompt_name] = auroc
        print(f"    {prompt_name}: AUROC = {auroc:.3f} "
              f"(vs self-calibrated: {prompt_results[prompt_name]['auroc']:.3f})", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'prompt_robustness',
        'experiment_number': 54,
        'timestamp': timestamp,
        'total_inferences': total_inferences,
        'prompts': {k: v for k, v in prompts.items()},
        'results': prompt_results,
        'cross_prompt': cross_results,
    }
    output_path = os.path.join(RESULTS_DIR, f"prompt_robustness_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)
    print(f"Total inferences: {total_inferences}", flush=True)


if __name__ == "__main__":
    main()
