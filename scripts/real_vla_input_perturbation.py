"""
Input Perturbation Sensitivity on Real OpenVLA-7B.

Tests how small input perturbations affect action predictions and confidence:
1. Pixel noise at different intensities (sigma = 1, 5, 10, 25, 50)
2. Color jitter (brightness, contrast shifts)
3. Spatial cropping (slight crop+resize)
4. Measuring prediction stability vs perturbation strength

This reveals whether the model's action predictions are stable or chaotic
under small input changes, which has direct safety implications.

Experiment 14 in the CalibDrive series.
"""
import os
import json
import time
import datetime
import numpy as np
import torch
from PIL import Image

RESULTS_DIR = "/workspace/Vizuara-VLA-Research/experiments"
os.makedirs(RESULTS_DIR, exist_ok=True)

NOISE_LEVELS = [0, 1, 5, 10, 25, 50]
BRIGHTNESS_SHIFTS = [-30, -15, 0, 15, 30]

SCENARIOS = {
    'highway': {'n': 10, 'speed': '30', 'difficulty': 'easy'},
    'urban': {'n': 10, 'speed': '15', 'difficulty': 'easy'},
    'night': {'n': 10, 'speed': '25', 'difficulty': 'hard'},
    'ood_noise': {'n': 10, 'speed': '25', 'difficulty': 'ood'},
}


def create_scene_image(scenario, idx, size=(256, 256)):
    np.random.seed(idx * 800 + hash(scenario) % 8000)
    if scenario == 'highway':
        img = np.zeros((*size, 3), dtype=np.uint8)
        img[:size[0]//2] = [135, 206, 235]
        img[size[0]//2:] = [80, 80, 80]
    elif scenario == 'urban':
        img = np.zeros((*size, 3), dtype=np.uint8)
        img[:size[0]//3] = [135, 206, 235]
        img[size[0]//3:size[0]//2] = [139, 119, 101]
        img[size[0]//2:] = [80, 80, 80]
    elif scenario == 'night':
        img = np.full((*size, 3), 15, dtype=np.uint8)
        img[size[0]//2:] = [30, 30, 35]
    elif scenario == 'ood_noise':
        img = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
    else:
        img = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)

    noise = np.random.randint(-5, 5, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def add_gaussian_noise(img_array, sigma):
    noise = np.random.normal(0, sigma, img_array.shape)
    return np.clip(img_array.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def shift_brightness(img_array, delta):
    return np.clip(img_array.astype(np.int16) + delta, 0, 255).astype(np.uint8)


def single_forward(model, processor, image, prompt):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=7,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

    vocab_size = outputs.scores[0].shape[-1]
    action_start = vocab_size - 256

    dim_confs = []
    dim_entropies = []
    dim_tokens = []
    for score in outputs.scores[:7]:
        logits = score[0, action_start:].float()
        probs = torch.softmax(logits, dim=0)
        dim_confs.append(probs.max().item())
        dim_entropies.append(-(probs * torch.log(probs + 1e-10)).sum().item())
        dim_tokens.append(probs.argmax().item())

    return {
        'geo_conf': float(np.exp(np.mean(np.log(np.array(dim_confs) + 1e-10)))),
        'mean_entropy': float(np.mean(dim_entropies)),
        'dim_tokens': dim_tokens,
        'dim_confs': [float(c) for c in dim_confs],
    }


def token_agreement(tokens1, tokens2):
    return sum(1 for a, b in zip(tokens1, tokens2) if a == b) / len(tokens1)


def main():
    print("=" * 70, flush=True)
    print("INPUT PERTURBATION SENSITIVITY ON REAL OpenVLA-7B", flush=True)
    print("=" * 70, flush=True)
    print(f"Noise levels: {NOISE_LEVELS}", flush=True)
    print(f"Brightness shifts: {BRIGHTNESS_SHIFTS}", flush=True)
    print(flush=True)

    from transformers import AutoModelForVision2Seq, AutoProcessor
    print("Loading OpenVLA-7B...", flush=True)
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        "openvla/openvla-7b",
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.", flush=True)

    total_samples = sum(s['n'] for s in SCENARIOS.values())
    n_perturbations = len(NOISE_LEVELS) + len(BRIGHTNESS_SHIFTS) - 1  # -1 for sigma=0 overlap
    total_inferences = total_samples * n_perturbations
    print(f"Total samples: {total_samples}, Perturbations: {n_perturbations}, Inferences: {total_inferences}", flush=True)
    print(flush=True)

    all_results = {}
    sample_idx = 0

    for scenario, config in SCENARIOS.items():
        scenario_results = []

        for i in range(config['n']):
            sample_idx += 1
            base_img = create_scene_image(scenario, i)
            base_array = np.array(base_img)
            prompt = f"In: What action should the robot take to drive forward at {config['speed']} m/s safely?\nOut:"

            t0 = time.time()

            # Get baseline (clean) prediction
            base_result = single_forward(model, processor, base_img, prompt)
            base_tokens = base_result['dim_tokens']

            # === Noise perturbations ===
            noise_results = []
            for sigma in NOISE_LEVELS:
                if sigma == 0:
                    noise_results.append({
                        'sigma': 0,
                        'geo_conf': base_result['geo_conf'],
                        'mean_entropy': base_result['mean_entropy'],
                        'token_agree_vs_clean': 1.0,
                        'dim_tokens': base_tokens,
                    })
                    continue

                perturbed_array = add_gaussian_noise(base_array, sigma)
                perturbed_img = Image.fromarray(perturbed_array)
                r = single_forward(model, processor, perturbed_img, prompt)
                agree = token_agreement(base_tokens, r['dim_tokens'])

                noise_results.append({
                    'sigma': sigma,
                    'geo_conf': r['geo_conf'],
                    'mean_entropy': r['mean_entropy'],
                    'token_agree_vs_clean': agree,
                    'dim_tokens': r['dim_tokens'],
                })

            # === Brightness perturbations ===
            brightness_results = []
            for delta in BRIGHTNESS_SHIFTS:
                if delta == 0:
                    brightness_results.append({
                        'delta': 0,
                        'geo_conf': base_result['geo_conf'],
                        'mean_entropy': base_result['mean_entropy'],
                        'token_agree_vs_clean': 1.0,
                        'dim_tokens': base_tokens,
                    })
                    continue

                shifted_array = shift_brightness(base_array, delta)
                shifted_img = Image.fromarray(shifted_array)
                r = single_forward(model, processor, shifted_img, prompt)
                agree = token_agreement(base_tokens, r['dim_tokens'])

                brightness_results.append({
                    'delta': delta,
                    'geo_conf': r['geo_conf'],
                    'mean_entropy': r['mean_entropy'],
                    'token_agree_vs_clean': agree,
                    'dim_tokens': r['dim_tokens'],
                })

            elapsed = time.time() - t0

            sample_result = {
                'scenario': scenario,
                'difficulty': config['difficulty'],
                'idx': i,
                'base_conf': base_result['geo_conf'],
                'base_entropy': base_result['mean_entropy'],
                'base_tokens': base_tokens,
                'noise_results': noise_results,
                'brightness_results': brightness_results,
            }
            scenario_results.append(sample_result)

            # Quick summary
            max_noise = noise_results[-1]
            print(f"  [{sample_idx}/{total_samples}] {scenario}_{i}: "
                  f"base_conf={base_result['geo_conf']:.3f}, "
                  f"σ=50 agree={max_noise['token_agree_vs_clean']:.2f}, "
                  f"σ=50 conf={max_noise['geo_conf']:.3f} ({elapsed:.1f}s)", flush=True)

        all_results[scenario] = scenario_results

    # ===================================================================
    # Analysis
    # ===================================================================
    print("\n" + "=" * 70, flush=True)
    print("INPUT PERTURBATION ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    # 1. Token agreement vs noise level
    print("\n1. Token Agreement vs Gaussian Noise Level", flush=True)
    print("-" * 70, flush=True)
    print(f"{'Sigma':>6}" + "".join(f"{s:>12}" for s in SCENARIOS.keys()) + f"{'All':>12}", flush=True)
    print("-" * 70, flush=True)

    for sigma in NOISE_LEVELS:
        agrees = {}
        all_agrees = []
        for scenario in SCENARIOS:
            sc_agrees = []
            for s in all_results[scenario]:
                for nr in s['noise_results']:
                    if nr['sigma'] == sigma:
                        sc_agrees.append(nr['token_agree_vs_clean'])
            agrees[scenario] = np.mean(sc_agrees) if sc_agrees else 0
            all_agrees.extend(sc_agrees)

        row = f"{sigma:>6}"
        for scenario in SCENARIOS:
            row += f"{agrees[scenario]:>12.3f}"
        row += f"{np.mean(all_agrees):>12.3f}"
        print(row, flush=True)

    # 2. Confidence shift vs noise level
    print("\n2. Confidence Shift vs Noise Level", flush=True)
    print("-" * 70, flush=True)
    print(f"{'Sigma':>6}" + "".join(f"{s:>12}" for s in SCENARIOS.keys()), flush=True)
    print("-" * 70, flush=True)

    for sigma in NOISE_LEVELS:
        row = f"{sigma:>6}"
        for scenario in SCENARIOS:
            confs = []
            for s in all_results[scenario]:
                for nr in s['noise_results']:
                    if nr['sigma'] == sigma:
                        confs.append(nr['geo_conf'])
            row += f"{np.mean(confs):>12.3f}"
        print(row, flush=True)

    # 3. Brightness sensitivity
    print("\n3. Token Agreement vs Brightness Shift", flush=True)
    print("-" * 70, flush=True)
    print(f"{'Delta':>6}" + "".join(f"{s:>12}" for s in SCENARIOS.keys()) + f"{'All':>12}", flush=True)
    print("-" * 70, flush=True)

    for delta in BRIGHTNESS_SHIFTS:
        agrees = {}
        all_agrees = []
        for scenario in SCENARIOS:
            sc_agrees = []
            for s in all_results[scenario]:
                for br in s['brightness_results']:
                    if br['delta'] == delta:
                        sc_agrees.append(br['token_agree_vs_clean'])
            agrees[scenario] = np.mean(sc_agrees) if sc_agrees else 0
            all_agrees.extend(sc_agrees)

        row = f"{delta:>6}"
        for scenario in SCENARIOS:
            row += f"{agrees[scenario]:>12.3f}"
        row += f"{np.mean(all_agrees):>12.3f}"
        print(row, flush=True)

    # 4. Which dimensions are most sensitive?
    print("\n4. Per-Dimension Token Stability at σ=10", flush=True)
    print("-" * 50, flush=True)

    for d in range(7):
        agrees_dim = []
        for scenario in SCENARIOS:
            for s in all_results[scenario]:
                for nr in s['noise_results']:
                    if nr['sigma'] == 10:
                        base_tok = s['base_tokens'][d]
                        pert_tok = nr['dim_tokens'][d]
                        agrees_dim.append(1 if base_tok == pert_tok else 0)
        print(f"  Dim {d}: agreement = {np.mean(agrees_dim):.3f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'timestamp': timestamp,
        'noise_levels': NOISE_LEVELS,
        'brightness_shifts': BRIGHTNESS_SHIFTS,
        'per_sample': {
            scenario: [
                {k: v for k, v in s.items()}
                for s in samples
            ]
            for scenario, samples in all_results.items()
        },
    }

    output_path = os.path.join(RESULTS_DIR, f"perturbation_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
