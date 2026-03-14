"""
Diverse OOD Evaluation on Real OpenVLA-7B.

Tests action mass and entropy on a wider range of OOD inputs:
1. Random noise (existing)
2. Solid blank (existing)
3. Checkerboard patterns
4. Inverted highway (color negation)
5. Rotated highway (180°)
6. Text-only image (no driving content)
7. Indoor scene (wrong domain)
8. Extreme brightness (whiteout/blackout)

Experiment 26 in the CalibDrive series.
"""
import os
import json
import time
import datetime
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

RESULTS_DIR = "/workspace/Vizuara-VLA-Research/experiments"
os.makedirs(RESULTS_DIR, exist_ok=True)

N_SAMPLES = 10  # Per OOD type

SCENARIOS = {
    # In-distribution
    'highway': {'n': 20, 'speed': '30', 'difficulty': 'easy'},
    'urban': {'n': 20, 'speed': '15', 'difficulty': 'easy'},
    # Original OOD
    'ood_noise': {'n': N_SAMPLES, 'speed': '25', 'difficulty': 'ood'},
    'ood_blank': {'n': N_SAMPLES, 'speed': '25', 'difficulty': 'ood'},
    # New OOD types
    'ood_checker': {'n': N_SAMPLES, 'speed': '25', 'difficulty': 'ood'},
    'ood_inverted': {'n': N_SAMPLES, 'speed': '30', 'difficulty': 'ood'},
    'ood_rotated': {'n': N_SAMPLES, 'speed': '30', 'difficulty': 'ood'},
    'ood_text': {'n': N_SAMPLES, 'speed': '25', 'difficulty': 'ood'},
    'ood_indoor': {'n': N_SAMPLES, 'speed': '25', 'difficulty': 'ood'},
    'ood_whiteout': {'n': N_SAMPLES, 'speed': '25', 'difficulty': 'ood'},
    'ood_blackout': {'n': N_SAMPLES, 'speed': '25', 'difficulty': 'ood'},
}


def create_scene_image(scenario, idx, size=(256, 256)):
    np.random.seed(idx * 2100 + hash(scenario) % 21000)

    if scenario == 'highway':
        img = np.zeros((*size, 3), dtype=np.uint8)
        img[:size[0]//2] = [135, 206, 235]
        img[size[0]//2:] = [80, 80, 80]
    elif scenario == 'urban':
        img = np.zeros((*size, 3), dtype=np.uint8)
        img[:size[0]//3] = [135, 206, 235]
        img[size[0]//3:size[0]//2] = [139, 119, 101]
        img[size[0]//2:] = [80, 80, 80]
    elif scenario == 'ood_noise':
        img = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
    elif scenario == 'ood_blank':
        img = np.full((*size, 3), 128, dtype=np.uint8)
    elif scenario == 'ood_checker':
        # Checkerboard pattern
        img = np.zeros((*size, 3), dtype=np.uint8)
        block = 32
        for y in range(0, size[0], block):
            for x in range(0, size[1], block):
                if (y // block + x // block) % 2 == 0:
                    img[y:y+block, x:x+block] = [255, 255, 255]
    elif scenario == 'ood_inverted':
        # Inverted highway
        img = np.zeros((*size, 3), dtype=np.uint8)
        img[:size[0]//2] = [135, 206, 235]
        img[size[0]//2:] = [80, 80, 80]
        img = 255 - img  # Negate
    elif scenario == 'ood_rotated':
        # Rotated highway (180°)
        img = np.zeros((*size, 3), dtype=np.uint8)
        img[:size[0]//2] = [135, 206, 235]
        img[size[0]//2:] = [80, 80, 80]
        img = np.flip(img, axis=(0, 1)).copy()
    elif scenario == 'ood_text':
        # Text-only image
        img = np.full((*size, 3), 240, dtype=np.uint8)
        # Draw text using PIL
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        texts = ["STOP", "ERROR", "NO ROAD", "SYSTEM FAULT", "INVALID"]
        text = texts[idx % len(texts)]
        draw.text((size[1]//4, size[0]//3), text, fill=(0, 0, 0))
        draw.text((size[1]//4, size[0]//2), f"Sample {idx}", fill=(100, 100, 100))
        return pil_img
    elif scenario == 'ood_indoor':
        # Indoor-like scene (warm colors, no sky/road)
        img = np.zeros((*size, 3), dtype=np.uint8)
        img[:size[0]//3] = [210, 180, 140]  # Ceiling/tan
        img[size[0]//3:2*size[0]//3] = [180, 120, 80]  # Walls/brown
        img[2*size[0]//3:] = [100, 70, 50]  # Floor/dark
    elif scenario == 'ood_whiteout':
        img = np.full((*size, 3), 250, dtype=np.uint8)
    elif scenario == 'ood_blackout':
        img = np.full((*size, 3), 5, dtype=np.uint8)
    else:
        img = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)

    noise = np.random.randint(-3, 3, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def forward_full_vocab(model, processor, image, prompt):
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

    dim_masses = []
    dim_entropies = []
    dim_confs = []

    for score in outputs.scores[:7]:
        full_logits = score[0].float()
        full_probs = torch.softmax(full_logits, dim=0).cpu().numpy()
        action_probs = full_probs[action_start:]
        dim_masses.append(float(action_probs.sum()))
        action_norm = action_probs / (action_probs.sum() + 1e-10)
        dim_entropies.append(float(-(action_norm * np.log(action_norm + 1e-10)).sum()))
        dim_confs.append(float(action_norm.max()))

    return {
        'action_mass': float(np.mean(dim_masses)),
        'dim_masses': dim_masses,
        'entropy': float(np.mean(dim_entropies)),
        'conf': float(np.exp(np.mean(np.log(np.array(dim_confs) + 1e-10)))),
    }


def compute_auroc(pos_scores, neg_scores):
    n_correct = sum(1 for p in pos_scores for n in neg_scores if p > n)
    n_ties = sum(0.5 for p in pos_scores for n in neg_scores if p == n)
    n_total = len(pos_scores) * len(neg_scores)
    return (n_correct + n_ties) / n_total if n_total > 0 else 0.5


def main():
    print("=" * 70, flush=True)
    print("DIVERSE OOD EVALUATION ON REAL OpenVLA-7B", flush=True)
    print("=" * 70, flush=True)

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

    total = sum(s['n'] for s in SCENARIOS.values())
    print(f"Total samples: {total}", flush=True)
    print(flush=True)

    prompt = "In: What action should the robot take to drive forward at {speed} m/s safely?\nOut:"
    all_samples = []
    sample_idx = 0

    for scenario, config in SCENARIOS.items():
        for i in range(config['n']):
            sample_idx += 1
            image = create_scene_image(scenario, i)
            p = prompt.format(speed=config['speed'])

            r = forward_full_vocab(model, processor, image, p)

            all_samples.append({
                'scenario': scenario,
                'difficulty': config['difficulty'],
                'idx': i,
                'action_mass': r['action_mass'],
                'entropy': r['entropy'],
                'conf': r['conf'],
            })

            if i % 5 == 0 or i == config['n'] - 1:
                print(f"  [{sample_idx}/{total}] {scenario}_{i}: "
                      f"mass={r['action_mass']:.4f}, ent={r['entropy']:.3f}",
                      flush=True)

    # ===================================================================
    # Analysis
    # ===================================================================
    print("\n" + "=" * 70, flush=True)
    print("DIVERSE OOD ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    print("\n1. Per-Scenario Statistics", flush=True)
    print("-" * 80, flush=True)
    print(f"{'Scenario':<15} {'N':>4} {'Mass':>8} {'±':>6} {'Entropy':>8} {'±':>6} {'Conf':>7}", flush=True)
    print("-" * 80, flush=True)

    for scenario in SCENARIOS:
        samples = [s for s in all_samples if s['scenario'] == scenario]
        masses = [s['action_mass'] for s in samples]
        ents = [s['entropy'] for s in samples]
        confs = [s['conf'] for s in samples]
        print(f"{scenario:<15} {len(samples):>4} {np.mean(masses):>8.4f} {np.std(masses):>6.4f} "
              f"{np.mean(ents):>8.3f} {np.std(ents):>6.3f} {np.mean(confs):>7.3f}", flush=True)

    # 2. AUROC for each OOD type vs easy
    print("\n2. Per-OOD-Type AUROC (vs Easy)", flush=True)
    print("-" * 60, flush=True)

    easy_mass = [-s['action_mass'] for s in all_samples if s['difficulty'] == 'easy']
    easy_ent = [s['entropy'] for s in all_samples if s['difficulty'] == 'easy']

    ood_types = [s for s in SCENARIOS.keys() if s.startswith('ood_')]
    for ood_type in ood_types:
        ood_s = [s for s in all_samples if s['scenario'] == ood_type]
        ood_mass = [-s['action_mass'] for s in ood_s]
        ood_ent = [s['entropy'] for s in ood_s]

        auroc_mass = compute_auroc(ood_mass, easy_mass)
        auroc_ent = compute_auroc(ood_ent, easy_ent)

        print(f"  {ood_type:<15}: Mass AUROC={auroc_mass:.3f}, Ent AUROC={auroc_ent:.3f}, "
              f"mass={np.mean([s['action_mass'] for s in ood_s]):.4f}", flush=True)

    # 3. Overall AUROC (all OOD types combined)
    print("\n3. Overall AUROC (All OOD Types Combined)", flush=True)
    print("-" * 40, flush=True)

    all_ood_mass = [-s['action_mass'] for s in all_samples if s['difficulty'] == 'ood']
    all_ood_ent = [s['entropy'] for s in all_samples if s['difficulty'] == 'ood']

    auroc_mass_all = compute_auroc(all_ood_mass, easy_mass)
    auroc_ent_all = compute_auroc(all_ood_ent, easy_ent)
    print(f"  Action Mass AUROC: {auroc_mass_all:.3f}", flush=True)
    print(f"  Entropy AUROC: {auroc_ent_all:.3f}", flush=True)

    # 4. Ranking of OOD types by difficulty (for action mass)
    print("\n4. OOD Types Ranked by Action Mass (hardest to detect first)", flush=True)
    print("-" * 50, flush=True)

    ood_rankings = []
    for ood_type in ood_types:
        ood_s = [s for s in all_samples if s['scenario'] == ood_type]
        avg_mass = np.mean([s['action_mass'] for s in ood_s])
        ood_rankings.append((ood_type, avg_mass))

    ood_rankings.sort(key=lambda x: -x[1])  # Highest mass = hardest to detect
    for rank, (name, mass) in enumerate(ood_rankings):
        print(f"  {rank+1}. {name:<15}: mass={mass:.4f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'timestamp': timestamp,
        'samples': all_samples,
    }

    output_path = os.path.join(RESULTS_DIR, f"diverse_ood_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
