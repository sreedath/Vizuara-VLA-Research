"""
OOD Detection Baselines Comparison on Real OpenVLA-7B.

Compares our cosine distance approach against standard OOD baselines:
1. Max Softmax Probability (MSP) — Hendrycks & Gimpel 2017
2. Energy Score — Liu et al. 2020: -log(sum(exp(logits)))
3. Max Logit — Hendrycks et al. 2022
4. Entropy — standard
5. Action Mass — our discovery
6. Cosine Distance — our primary method
7. Per-scene Cosine — our improved method
8. Cosine + Mass combo — our optimal method

Tests on both simple and realistic images.

Experiment 47 in the CalibDrive series.
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


# === Image generators ===
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

def create_blank(idx):
    rng = np.random.default_rng(idx * 5005)
    val = rng.integers(200, 256)
    return np.full((*SIZE, 3), val, dtype=np.uint8)

def create_indoor(idx):
    rng = np.random.default_rng(idx * 5004)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:] = [200, 180, 160]
    img[SIZE[0]//2:] = [139, 90, 43]
    for x in range(0, SIZE[1], 30):
        img[10:SIZE[0]//2-10, x:x+25] = [100, 150, 200]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_inverted(idx):
    base = create_highway(idx + 3000)
    return 255 - base

def create_blackout(idx):
    return np.zeros((*SIZE, 3), dtype=np.uint8)

# Realistic
def create_r_highway(idx):
    rng = np.random.default_rng(idx * 4101)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    for y in range(SIZE[0]//2):
        frac = y / (SIZE[0]//2)
        img[y, :] = [int(80+55*frac), int(150+56*frac), int(255-20*frac)]
    for y in range(SIZE[0]//2, SIZE[0]):
        base = 60 + rng.integers(-5, 6)
        img[y, :] = [base, base, base]
    noise = rng.integers(-3, 4, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_r_urban(idx):
    rng = np.random.default_rng(idx * 4102)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//4] = [135, 206, 235]
    for x_start in range(0, SIZE[1], 40):
        h = rng.integers(SIZE[0]//4, SIZE[0]//2)
        color = rng.integers(100, 200, 3)
        img[SIZE[0]//4:h, x_start:x_start+38] = color
    img[SIZE[0]//2:SIZE[0]//2+20] = [180, 170, 160]
    img[SIZE[0]//2+20:] = [70, 70, 70]
    noise = rng.integers(-3, 4, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_offroad(idx):
    rng = np.random.default_rng(idx * 4107)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [135, 206, 235]
    for x in range(0, SIZE[1], 15):
        h = SIZE[0]//3 + rng.integers(0, 30)
        img[SIZE[0]//3:h, x:x+12] = [30+rng.integers(0,30), 100+rng.integers(0,50), 20+rng.integers(0,30)]
    img[SIZE[0]//2:] = [140, 100, 60]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_snow(idx):
    rng = np.random.default_rng(idx * 4105)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [200, 210, 220]
    for y in range(SIZE[0]//2, SIZE[0]):
        b = 200 + rng.integers(-10, 10)
        img[y, :] = [b, b, b]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def extract_all_signals(model, processor, image, prompt):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=7, do_sample=False,
            output_scores=True, output_hidden_states=True,
            return_dict_in_generate=True,
        )
    # Hidden state
    if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
        last_step = outputs.hidden_states[-1]
        if isinstance(last_step, tuple):
            hidden = last_step[-1][0, -1, :].float().cpu().numpy()
        else:
            hidden = last_step[0, -1, :].float().cpu().numpy()
    else:
        hidden = np.zeros(4096)

    vocab_size = outputs.scores[0].shape[-1]
    action_start = vocab_size - 256

    # Compute all OOD detection signals per dimension
    msp_vals = []  # max softmax prob
    energy_vals = []  # energy = -logsumexp
    max_logit_vals = []
    entropy_vals = []
    mass_vals = []

    for score in outputs.scores[:7]:
        logits = score[0].float()

        # MSP: max probability over full vocabulary
        probs = torch.softmax(logits, dim=0)
        msp_vals.append(float(probs.max()))

        # Energy: -log(sum(exp(logits))) — typically negative, more negative = more OOD
        # Following Liu et al. 2020: E(x) = -T * log(sum(exp(f_i/T)))
        # Lower energy = in-distribution, higher energy = OOD
        T = 1.0
        energy = -T * torch.logsumexp(logits / T, dim=0).item()
        energy_vals.append(energy)

        # Max logit: max raw logit value
        max_logit_vals.append(float(logits.max()))

        # Entropy over full vocab
        ent = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        entropy_vals.append(ent)

        # Action mass
        mass_vals.append(float(probs[action_start:].sum()))

    return {
        'hidden': hidden,
        'msp': float(np.mean(msp_vals)),
        'energy': float(np.mean(energy_vals)),
        'max_logit': float(np.mean(max_logit_vals)),
        'entropy': float(np.mean(entropy_vals)),
        'action_mass': float(np.mean(mass_vals)),
    }


def cosine_dist(a, b):
    return 1.0 - float(np.dot(a / (np.linalg.norm(a) + 1e-10),
                               b / (np.linalg.norm(b) + 1e-10)))


def main():
    print("=" * 70, flush=True)
    print("OOD DETECTION BASELINES COMPARISON", flush=True)
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

    # ==============================
    # PART A: Simple images
    # ==============================
    print("\n=== PART A: Simple Images ===", flush=True)

    # Calibration
    cal_hidden = []
    for fn in [create_highway, create_urban]:
        for i in range(10):
            data = extract_all_signals(model, processor,
                                       Image.fromarray(fn(i + 800)), prompt)
            cal_hidden.append(data['hidden'])
    centroid = np.mean(cal_hidden, axis=0)

    # Test
    simple_tests = {
        'highway': (create_highway, False, 10),
        'urban': (create_urban, False, 10),
        'noise': (create_noise, True, 8),
        'blank': (create_blank, True, 8),
        'indoor': (create_indoor, True, 8),
        'inverted': (create_inverted, True, 8),
        'blackout': (create_blackout, True, 8),
    }

    simple_results = []
    total = sum(v[2] for v in simple_tests.values())
    cnt = 0
    for scene, (fn, is_ood, n) in simple_tests.items():
        for i in range(n):
            cnt += 1
            data = extract_all_signals(model, processor,
                                       Image.fromarray(fn(i + 100)), prompt)
            data['cos_dist'] = cosine_dist(data['hidden'], centroid)
            data['scenario'] = scene
            data['is_ood'] = is_ood
            del data['hidden']
            simple_results.append(data)
            if cnt % 10 == 0:
                print(f"  [{cnt}/{total}] {scene}_{i}", flush=True)

    # ==============================
    # PART B: Realistic images
    # ==============================
    print("\n=== PART B: Realistic Images ===", flush=True)

    cal_r_hidden = {}
    cal_r_all = []
    for name, fn in [('highway', create_r_highway), ('urban', create_r_urban)]:
        cal_r_hidden[name] = []
        for i in range(8):
            data = extract_all_signals(model, processor,
                                       Image.fromarray(fn(i + 900)), prompt)
            cal_r_hidden[name].append(data['hidden'])
            cal_r_all.append(data['hidden'])
    r_centroid = np.mean(cal_r_all, axis=0)
    r_scene_centroids = {k: np.mean(v, axis=0) for k, v in cal_r_hidden.items()}

    realistic_tests = {
        'highway_r': (create_r_highway, False, 10),
        'urban_r': (create_r_urban, False, 10),
        'offroad': (create_offroad, True, 8),
        'snow': (create_snow, True, 8),
    }

    realistic_results = []
    total_r = sum(v[2] for v in realistic_tests.values())
    cnt = 0
    for scene, (fn, is_ood, n) in realistic_tests.items():
        for i in range(n):
            cnt += 1
            data = extract_all_signals(model, processor,
                                       Image.fromarray(fn(i + 100)), prompt)
            data['cos_dist'] = cosine_dist(data['hidden'], r_centroid)
            data['cos_per_scene'] = min(cosine_dist(data['hidden'], c)
                                        for c in r_scene_centroids.values())
            data['scenario'] = scene
            data['is_ood'] = is_ood
            del data['hidden']
            realistic_results.append(data)
            if cnt % 10 == 0:
                print(f"  [{cnt}/{total_r}] {scene}_{i}", flush=True)

    # ===================================================================
    # Analysis
    # ===================================================================
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    for part_name, results in [("Part A: Simple Images", simple_results),
                                ("Part B: Realistic Images", realistic_results)]:
        print(f"\n{'='*50}", flush=True)
        print(f"  {part_name}", flush=True)
        print(f"{'='*50}", flush=True)

        easy = [r for r in results if not r['is_ood']]
        ood = [r for r in results if r['is_ood']]
        labels = [0]*len(easy) + [1]*len(ood)
        all_r = easy + ood

        # Compute AUROCs
        # For MSP and max_logit: LOWER = more OOD (use negative)
        # For energy: HIGHER = more OOD (as-is)
        # For entropy: HIGHER = more OOD (as-is)
        # For action mass: LOWER mass = more OOD (use 1-mass)
        # For cosine: HIGHER = more OOD (as-is)

        signals = {
            'MSP (1-max prob)': [1 - r['msp'] for r in all_r],
            'Energy score': [r['energy'] for r in all_r],
            'Max logit (neg)': [-r['max_logit'] for r in all_r],
            'Entropy': [r['entropy'] for r in all_r],
            'Action mass (1-mass)': [1 - r['action_mass'] for r in all_r],
            'Cosine distance': [r['cos_dist'] for r in all_r],
        }

        if 'cos_per_scene' in results[0]:
            signals['Per-scene cosine'] = [r['cos_per_scene'] for r in all_r]

            # Optimal combo
            cos_vals = np.array([r['cos_per_scene'] for r in all_r])
            mass_vals = np.array([1 - r['action_mass'] for r in all_r])
            cos_n = (cos_vals - cos_vals.min()) / (cos_vals.max() - cos_vals.min() + 1e-10)
            mass_n = (mass_vals - mass_vals.min()) / (mass_vals.max() - mass_vals.min() + 1e-10)
            signals['Optimal (0.7cos+0.3mass)'] = list(0.7 * cos_n + 0.3 * mass_n)

        print(f"\n  {'Method':<30} | {'AUROC':>8} | {'Reference':>15}", flush=True)
        print("  " + "-"*60, flush=True)
        refs = {
            'MSP (1-max prob)': 'Hendrycks+ 2017',
            'Energy score': 'Liu+ 2020',
            'Max logit (neg)': 'Hendrycks+ 2022',
            'Entropy': 'Standard',
            'Action mass (1-mass)': 'Ours',
            'Cosine distance': 'Ours',
            'Per-scene cosine': 'Ours',
            'Optimal (0.7cos+0.3mass)': 'Ours',
        }

        for name, scores in sorted(signals.items(), key=lambda x: -roc_auc_score(labels, x[1])):
            auroc = roc_auc_score(labels, scores)
            ref = refs.get(name, '')
            print(f"  {name:<30} | {auroc:>8.3f} | {ref:>15}", flush=True)

        # Per-scenario statistics for top methods
        print(f"\n  Per-Scenario Mean Values:", flush=True)
        print(f"  {'Scenario':<15} {'OOD':>4} {'MSP':>8} {'Energy':>8} "
              f"{'Entropy':>8} {'Mass':>8} {'Cosine':>8}", flush=True)
        print("  " + "-"*65, flush=True)
        for s in sorted(set(r['scenario'] for r in results)):
            s_r = [r for r in results if r['scenario'] == s]
            is_ood = s_r[0]['is_ood']
            msp = np.mean([r['msp'] for r in s_r])
            energy = np.mean([r['energy'] for r in s_r])
            entropy = np.mean([r['entropy'] for r in s_r])
            mass = np.mean([r['action_mass'] for r in s_r])
            cos = np.mean([r['cos_dist'] for r in s_r])
            label = 'Yes' if is_ood else 'No'
            print(f"  {s:<15} {label:>4} {msp:>8.4f} {energy:>8.2f} "
                  f"{entropy:>8.3f} {mass:>8.4f} {cos:>8.4f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'ood_baselines',
        'experiment_number': 47,
        'timestamp': timestamp,
        'simple_results': simple_results,
        'realistic_results': realistic_results,
    }
    output_path = os.path.join(RESULTS_DIR, f"ood_baselines_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
