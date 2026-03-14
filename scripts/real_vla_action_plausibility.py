"""
Action Plausibility as OOD Signal on Real OpenVLA-7B.

Hypothesis: OOD inputs produce implausible action sequences that violate
physical constraints (e.g., contradictory steering + acceleration, extreme
values, high per-dimension entropy). This tests whether action-level signals
can complement hidden state cosine distance for realistic image detection.

Tests:
1. Action value spread (std across 7 dims)
2. Action entropy (per-dim mean)
3. Max action confidence (per-dim mean)
4. Action smoothness (change between consecutive dims)
5. Combined action plausibility score
6. Cosine distance (baseline)
7. Per-scene cosine (baseline)

Uses both simple and realistic images to compare.

Experiment 43 in the CalibDrive series.
"""
import os
import json
import time
import datetime
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

RESULTS_DIR = "/workspace/Vizuara-VLA-Research/experiments"
os.makedirs(RESULTS_DIR, exist_ok=True)

SIZE = (256, 256)


# === Simple image generators (from prior experiments) ===
def create_simple_highway(idx):
    rng = np.random.default_rng(idx * 5001)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]
    img[SIZE[0]//2:] = [80, 80, 80]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [255, 255, 255]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_simple_urban(idx):
    rng = np.random.default_rng(idx * 5002)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [135, 206, 235]
    img[SIZE[0]//3:SIZE[0]//2] = [139, 119, 101]
    img[SIZE[0]//2:] = [60, 60, 60]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_simple_noise(idx):
    rng = np.random.default_rng(idx * 5003)
    return rng.integers(0, 256, (*SIZE, 3), dtype=np.uint8)

def create_simple_indoor(idx):
    rng = np.random.default_rng(idx * 5004)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:] = [200, 180, 160]
    img[SIZE[0]//2:] = [139, 90, 43]
    for x in range(0, SIZE[1], 30):
        img[10:SIZE[0]//2-10, x:x+25] = [100, 150, 200]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


# === Realistic image generators ===
def create_realistic_highway(idx):
    rng = np.random.default_rng(idx * 4101)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    for y in range(SIZE[0]//2):
        frac = y / (SIZE[0]//2)
        img[y, :] = [int(80+55*frac), int(150+56*frac), int(255-20*frac)]
    for y in range(SIZE[0]//2, SIZE[0]):
        base = 60 + rng.integers(-5, 6)
        img[y, :] = [base, base, base]
    for y in range(SIZE[0]//2+10, SIZE[0], 20):
        if (y//20)%2 == 0:
            img[y:y+8, SIZE[1]//2-2:SIZE[1]//2+2] = [220, 220, 220]
    noise = rng.integers(-3, 4, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_realistic_urban(idx):
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

def create_snow_road(idx):
    rng = np.random.default_rng(idx * 4105)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [200, 210, 220]
    for y in range(SIZE[0]//2, SIZE[0]):
        b = 200 + rng.integers(-10, 10)
        img[y, :] = [b, b, b]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
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

def create_tunnel(idx):
    rng = np.random.default_rng(idx * 4108)
    img = np.full((*SIZE, 3), 10, dtype=np.uint8)
    cy, cx = SIZE[0]//2 - 20, SIZE[1]//2
    for y in range(SIZE[0]):
        for x in range(SIZE[1]):
            dist = np.sqrt((y - cy)**2 + (x - cx)**2)
            if dist < 40:
                b = int(200 * (1 - dist/40))
                img[y, x] = [b, b, b]
    img[SIZE[0]//2:] = np.clip(img[SIZE[0]//2:].astype(int) + 20, 0, 255).astype(np.uint8)
    noise = rng.integers(-2, 3, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def extract_full(model, processor, image, prompt):
    """Extract hidden state, full score info, and decoded actions."""
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

    dim_masses = []
    dim_entropies = []
    dim_max_confs = []
    dim_action_values = []  # argmax bin index per dim
    dim_action_probs = []   # full 256-bin distribution per dim

    for score in outputs.scores[:7]:
        full_logits = score[0].float()
        full_probs = torch.softmax(full_logits, dim=0)

        # Action mass
        action_probs = full_probs[action_start:]
        dim_masses.append(float(action_probs.sum()))

        # Per-dim entropy over action bins
        ap = action_probs.cpu().numpy()
        ap = ap / (ap.sum() + 1e-10)
        ent = -np.sum(ap * np.log(ap + 1e-10))
        dim_entropies.append(float(ent))

        # Max confidence
        dim_max_confs.append(float(action_probs.max()))

        # Argmax action bin
        dim_action_values.append(int(action_probs.argmax()))
        dim_action_probs.append(ap)

    # Action plausibility metrics
    action_vals = np.array(dim_action_values, dtype=float)
    action_spread = float(np.std(action_vals))
    action_range = float(np.max(action_vals) - np.min(action_vals))

    # Smoothness: how much action changes between consecutive dims
    diffs = np.diff(action_vals)
    action_roughness = float(np.mean(np.abs(diffs)))

    # Center deviation: how far from center bin (128)
    center_dev = float(np.mean(np.abs(action_vals - 128)))

    # Entropy statistics
    mean_entropy = float(np.mean(dim_entropies))
    max_entropy = float(np.max(dim_entropies))
    entropy_std = float(np.std(dim_entropies))

    # Mean max confidence
    mean_max_conf = float(np.mean(dim_max_confs))

    return {
        'hidden': hidden,
        'action_mass': float(np.mean(dim_masses)),
        'action_vals': dim_action_values,
        'action_spread': action_spread,
        'action_range': action_range,
        'action_roughness': action_roughness,
        'center_dev': center_dev,
        'mean_entropy': mean_entropy,
        'max_entropy': max_entropy,
        'entropy_std': entropy_std,
        'mean_max_conf': mean_max_conf,
    }


def cosine_dist(a, b):
    a_n = a / (np.linalg.norm(a) + 1e-10)
    b_n = b / (np.linalg.norm(b) + 1e-10)
    return 1.0 - float(np.dot(a_n, b_n))


def main():
    print("=" * 70, flush=True)
    print("ACTION PLAUSIBILITY AS OOD SIGNAL", flush=True)
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

    prompt = "In: What action should the robot take to drive forward at 25 m/s safely?\nOut:"

    # === Part A: Simple images (where cosine works great) ===
    print("\n=== PART A: Simple Images ===", flush=True)
    cal_simple = []
    for fn in [create_simple_highway, create_simple_urban]:
        for i in range(10):
            img = Image.fromarray(fn(i + 800))
            data = extract_full(model, processor, img, prompt)
            cal_simple.append(data['hidden'])

    simple_centroid = np.mean(cal_simple, axis=0)

    simple_results = []
    simple_fns = {
        'highway': (create_simple_highway, False, 8),
        'urban': (create_simple_urban, False, 8),
        'noise': (create_simple_noise, True, 8),
        'indoor': (create_simple_indoor, True, 8),
    }
    total_a = sum(v[2] for v in simple_fns.values())
    cnt = 0
    for scene, (fn, is_ood, n) in simple_fns.items():
        for i in range(n):
            cnt += 1
            img = Image.fromarray(fn(i + 100))
            data = extract_full(model, processor, img, prompt)
            data['cos_dist'] = cosine_dist(data['hidden'], simple_centroid)
            data['scenario'] = scene
            data['is_ood'] = is_ood
            del data['hidden']
            simple_results.append(data)
            if cnt % 8 == 0:
                print(f"  [{cnt}/{total_a}] {scene}_{i}: cos={data['cos_dist']:.4f}, "
                      f"spread={data['action_spread']:.1f}, rough={data['action_roughness']:.1f}",
                      flush=True)

    # === Part B: Realistic images ===
    print("\n=== PART B: Realistic Images ===", flush=True)
    cal_realistic = {}
    cal_all = []
    for name, fn in [('highway', create_realistic_highway), ('urban', create_realistic_urban)]:
        cal_realistic[name] = []
        for i in range(8):
            img = Image.fromarray(fn(i + 900))
            data = extract_full(model, processor, img, prompt)
            cal_realistic[name].append(data['hidden'])
            cal_all.append(data['hidden'])
    realistic_centroid = np.mean(cal_all, axis=0)
    scene_centroids = {k: np.mean(v, axis=0) for k, v in cal_realistic.items()}

    realistic_results = []
    realistic_fns = {
        'highway_r': (create_realistic_highway, False, 8),
        'urban_r': (create_realistic_urban, False, 8),
        'snow': (create_snow_road, True, 8),
        'offroad': (create_offroad, True, 8),
        'tunnel': (create_tunnel, True, 8),
    }
    total_b = sum(v[2] for v in realistic_fns.values())
    cnt = 0
    for scene, (fn, is_ood, n) in realistic_fns.items():
        for i in range(n):
            cnt += 1
            img = Image.fromarray(fn(i + 100))
            data = extract_full(model, processor, img, prompt)
            data['cos_global'] = cosine_dist(data['hidden'], realistic_centroid)
            data['cos_per_scene'] = min(cosine_dist(data['hidden'], c) for c in scene_centroids.values())
            data['scenario'] = scene
            data['is_ood'] = is_ood
            del data['hidden']
            realistic_results.append(data)
            if cnt % 8 == 0:
                print(f"  [{cnt}/{total_b}] {scene}_{i}: cos={data['cos_global']:.4f}, "
                      f"spread={data['action_spread']:.1f}, rough={data['action_roughness']:.1f}",
                      flush=True)

    # ===================================================================
    # Analysis
    # ===================================================================
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    for part_name, results, cos_key in [
        ("Part A: Simple Images", simple_results, 'cos_dist'),
        ("Part B: Realistic Images", realistic_results, 'cos_global'),
    ]:
        print(f"\n{'='*50}", flush=True)
        print(f"  {part_name}", flush=True)
        print(f"{'='*50}", flush=True)

        easy = [r for r in results if not r['is_ood']]
        ood = [r for r in results if r['is_ood']]
        labels = [0]*len(easy) + [1]*len(ood)
        all_r = easy + ood

        # Compute AUROCs
        signals = {
            'Cosine distance': [r[cos_key] for r in all_r],
            'Action spread (std)': [r['action_spread'] for r in all_r],
            'Action range': [r['action_range'] for r in all_r],
            'Action roughness': [r['action_roughness'] for r in all_r],
            'Center deviation': [r['center_dev'] for r in all_r],
            'Mean action entropy': [r['mean_entropy'] for r in all_r],
            'Max action entropy': [r['max_entropy'] for r in all_r],
            'Entropy std': [r['entropy_std'] for r in all_r],
            'Mean max conf (inv)': [1 - r['mean_max_conf'] for r in all_r],
            'Action mass (inv)': [1 - r['action_mass'] for r in all_r],
        }

        if part_name.startswith("Part B"):
            signals['Per-scene cosine'] = [r['cos_per_scene'] for r in all_r]

        print("\n  Signal AUROC Comparison:", flush=True)
        print("  " + "-"*55, flush=True)
        for name, scores in sorted(signals.items(), key=lambda x: -roc_auc_score(labels, x[1])):
            auroc = roc_auc_score(labels, scores)
            print(f"    {name:<30}: AUROC = {auroc:.3f}", flush=True)

        # Per-scenario statistics
        print(f"\n  Per-Scenario Mean Values:", flush=True)
        print(f"  {'Scenario':<15} {'OOD':>4} {'Cos':>8} {'Spread':>8} {'Rough':>8} "
              f"{'CtrDev':>8} {'Entropy':>8} {'Mass':>8}", flush=True)
        print("  " + "-"*75, flush=True)
        scenarios = set(r['scenario'] for r in results)
        for s in sorted(scenarios):
            s_r = [r for r in results if r['scenario'] == s]
            is_ood = s_r[0]['is_ood']
            cos = np.mean([r[cos_key] for r in s_r])
            spread = np.mean([r['action_spread'] for r in s_r])
            rough = np.mean([r['action_roughness'] for r in s_r])
            ctr = np.mean([r['center_dev'] for r in s_r])
            ent = np.mean([r['mean_entropy'] for r in s_r])
            mass = np.mean([r['action_mass'] for r in s_r])
            label = 'Yes' if is_ood else 'No'
            print(f"  {s:<15} {label:>4} {cos:>8.4f} {spread:>8.1f} {rough:>8.1f} "
                  f"{ctr:>8.1f} {ent:>8.3f} {mass:>8.4f}", flush=True)

        # Best combination
        cos_scores = np.array(signals['Cosine distance'])
        spread_scores = np.array(signals['Action spread (std)'])
        rough_scores = np.array(signals['Action roughness'])

        # Normalize to [0, 1]
        def norm01(arr):
            mn, mx = arr.min(), arr.max()
            if mx - mn < 1e-10:
                return np.zeros_like(arr)
            return (arr - mn) / (mx - mn)

        cos_n = norm01(cos_scores)
        spread_n = norm01(spread_scores)
        rough_n = norm01(rough_scores)

        combo = cos_n * 0.6 + spread_n * 0.2 + rough_n * 0.2
        combo_auroc = roc_auc_score(labels, combo)
        print(f"\n  Combined (0.6*cos + 0.2*spread + 0.2*rough): AUROC = {combo_auroc:.3f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'action_plausibility',
        'experiment_number': 43,
        'timestamp': timestamp,
        'n_cal_simple': len(cal_simple),
        'n_cal_realistic': len(cal_all),
        'simple_results': simple_results,
        'realistic_results': realistic_results,
    }
    output_path = os.path.join(RESULTS_DIR, f"action_plausibility_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
