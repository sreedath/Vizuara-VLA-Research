"""
Near-OOD Detection Challenge.

Tests detection on "near-OOD" inputs that are semantically similar to ID
but should still be flagged:
1. Highway at different time of day (twilight)
2. Highway with slightly different road color
3. Urban with different building colors
4. Highway with occlusion (partial obstruction)

These are much harder than pixel-level OOD (noise, blackout) and test
the practical limits of our approach.

Experiment 67 in the CalibDrive series.
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

# Near-OOD types
def create_twilight_highway(idx):
    """Highway at twilight — darker sky, orange tint."""
    rng = np.random.default_rng(idx * 5010)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [70, 50, 80]  # Dark purple sky
    img[SIZE[0]//2:] = [60, 60, 60]  # Dark road
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [200, 200, 100]  # Yellowish lines
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_wet_highway(idx):
    """Wet highway — reflective road surface."""
    rng = np.random.default_rng(idx * 5011)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [100, 120, 140]  # Overcast sky
    img[SIZE[0]//2:] = [50, 55, 65]  # Wet dark road
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [180, 180, 200]  # Reflective lines
    # Add shimmer/reflection effect
    for y in range(SIZE[0]//2, SIZE[0]):
        if rng.random() > 0.7:
            img[y, :] = np.clip(img[y, :].astype(np.int16) + 20, 0, 255).astype(np.uint8)
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_construction(idx):
    """Highway with construction barriers — partial road change."""
    rng = np.random.default_rng(idx * 5012)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]  # Normal sky
    img[SIZE[0]//2:] = [80, 80, 80]  # Normal road
    # Add orange construction barriers
    barrier_x = rng.integers(SIZE[1]//4, 3*SIZE[1]//4)
    img[SIZE[0]//2:3*SIZE[0]//4, max(0,barrier_x-20):barrier_x+20] = [255, 140, 0]
    # Add cones
    for cone_x in rng.integers(0, SIZE[1], size=5):
        img[SIZE[0]-30:SIZE[0]-10, max(0,cone_x-5):cone_x+5] = [255, 100, 0]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_occluded(idx):
    """Highway with partial camera occlusion (dirt/water splash)."""
    rng = np.random.default_rng(idx * 5013)
    img = create_highway(idx + 7000)
    # Add random dark patches (occlusion)
    for _ in range(rng.integers(3, 8)):
        cx, cy = rng.integers(0, SIZE[1]), rng.integers(0, SIZE[0])
        r = rng.integers(10, 40)
        for dy in range(-r, r):
            for dx in range(-r, r):
                if dx*dx + dy*dy <= r*r:
                    ny, nx = cy+dy, cx+dx
                    if 0 <= ny < SIZE[0] and 0 <= nx < SIZE[1]:
                        img[ny, nx] = np.clip(img[ny, nx].astype(np.int16) - 80, 0, 255).astype(np.uint8)
    return img

def create_snow(idx):
    """Snowy highway — white ground, limited visibility."""
    rng = np.random.default_rng(idx * 5014)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [200, 200, 210]  # White-ish sky
    img[SIZE[0]//2:] = [220, 220, 230]  # Snowy road
    # Faint lane lines
    img[SIZE[0]//2:, SIZE[1]//2-2:SIZE[1]//2+2] = [180, 180, 190]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

# Standard OOD
def create_noise(idx):
    rng = np.random.default_rng(idx * 5003)
    return rng.integers(0, 256, (*SIZE, 3), dtype=np.uint8)

def create_blackout(idx):
    return np.zeros((*SIZE, 3), dtype=np.uint8)


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
        result['attn_entropy'] = float(np.mean([
            -np.sum((last_attn[h]+1e-10) * np.log(last_attn[h]+1e-10))
            for h in range(n_heads)
        ]))

    if hasattr(fwd, 'hidden_states') and fwd.hidden_states:
        result['hidden'] = fwd.hidden_states[-1][0, -1, :].float().cpu().numpy()

    return result


def cosine_dist(a, b):
    return 1.0 - float(np.dot(a / (np.linalg.norm(a) + 1e-10),
                               b / (np.linalg.norm(b) + 1e-10)))


def main():
    print("=" * 70, flush=True)
    print("NEAR-OOD DETECTION CHALLENGE", flush=True)
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

    # Calibration
    print("\nCalibrating...", flush=True)
    cal_hidden = []
    for fn in [create_highway, create_urban]:
        for i in range(10):
            sig = extract_signals(model, processor,
                                  Image.fromarray(fn(i + 9000)), prompt)
            if 'hidden' in sig:
                cal_hidden.append(sig['hidden'])
    centroid = np.mean(cal_hidden, axis=0)
    print(f"  Calibration: {len(cal_hidden)} samples", flush=True)

    # Test scenarios
    test_fns = {
        # ID
        'highway': (create_highway, 'id', 10),
        'urban': (create_urban, 'id', 10),
        # Near-OOD (should be flagged as unusual but structurally similar to ID)
        'twilight': (create_twilight_highway, 'near_ood', 8),
        'wet': (create_wet_highway, 'near_ood', 8),
        'construction': (create_construction, 'near_ood', 8),
        'occluded': (create_occluded, 'near_ood', 8),
        'snow': (create_snow, 'near_ood', 8),
        # Far-OOD
        'noise': (create_noise, 'far_ood', 6),
        'blackout': (create_blackout, 'far_ood', 6),
    }

    all_data = []
    cnt = 0
    total = sum(v[2] for v in test_fns.values())
    for scene, (fn, ood_type, n) in test_fns.items():
        for i in range(n):
            cnt += 1
            sig = extract_signals(model, processor,
                                  Image.fromarray(fn(i + 200)), prompt)
            sig['scenario'] = scene
            sig['ood_type'] = ood_type
            sig['is_ood'] = ood_type != 'id'
            if 'hidden' in sig:
                sig['cosine'] = cosine_dist(sig['hidden'], centroid)
            all_data.append(sig)
            if cnt % 10 == 0:
                print(f"  [{cnt}/{total}] {scene}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("RESULTS", flush=True)
    print("=" * 70, flush=True)

    # Per-scenario statistics
    print("\n  Per-scenario scores:", flush=True)
    print(f"    {'Scenario':<15} {'Type':<10} {'Cosine':>8} {'Attn Max':>10} {'Attn Ent':>10}",
          flush=True)
    print("    " + "-" * 55, flush=True)

    per_scenario = {}
    for scene in sorted(set(d['scenario'] for d in all_data)):
        scene_data = [d for d in all_data if d['scenario'] == scene]
        cos = [d.get('cosine', 0) for d in scene_data]
        amax = [d.get('attn_max', 0) for d in scene_data]
        aent = [d.get('attn_entropy', 0) for d in scene_data]
        ood_type = scene_data[0]['ood_type']
        per_scenario[scene] = {
            'ood_type': ood_type,
            'cosine_mean': float(np.mean(cos)),
            'cosine_std': float(np.std(cos)),
            'attn_max_mean': float(np.mean(amax)),
            'attn_entropy_mean': float(np.mean(aent)),
        }
        print(f"    {scene:<15} {ood_type:<10} {np.mean(cos):>8.4f} "
              f"{np.mean(amax):>10.4f} {np.mean(aent):>10.4f}", flush=True)

    # ID vs Near-OOD detection
    print("\n  Detection AUROC:", flush=True)
    id_data = [d for d in all_data if d['ood_type'] == 'id']
    near_ood = [d for d in all_data if d['ood_type'] == 'near_ood']
    far_ood = [d for d in all_data if d['ood_type'] == 'far_ood']

    for test_name, ood_group in [('Near-OOD', near_ood), ('Far-OOD', far_ood), ('All OOD', near_ood + far_ood)]:
        labels = [0]*len(id_data) + [1]*len(ood_group)
        for sig_name in ['cosine', 'attn_max', 'attn_entropy']:
            if sig_name == 'cosine':
                scores = [d.get('cosine', 0) for d in id_data] + [d.get('cosine', 0) for d in ood_group]
            elif sig_name == 'attn_max':
                scores = [d.get('attn_max', 0) for d in id_data] + [d.get('attn_max', 0) for d in ood_group]
            else:
                scores = [-d.get('attn_entropy', 0) for d in id_data] + [-d.get('attn_entropy', 0) for d in ood_group]
            auroc = roc_auc_score(labels, scores)
            print(f"    {test_name:<15} {sig_name:<15}: AUROC={auroc:.3f}", flush=True)

    # Per near-OOD type detection
    print("\n  Per near-OOD type:", flush=True)
    for scene in ['twilight', 'wet', 'construction', 'occluded', 'snow']:
        scene_data = [d for d in all_data if d['scenario'] == scene]
        labels = [0]*len(id_data) + [1]*len(scene_data)
        for sig_name in ['cosine', 'attn_max']:
            if sig_name == 'cosine':
                scores = [d.get('cosine', 0) for d in id_data] + [d.get('cosine', 0) for d in scene_data]
            else:
                scores = [d.get('attn_max', 0) for d in id_data] + [d.get('attn_max', 0) for d in scene_data]
            auroc = roc_auc_score(labels, scores)
            print(f"    {scene:<15} {sig_name}: AUROC={auroc:.3f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'near_ood',
        'experiment_number': 67,
        'timestamp': timestamp,
        'n_id': len(id_data),
        'n_near_ood': len(near_ood),
        'n_far_ood': len(far_ood),
        'per_scenario': per_scenario,
    }
    output_path = os.path.join(RESULTS_DIR, f"near_ood_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
