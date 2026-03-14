"""
Ensemble OOD Detection: Combining Multiple Detectors.

Tests various fusion strategies for combining cosine distance,
attention max, attention entropy, and output-based signals:
1. Simple averaging
2. Weighted average (learned on calibration)
3. Maximum rule (conservative)
4. Product rule (aggressive)
5. Voting with threshold

The hypothesis is that ensembles can close the gap between
calibrated (cosine) and calibration-free (attention) approaches,
especially for near-OOD detection.

Experiment 69 in the CalibDrive series.
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

# Near-OOD
def create_twilight_highway(idx):
    rng = np.random.default_rng(idx * 5010)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [70, 50, 80]
    img[SIZE[0]//2:] = [60, 60, 60]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [200, 200, 100]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_wet_highway(idx):
    rng = np.random.default_rng(idx * 5011)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [100, 120, 140]
    img[SIZE[0]//2:] = [50, 55, 65]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [180, 180, 200]
    for y in range(SIZE[0]//2, SIZE[0]):
        if rng.random() > 0.7:
            img[y, :] = np.clip(img[y, :].astype(np.int16) + 20, 0, 255).astype(np.uint8)
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_construction(idx):
    rng = np.random.default_rng(idx * 5012)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]
    img[SIZE[0]//2:] = [80, 80, 80]
    barrier_x = rng.integers(SIZE[1]//4, 3*SIZE[1]//4)
    img[SIZE[0]//2:3*SIZE[0]//4, max(0,barrier_x-20):barrier_x+20] = [255, 140, 0]
    for cone_x in rng.integers(0, SIZE[1], size=5):
        img[SIZE[0]-30:SIZE[0]-10, max(0,cone_x-5):cone_x+5] = [255, 100, 0]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_occluded(idx):
    rng = np.random.default_rng(idx * 5013)
    img = create_highway(idx + 7000)
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
    rng = np.random.default_rng(idx * 5014)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [200, 200, 210]
    img[SIZE[0]//2:] = [220, 220, 230]
    img[SIZE[0]//2:, SIZE[1]//2-2:SIZE[1]//2+2] = [180, 180, 190]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

# Far-OOD
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

    if hasattr(fwd, 'logits') and fwd.logits is not None:
        logits = fwd.logits[0, -1, :].float().cpu().numpy()
        probs = np.exp(logits - np.max(logits))
        probs = probs / probs.sum()
        result['msp'] = float(np.max(probs))
        result['energy'] = float(np.log(np.sum(np.exp(logits - np.max(logits)))) + np.max(logits))

    return result


def cosine_dist(a, b):
    return 1.0 - float(np.dot(a / (np.linalg.norm(a) + 1e-10),
                               b / (np.linalg.norm(b) + 1e-10)))


def normalize_scores(scores):
    """Min-max normalize to [0, 1]."""
    arr = np.array(scores)
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-10:
        return np.ones_like(arr) * 0.5
    return (arr - mn) / (mx - mn)


def main():
    print("=" * 70, flush=True)
    print("ENSEMBLE OOD DETECTION", flush=True)
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
    cal_signals = []
    for fn in [create_highway, create_urban]:
        for i in range(15):
            sig = extract_signals(model, processor,
                                  Image.fromarray(fn(i + 9000)), prompt)
            if 'hidden' in sig:
                cal_hidden.append(sig['hidden'])
            cal_signals.append(sig)
    centroid = np.mean(cal_hidden, axis=0)
    print(f"  Calibration: {len(cal_hidden)} samples", flush=True)

    # Test scenarios
    test_fns = {
        'highway': (create_highway, 'id', 12),
        'urban': (create_urban, 'id', 12),
        'twilight': (create_twilight_highway, 'near_ood', 8),
        'wet': (create_wet_highway, 'near_ood', 8),
        'construction': (create_construction, 'near_ood', 8),
        'occluded': (create_occluded, 'near_ood', 8),
        'snow': (create_snow, 'near_ood', 8),
        'noise': (create_noise, 'far_ood', 8),
        'indoor': (create_indoor, 'far_ood', 8),
        'blackout': (create_blackout, 'far_ood', 8),
    }

    all_data = []
    cnt = 0
    total = sum(v[2] for v in test_fns.values())
    for scene, (fn, ood_type, n) in test_fns.items():
        for i in range(n):
            cnt += 1
            sig = extract_signals(model, processor,
                                  Image.fromarray(fn(i + 400)), prompt)
            sig['scenario'] = scene
            sig['ood_type'] = ood_type
            sig['is_ood'] = ood_type != 'id'
            if 'hidden' in sig:
                sig['cosine'] = cosine_dist(sig['hidden'], centroid)
            all_data.append(sig)
            if cnt % 10 == 0:
                print(f"  [{cnt}/{total}] {scene}", flush=True)

    print(f"\nCollected {len(all_data)} samples.", flush=True)

    # Split
    id_data = [d for d in all_data if d['ood_type'] == 'id']
    near_ood = [d for d in all_data if d['ood_type'] == 'near_ood']
    far_ood = [d for d in all_data if d['ood_type'] == 'far_ood']
    all_ood = near_ood + far_ood

    # Extract raw signal arrays
    def get_signals(data_list, signal_name, negate=False):
        vals = []
        for d in data_list:
            if signal_name == 'cosine':
                vals.append(d.get('cosine', 0))
            elif signal_name == 'attn_max':
                vals.append(d.get('attn_max', 0))
            elif signal_name == 'attn_entropy':
                v = d.get('attn_entropy', 0)
                vals.append(-v if negate else v)
            elif signal_name == 'msp':
                vals.append(-d.get('msp', 1))
            elif signal_name == 'energy':
                vals.append(-d.get('energy', 0))
        return np.array(vals)

    print("\n" + "=" * 70, flush=True)
    print("ENSEMBLE RESULTS", flush=True)
    print("=" * 70, flush=True)

    results = {}

    for test_name, ood_group in [('far_ood', far_ood), ('near_ood', near_ood), ('all_ood', all_ood)]:
        print(f"\n  === {test_name.upper()} ===", flush=True)
        results[test_name] = {}

        combined = id_data + list(ood_group)
        labels = np.array([0]*len(id_data) + [1]*len(ood_group))

        # Individual signals
        signal_names = ['cosine', 'attn_max', 'attn_entropy', 'msp', 'energy']
        raw_signals = {}
        for sn in signal_names:
            raw_signals[sn] = get_signals(combined, sn, negate=(sn == 'attn_entropy'))

        # Normalize all signals
        norm_signals = {}
        for sn in signal_names:
            norm_signals[sn] = normalize_scores(raw_signals[sn])

        # Individual baselines
        for sn in signal_names:
            auroc = roc_auc_score(labels, raw_signals[sn])
            results[test_name][sn] = float(auroc)
            print(f"    {sn:<20}: AUROC={auroc:.3f}", flush=True)

        # Ensemble strategies
        # 1. Simple average (all signals)
        avg_all = np.mean([norm_signals[sn] for sn in signal_names], axis=0)
        auroc = roc_auc_score(labels, avg_all)
        results[test_name]['ensemble_avg_all'] = float(auroc)
        print(f"    {'Avg (all 5)':<20}: AUROC={auroc:.3f}", flush=True)

        # 2. Average top-3 (cosine + attn_max + attn_entropy)
        avg_top3 = np.mean([norm_signals['cosine'], norm_signals['attn_max'],
                            norm_signals['attn_entropy']], axis=0)
        auroc = roc_auc_score(labels, avg_top3)
        results[test_name]['ensemble_avg_top3'] = float(auroc)
        print(f"    {'Avg (top 3)':<20}: AUROC={auroc:.3f}", flush=True)

        # 3. Cosine + Attn Max only
        avg_cos_attn = np.mean([norm_signals['cosine'], norm_signals['attn_max']], axis=0)
        auroc = roc_auc_score(labels, avg_cos_attn)
        results[test_name]['ensemble_cos_attn'] = float(auroc)
        print(f"    {'Avg (cos+attn)':<20}: AUROC={auroc:.3f}", flush=True)

        # 4. Maximum rule (take highest anomaly score)
        max_cos_attn = np.maximum(norm_signals['cosine'], norm_signals['attn_max'])
        auroc = roc_auc_score(labels, max_cos_attn)
        results[test_name]['ensemble_max_cos_attn'] = float(auroc)
        print(f"    {'Max (cos+attn)':<20}: AUROC={auroc:.3f}", flush=True)

        # 5. Weighted: 0.6 cosine + 0.4 attn (emphasize calibrated)
        weighted_cos_heavy = 0.6 * norm_signals['cosine'] + 0.4 * norm_signals['attn_max']
        auroc = roc_auc_score(labels, weighted_cos_heavy)
        results[test_name]['ensemble_w60_cos'] = float(auroc)
        print(f"    {'W(0.6cos+0.4at)':<20}: AUROC={auroc:.3f}", flush=True)

        # 6. Weighted: 0.4 cosine + 0.6 attn (emphasize cal-free)
        weighted_attn_heavy = 0.4 * norm_signals['cosine'] + 0.6 * norm_signals['attn_max']
        auroc = roc_auc_score(labels, weighted_attn_heavy)
        results[test_name]['ensemble_w60_attn'] = float(auroc)
        print(f"    {'W(0.4cos+0.6at)':<20}: AUROC={auroc:.3f}", flush=True)

        # 7. Product rule
        prod = norm_signals['cosine'] * norm_signals['attn_max']
        auroc = roc_auc_score(labels, prod)
        results[test_name]['ensemble_product'] = float(auroc)
        print(f"    {'Product (cos*at)':<20}: AUROC={auroc:.3f}", flush=True)

        # 8. Voting (majority of top-3 signals above median threshold)
        vote = np.zeros(len(combined))
        for sn in ['cosine', 'attn_max', 'attn_entropy']:
            median_val = np.median(norm_signals[sn])
            vote += (norm_signals[sn] > median_val).astype(float)
        auroc = roc_auc_score(labels, vote)
        results[test_name]['ensemble_vote'] = float(auroc)
        print(f"    {'Vote (top 3)':<20}: AUROC={auroc:.3f}", flush=True)

        # 9. Adaptive: use attn for far-OOD, cosine for near-OOD
        # Proxy: if attn_max is very high → likely far-OOD → trust attn
        # If attn_max is moderate → might be near-OOD → blend with cosine
        adaptive = np.where(
            norm_signals['attn_max'] > 0.8,
            norm_signals['attn_max'],  # trust attention for obvious OOD
            0.7 * norm_signals['cosine'] + 0.3 * norm_signals['attn_max']  # blend for subtle
        )
        auroc = roc_auc_score(labels, adaptive)
        results[test_name]['ensemble_adaptive'] = float(auroc)
        print(f"    {'Adaptive':<20}: AUROC={auroc:.3f}", flush=True)

    # Weight sweep
    print("\n  === WEIGHT SWEEP (cos_w, attn_w) ===", flush=True)
    weight_sweep = {}
    combined_all = id_data + list(all_ood)
    labels_all = np.array([0]*len(id_data) + [1]*len(all_ood))
    cos_norm = normalize_scores(get_signals(combined_all, 'cosine'))
    attn_norm = normalize_scores(get_signals(combined_all, 'attn_max'))

    combined_near = id_data + list(near_ood)
    labels_near = np.array([0]*len(id_data) + [1]*len(near_ood))
    cos_norm_n = normalize_scores(get_signals(combined_near, 'cosine'))
    attn_norm_n = normalize_scores(get_signals(combined_near, 'attn_max'))

    for w in np.arange(0, 1.05, 0.1):
        w = round(w, 1)
        blend_all = w * cos_norm + (1 - w) * attn_norm
        blend_near = w * cos_norm_n + (1 - w) * attn_norm_n
        auroc_all = roc_auc_score(labels_all, blend_all)
        auroc_near = roc_auc_score(labels_near, blend_near)
        weight_sweep[str(w)] = {
            'all_ood': float(auroc_all),
            'near_ood': float(auroc_near),
        }
        print(f"    cos_w={w:.1f}: all_AUROC={auroc_all:.3f}, near_AUROC={auroc_near:.3f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'ensemble_detection',
        'experiment_number': 69,
        'timestamp': timestamp,
        'n_id': len(id_data),
        'n_near_ood': len(near_ood),
        'n_far_ood': len(far_ood),
        'results': results,
        'weight_sweep': weight_sweep,
    }
    output_path = os.path.join(RESULTS_DIR, f"ensemble_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
