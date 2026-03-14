"""
Cross-Domain Transfer of OOD Detection.

Tests whether calibration on one driving domain (e.g., highway)
transfers to detecting OOD inputs when the test ID distribution
shifts to a different domain (e.g., urban).

Critical question: Does the detector need recalibration when the
deployment domain changes?

Experiment 62 in the CalibDrive series.
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
    print("CROSS-DOMAIN TRANSFER ANALYSIS", flush=True)
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

    # Collect embeddings for all domains
    print("\nCollecting domain embeddings...", flush=True)
    domains = {
        'highway': (create_highway, 20),
        'urban': (create_urban, 20),
    }
    ood_domains = {
        'noise': (create_noise, 10),
        'indoor': (create_indoor, 10),
        'inverted': (create_inverted, 10),
        'blackout': (create_blackout, 10),
    }

    domain_hidden = {}
    cnt = 0
    total = sum(v[1] for v in {**domains, **ood_domains}.values())
    for name, (fn, n) in {**domains, **ood_domains}.items():
        domain_hidden[name] = []
        for i in range(n):
            cnt += 1
            h = extract_hidden(model, processor,
                               Image.fromarray(fn(i + 200)), prompt)
            domain_hidden[name].append(h)
            if cnt % 10 == 0:
                print(f"  [{cnt}/{total}] {name}", flush=True)

    # Cross-domain experiments
    print("\n" + "=" * 70, flush=True)
    print("CROSS-DOMAIN TRANSFER RESULTS", flush=True)
    print("=" * 70, flush=True)

    cal_configs = {
        'highway_only': ['highway'],
        'urban_only': ['urban'],
        'mixed': ['highway', 'urban'],
    }

    ood_types = list(ood_domains.keys())
    results = {}

    for cal_name, cal_domains in cal_configs.items():
        print(f"\n  Calibration: {cal_name}", flush=True)
        cal_h = []
        for d in cal_domains:
            cal_h.extend(domain_hidden[d][:10])  # First 10 for calibration
        centroid = np.mean(cal_h, axis=0)

        results[cal_name] = {}

        # Test on each ID domain
        for test_id_name in ['highway', 'urban']:
            test_id = domain_hidden[test_id_name][10:]  # Last 10 for testing
            id_scores = [cosine_dist(h, centroid) for h in test_id]

            for ood_type in ood_types:
                test_ood = domain_hidden[ood_type]
                ood_scores = [cosine_dist(h, centroid) for h in test_ood]

                labels = [0]*len(id_scores) + [1]*len(ood_scores)
                scores = id_scores + ood_scores
                auroc = roc_auc_score(labels, scores)

                key = f"{test_id_name}_vs_{ood_type}"
                results[cal_name][key] = float(auroc)
                print(f"    {cal_name} → test_id={test_id_name} vs {ood_type}: AUROC={auroc:.3f}",
                      flush=True)

        # Overall AUROC (all ID vs all OOD)
        all_id = domain_hidden['highway'][10:] + domain_hidden['urban'][10:]
        all_ood = []
        for ood_type in ood_types:
            all_ood.extend(domain_hidden[ood_type])
        id_scores = [cosine_dist(h, centroid) for h in all_id]
        ood_scores = [cosine_dist(h, centroid) for h in all_ood]
        labels = [0]*len(id_scores) + [1]*len(ood_scores)
        scores = id_scores + ood_scores
        overall = roc_auc_score(labels, scores)
        results[cal_name]['overall'] = float(overall)
        print(f"    Overall: AUROC={overall:.3f}", flush=True)

    # Cross-calibration analysis
    print("\n  Cross-calibration summary:", flush=True)
    print(f"    {'Calibration':<15} {'Highway Test':>15} {'Urban Test':>15} {'Overall':>10}",
          flush=True)
    print("    " + "-" * 55, flush=True)
    for cal_name in cal_configs:
        hw_avg = np.mean([results[cal_name].get(f'highway_vs_{o}', 0.5) for o in ood_types])
        ur_avg = np.mean([results[cal_name].get(f'urban_vs_{o}', 0.5) for o in ood_types])
        overall = results[cal_name]['overall']
        print(f"    {cal_name:<15} {hw_avg:>15.3f} {ur_avg:>15.3f} {overall:>10.3f}",
              flush=True)

    # Domain distance analysis
    print("\n  Domain centroid distances:", flush=True)
    hw_centroid = np.mean(domain_hidden['highway'], axis=0)
    ur_centroid = np.mean(domain_hidden['urban'], axis=0)
    mixed_centroid = np.mean(domain_hidden['highway'] + domain_hidden['urban'], axis=0)

    print(f"    Highway-Urban cosine: {cosine_dist(hw_centroid, ur_centroid):.4f}", flush=True)
    print(f"    Highway-Mixed cosine: {cosine_dist(hw_centroid, mixed_centroid):.4f}", flush=True)
    print(f"    Urban-Mixed cosine: {cosine_dist(ur_centroid, mixed_centroid):.4f}", flush=True)

    for ood_type in ood_types:
        ood_centroid = np.mean(domain_hidden[ood_type], axis=0)
        print(f"    {ood_type} to HW: {cosine_dist(ood_centroid, hw_centroid):.4f}, "
              f"to UR: {cosine_dist(ood_centroid, ur_centroid):.4f}, "
              f"to MX: {cosine_dist(ood_centroid, mixed_centroid):.4f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'cross_domain',
        'experiment_number': 62,
        'timestamp': timestamp,
        'n_per_domain': {k: len(v) for k, v in domain_hidden.items()},
        'results': results,
        'domain_distances': {
            'hw_ur': float(cosine_dist(hw_centroid, ur_centroid)),
            'hw_mx': float(cosine_dist(hw_centroid, mixed_centroid)),
            'ur_mx': float(cosine_dist(ur_centroid, mixed_centroid)),
        },
    }
    output_path = os.path.join(RESULTS_DIR, f"cross_domain_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
