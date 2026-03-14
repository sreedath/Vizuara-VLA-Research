"""
Action Prediction Consistency Under OOD.

Analyzes how OOD inputs affect action predictions: do OOD images
produce erratic/inconsistent actions? We measure action variance
across similar inputs for ID vs OOD, testing whether OOD not only
shifts hidden states but also destabilizes action outputs.

Experiment 82 in the CalibDrive series.
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

def create_twilight_highway(idx):
    rng = np.random.default_rng(idx * 5010)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [70, 50, 80]
    img[SIZE[0]//2:] = [60, 60, 60]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [200, 200, 100]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_snow(idx):
    rng = np.random.default_rng(idx * 5014)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [200, 200, 210]
    img[SIZE[0]//2:] = [220, 220, 230]
    img[SIZE[0]//2:, SIZE[1]//2-2:SIZE[1]//2+2] = [180, 180, 190]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def extract_action_and_hidden(model, processor, image, prompt):
    """Extract both action tokens and hidden states."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)

    hidden = None
    if hasattr(fwd, 'hidden_states') and fwd.hidden_states:
        hidden = fwd.hidden_states[-1][0, -1, :].float().cpu().numpy()

    # Get predicted action token (argmax of logits)
    logits = fwd.logits[0, -1, :].float().cpu().numpy()
    action_token = int(np.argmax(logits))
    top_prob = float(np.exp(logits[action_token] - np.max(logits)) /
                     np.sum(np.exp(logits - np.max(logits))))

    # Get top-5 token distribution
    top5_idx = np.argsort(logits)[-5:][::-1]
    top5_logits = logits[top5_idx]
    top5_probs = np.exp(top5_logits - np.max(top5_logits))
    top5_probs = top5_probs / np.sum(np.exp(logits - np.max(logits)))

    # Entropy of the action distribution
    probs = np.exp(logits - np.max(logits))
    probs = probs / probs.sum()
    entropy = -float(np.sum(probs * np.log(probs + 1e-10)))

    return {
        'hidden': hidden,
        'action_token': action_token,
        'top_prob': top_prob,
        'entropy': entropy,
        'top5_tokens': top5_idx.tolist(),
        'top5_probs': top5_probs.tolist(),
    }


def cosine_dist(a, b):
    return 1.0 - float(np.dot(a / (np.linalg.norm(a) + 1e-10),
                               b / (np.linalg.norm(b) + 1e-10)))


def main():
    print("=" * 70, flush=True)
    print("ACTION PREDICTION CONSISTENCY", flush=True)
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

    # Calibrate
    print("\nCalibrating...", flush=True)
    cal_hidden = []
    for fn in [create_highway, create_urban]:
        for i in range(15):
            r = extract_action_and_hidden(model, processor,
                                           Image.fromarray(fn(i + 9000)), prompt)
            if r['hidden'] is not None:
                cal_hidden.append(r['hidden'])
    centroid = np.mean(cal_hidden, axis=0)
    print(f"  {len(cal_hidden)} calibration samples", flush=True)

    # Collect detailed results per scenario
    scenarios = {
        'highway': (create_highway, range(500, 516)),
        'urban': (create_urban, range(500, 516)),
        'noise': (create_noise, range(500, 510)),
        'indoor': (create_indoor, range(500, 510)),
        'twilight': (create_twilight_highway, range(500, 510)),
        'snow': (create_snow, range(500, 510)),
    }

    all_results = {}
    cnt = 0
    total = sum(len(list(ids)) for _, ids in scenarios.values())

    for name, (fn, indices) in scenarios.items():
        scenario_data = []
        for i in indices:
            cnt += 1
            r = extract_action_and_hidden(model, processor,
                                           Image.fromarray(fn(i)), prompt)
            if r['hidden'] is not None:
                r['cosine_dist'] = cosine_dist(r['hidden'], centroid)
                del r['hidden']  # Don't store full hidden state
                scenario_data.append(r)
            if cnt % 10 == 0:
                print(f"  [{cnt}/{total}] {name}", flush=True)

        # Aggregate statistics
        tokens = [d['action_token'] for d in scenario_data]
        entropies = [d['entropy'] for d in scenario_data]
        top_probs = [d['top_prob'] for d in scenario_data]
        cos_dists = [d['cosine_dist'] for d in scenario_data]

        unique_tokens = len(set(tokens))
        token_mode = max(set(tokens), key=tokens.count)
        token_agreement = tokens.count(token_mode) / len(tokens)

        all_results[name] = {
            'n_samples': len(scenario_data),
            'unique_action_tokens': unique_tokens,
            'token_agreement_rate': float(token_agreement),
            'modal_token': int(token_mode),
            'mean_entropy': float(np.mean(entropies)),
            'std_entropy': float(np.std(entropies)),
            'mean_top_prob': float(np.mean(top_probs)),
            'std_top_prob': float(np.std(top_probs)),
            'mean_cosine_dist': float(np.mean(cos_dists)),
            'is_ood': name not in ('highway', 'urban'),
        }
        print(f"  {name}: {unique_tokens} unique tokens, "
              f"agreement={token_agreement:.2f}, "
              f"entropy={np.mean(entropies):.3f}", flush=True)

    # Summary
    print("\n" + "=" * 70, flush=True)
    print("ACTION CONSISTENCY SUMMARY", flush=True)
    print("=" * 70, flush=True)

    id_entropies = []
    ood_entropies = []
    id_agreements = []
    ood_agreements = []

    for name, r in all_results.items():
        label = "OOD" if r['is_ood'] else "ID"
        print(f"  {name:<12} [{label}]: "
              f"unique_tokens={r['unique_action_tokens']}, "
              f"agreement={r['token_agreement_rate']:.2f}, "
              f"entropy={r['mean_entropy']:.3f}, "
              f"top_prob={r['mean_top_prob']:.4f}", flush=True)
        if r['is_ood']:
            ood_entropies.append(r['mean_entropy'])
            ood_agreements.append(r['token_agreement_rate'])
        else:
            id_entropies.append(r['mean_entropy'])
            id_agreements.append(r['token_agreement_rate'])

    print(f"\n  ID  mean entropy: {np.mean(id_entropies):.3f}, "
          f"agreement: {np.mean(id_agreements):.2f}", flush=True)
    print(f"  OOD mean entropy: {np.mean(ood_entropies):.3f}, "
          f"agreement: {np.mean(ood_agreements):.2f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'action_consistency',
        'experiment_number': 82,
        'timestamp': timestamp,
        'n_cal': len(cal_hidden),
        'results': all_results,
        'summary': {
            'id_mean_entropy': float(np.mean(id_entropies)),
            'ood_mean_entropy': float(np.mean(ood_entropies)),
            'id_mean_agreement': float(np.mean(id_agreements)),
            'ood_mean_agreement': float(np.mean(ood_agreements)),
        }
    }
    output_path = os.path.join(RESULTS_DIR, f"action_consistency_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
