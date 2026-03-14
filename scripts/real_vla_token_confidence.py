"""
Token Confidence Analysis for OOD Detection.

Analyzes the softmax probability (confidence) of predicted action
tokens for ID vs OOD inputs.  Hypotheses:
1. OOD inputs produce lower max probabilities
2. OOD inputs produce flatter probability distributions (higher entropy)
3. Token confidence can serve as an OOD signal

Experiment 127 in the CalibDrive series.
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
ACTION_TOKEN_BASE = 31744


def create_highway(idx):
    rng = np.random.default_rng(idx * 20001)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]
    img[SIZE[0]//2:] = [80, 80, 80]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [255, 255, 255]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_urban(idx):
    rng = np.random.default_rng(idx * 20002)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [135, 206, 235]
    img[SIZE[0]//3:SIZE[0]//2] = [139, 119, 101]
    img[SIZE[0]//2:] = [60, 60, 60]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_noise(idx):
    rng = np.random.default_rng(idx * 20003)
    return rng.integers(0, 256, (*SIZE, 3), dtype=np.uint8)

def create_indoor(idx):
    rng = np.random.default_rng(idx * 20004)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:] = [200, 180, 160]
    img[SIZE[0]//2:, :] = [100, 80, 60]
    img[:SIZE[0]//3, SIZE[1]//3:2*SIZE[1]//3] = [150, 200, 255]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_twilight_highway(idx):
    rng = np.random.default_rng(idx * 20010)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [70, 50, 80]
    img[SIZE[0]//2:] = [60, 60, 60]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [200, 200, 100]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_snow(idx):
    rng = np.random.default_rng(idx * 20014)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [200, 200, 210]
    img[SIZE[0]//2:] = [220, 220, 230]
    img[SIZE[0]//2:, SIZE[1]//2-2:SIZE[1]//2+2] = [180, 180, 190]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def get_token_confidences(model, processor, image, prompt):
    """Get confidence (softmax prob) for each generated action token."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

    gen_ids = output.sequences[0, inputs['input_ids'].shape[1]:]
    scores = output.scores  # tuple of logits at each step

    confidences = []
    entropies = []
    action_bins = []

    for step_idx, (tid, score) in enumerate(zip(gen_ids.tolist(), scores)):
        # Only analyze action tokens
        if ACTION_TOKEN_BASE <= tid < ACTION_TOKEN_BASE + 256:
            probs = torch.softmax(score[0].float(), dim=-1)
            max_prob = float(probs[tid])
            # Entropy over action tokens only
            action_probs = probs[ACTION_TOKEN_BASE:ACTION_TOKEN_BASE+256]
            action_probs = action_probs / (action_probs.sum() + 1e-10)
            entropy = float(-torch.sum(action_probs * torch.log(action_probs + 1e-10)))

            confidences.append(max_prob)
            entropies.append(entropy)
            action_bins.append(tid - ACTION_TOKEN_BASE)

    return {
        'confidences': confidences,
        'entropies': entropies,
        'action_bins': action_bins,
        'mean_confidence': float(np.mean(confidences)) if confidences else 0.0,
        'mean_entropy': float(np.mean(entropies)) if entropies else 0.0,
        'min_confidence': float(np.min(confidences)) if confidences else 0.0,
    }


def main():
    print("=" * 70, flush=True)
    print("TOKEN CONFIDENCE ANALYSIS", flush=True)
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

    categories = {
        'highway': (create_highway, 'ID'),
        'urban': (create_urban, 'ID'),
        'noise': (create_noise, 'OOD'),
        'indoor': (create_indoor, 'OOD'),
        'twilight': (create_twilight_highway, 'OOD'),
        'snow': (create_snow, 'OOD'),
    }

    print("\n--- Collecting token confidences ---", flush=True)
    results = {}
    for cat_name, (fn, group) in categories.items():
        print(f"\n  {cat_name} ({group}):", flush=True)
        all_confs = []
        all_entropies = []
        all_min_confs = []

        for i in range(10):
            img = Image.fromarray(fn(i + 2700))
            tc = get_token_confidences(model, processor, img, prompt)
            all_confs.append(tc['mean_confidence'])
            all_entropies.append(tc['mean_entropy'])
            all_min_confs.append(tc['min_confidence'])
            print(f"    {i}: mean_conf={tc['mean_confidence']:.4f}, mean_ent={tc['mean_entropy']:.3f}, "
                  f"min_conf={tc['min_confidence']:.4f}, n_tokens={len(tc['confidences'])}", flush=True)

        results[cat_name] = {
            'group': group,
            'mean_confidence': float(np.mean(all_confs)),
            'std_confidence': float(np.std(all_confs)),
            'mean_entropy': float(np.mean(all_entropies)),
            'std_entropy': float(np.std(all_entropies)),
            'mean_min_conf': float(np.mean(all_min_confs)),
            'all_confs': all_confs,
            'all_entropies': all_entropies,
        }

    # Detection using confidence
    print("\n--- Confidence as OOD Detector ---", flush=True)
    id_confs = []
    ood_confs = []
    id_ents = []
    ood_ents = []
    labels = []
    conf_scores = []
    ent_scores = []

    for cat_name, data in results.items():
        for c, e in zip(data['all_confs'], data['all_entropies']):
            if data['group'] == 'ID':
                id_confs.append(c)
                id_ents.append(e)
            else:
                ood_confs.append(c)
                ood_ents.append(e)
            labels.append(0 if data['group'] == 'ID' else 1)
            conf_scores.append(-c)  # lower confidence = more OOD
            ent_scores.append(e)    # higher entropy = more OOD

    labels = np.array(labels)
    conf_scores = np.array(conf_scores)
    ent_scores = np.array(ent_scores)

    conf_auroc = float(roc_auc_score(labels, conf_scores))
    ent_auroc = float(roc_auc_score(labels, ent_scores))

    id_conf_mean = float(np.mean(id_confs))
    ood_conf_mean = float(np.mean(ood_confs))
    id_ent_mean = float(np.mean(id_ents))
    ood_ent_mean = float(np.mean(ood_ents))

    conf_d = float((np.mean(-np.array(ood_confs)) - np.mean(-np.array(id_confs))) / (np.std(-np.array(id_confs)) + 1e-10))
    ent_d = float((ood_ent_mean - id_ent_mean) / (np.std(id_ents) + 1e-10))

    print(f"  Confidence detector: AUROC={conf_auroc:.4f}, d={conf_d:.2f}", flush=True)
    print(f"    ID mean conf: {id_conf_mean:.4f}, OOD mean conf: {ood_conf_mean:.4f}", flush=True)
    print(f"  Entropy detector: AUROC={ent_auroc:.4f}, d={ent_d:.2f}", flush=True)
    print(f"    ID mean entropy: {id_ent_mean:.3f}, OOD mean entropy: {ood_ent_mean:.3f}", flush=True)

    # Summary
    print("\n--- Per-Category Summary ---", flush=True)
    for cat_name, data in results.items():
        print(f"  {cat_name:12s}: conf={data['mean_confidence']:.4f}+/-{data['std_confidence']:.4f}, "
              f"ent={data['mean_entropy']:.3f}+/-{data['std_entropy']:.3f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'token_confidence',
        'experiment_number': 127,
        'timestamp': timestamp,
        'detection': {
            'confidence_auroc': conf_auroc,
            'confidence_d': conf_d,
            'entropy_auroc': ent_auroc,
            'entropy_d': ent_d,
            'id_conf_mean': id_conf_mean,
            'ood_conf_mean': ood_conf_mean,
            'id_ent_mean': id_ent_mean,
            'ood_ent_mean': ood_ent_mean,
        },
        'per_category': {
            k: {kk: vv for kk, vv in v.items() if kk not in ['all_confs', 'all_entropies']}
            for k, v in results.items()
        },
    }
    output_path = os.path.join(RESULTS_DIR, f"token_confidence_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
