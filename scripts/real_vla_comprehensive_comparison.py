"""
Comprehensive Method Comparison.

Single experiment running ALL detection methods on the same test set
for a fair, unified comparison table. Methods:
1. Cosine distance (calibrated)
2. Action mass (output-based)
3. MSP - max softmax probability (output-based)
4. Energy score (output-based)
5. Attention max (calibration-free)
6. Attention entropy (calibration-free)
7. Feature norm (calibration-free)
8. Combined: cosine + mass
9. Combined: attention + cosine
10. Best-of-both

Experiment 66 in the CalibDrive series.
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


def extract_all(model, processor, image, prompt):
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
        result['fwd_hidden'] = fwd.hidden_states[-1][0, -1, :].float().cpu().numpy()
        result['feature_norm'] = float(np.linalg.norm(result['fwd_hidden']))

    with torch.no_grad():
        gen = model.generate(
            **inputs, max_new_tokens=7, do_sample=False,
            output_scores=True, output_hidden_states=True,
            return_dict_in_generate=True,
        )

    if hasattr(gen, 'scores') and gen.scores:
        vocab_size = gen.scores[0].shape[-1]
        action_start = vocab_size - 256
        masses, msps, energies = [], [], []
        for score in gen.scores[:7]:
            logits = score[0].float()
            probs = torch.softmax(logits, dim=0)
            action_probs = probs[action_start:].cpu().numpy()
            masses.append(float(action_probs.sum()))
            msps.append(float(action_probs.max()))
            energies.append(float(torch.logsumexp(logits[action_start:], dim=0).item()))
        result['action_mass'] = float(np.mean(masses))
        result['msp'] = float(np.mean(msps))
        result['energy'] = float(np.mean(energies))

    if hasattr(gen, 'hidden_states') and gen.hidden_states:
        last_step = gen.hidden_states[-1]
        if isinstance(last_step, tuple):
            result['gen_hidden'] = last_step[-1][0, -1, :].float().cpu().numpy()
        else:
            result['gen_hidden'] = last_step[0, -1, :].float().cpu().numpy()

    return result


def cosine_dist(a, b):
    return 1.0 - float(np.dot(a / (np.linalg.norm(a) + 1e-10),
                               b / (np.linalg.norm(b) + 1e-10)))


def main():
    print("=" * 70, flush=True)
    print("COMPREHENSIVE METHOD COMPARISON", flush=True)
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

    print("\nCalibrating...", flush=True)
    cal_data = []
    for fn in [create_highway, create_urban]:
        for i in range(10):
            signals = extract_all(model, processor,
                                  Image.fromarray(fn(i + 9000)), prompt)
            cal_data.append(signals)
    print(f"  Calibration: {len(cal_data)} samples", flush=True)

    centroid = np.mean([d['gen_hidden'] for d in cal_data if 'gen_hidden' in d], axis=0)
    cal_norms = [d['feature_norm'] for d in cal_data if 'feature_norm' in d]

    print("\nCollecting test set...", flush=True)
    test_fns = {
        'highway': (create_highway, False, 12),
        'urban': (create_urban, False, 12),
        'noise': (create_noise, True, 8),
        'indoor': (create_indoor, True, 8),
        'inverted': (create_inverted, True, 8),
        'blackout': (create_blackout, True, 8),
    }

    test_data = []
    test_labels = []
    test_scenarios = []
    cnt = 0
    total = sum(v[2] for v in test_fns.values())
    for scene, (fn, is_ood, n) in test_fns.items():
        for i in range(n):
            cnt += 1
            signals = extract_all(model, processor,
                                  Image.fromarray(fn(i + 200)), prompt)
            test_data.append(signals)
            test_labels.append(1 if is_ood else 0)
            test_scenarios.append(scene)
            if cnt % 10 == 0:
                print(f"  [{cnt}/{total}] {scene}", flush=True)

    test_labels = np.array(test_labels)

    print("\n" + "=" * 70, flush=True)
    print("RESULTS", flush=True)
    print("=" * 70, flush=True)

    methods = {}
    methods['cosine'] = [cosine_dist(d['gen_hidden'], centroid) for d in test_data]
    methods['norm_diff'] = [abs(d['feature_norm'] - np.mean(cal_norms)) for d in test_data]
    methods['1-mass'] = [1 - d['action_mass'] for d in test_data]
    methods['1-msp'] = [1 - d['msp'] for d in test_data]
    methods['-energy'] = [-d['energy'] for d in test_data]
    methods['attn_max'] = [d['attn_max'] for d in test_data]
    methods['-attn_entropy'] = [-d['attn_entropy'] for d in test_data]

    def normalize(scores):
        arr = np.array(scores)
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-10)

    cos_n = normalize(methods['cosine'])
    mass_n = normalize(methods['1-mass'])
    attn_n = normalize(methods['attn_max'])
    ent_n = normalize(methods['-attn_entropy'])

    methods['0.7cos+0.3mass'] = list(0.7 * cos_n + 0.3 * mass_n)
    methods['0.5cos+0.5attn'] = list(0.5 * cos_n + 0.5 * attn_n)
    methods['best_of_both'] = list(np.maximum(cos_n, attn_n))
    methods['all_equal'] = list(0.25 * cos_n + 0.25 * mass_n + 0.25 * attn_n + 0.25 * ent_n)

    method_meta = {
        'cosine': ('Yes', 'Hidden'),
        'norm_diff': ('Yes', 'Hidden'),
        '1-mass': ('No', 'Output'),
        '1-msp': ('No', 'Output'),
        '-energy': ('No', 'Output'),
        'attn_max': ('No', 'Attention'),
        '-attn_entropy': ('No', 'Attention'),
        '0.7cos+0.3mass': ('Yes', 'Combined'),
        '0.5cos+0.5attn': ('Yes', 'Combined'),
        'best_of_both': ('Yes', 'Combined'),
        'all_equal': ('Yes', 'Combined'),
    }

    print(f"\n  {'Method':<25} {'AUROC':>8} {'Cal?':>6} {'Type':>15}", flush=True)
    print("  " + "-" * 56, flush=True)

    aurocs = {}
    for name, scores in methods.items():
        auroc = roc_auc_score(test_labels, scores)
        aurocs[name] = auroc
        cal, typ = method_meta.get(name, ('?', '?'))
        print(f"  {name:<25} {auroc:>8.3f} {cal:>6} {typ:>15}", flush=True)

    # Per-scenario
    print("\n  Per-scenario AUROC:", flush=True)
    top_methods = ['cosine', 'attn_max', '-attn_entropy', '0.7cos+0.3mass', 'best_of_both']
    header = f"    {'OOD Type':<12}" + "".join(f"{m:>18}" for m in top_methods)
    print(header, flush=True)

    id_mask = test_labels == 0
    per_scenario = {}
    for ood_type in ['noise', 'indoor', 'inverted', 'blackout']:
        mask = np.array([s == ood_type for s in test_scenarios])
        type_labels = np.concatenate([np.zeros(id_mask.sum()), np.ones(mask.sum())])
        per_scenario[ood_type] = {}
        row = f"    {ood_type:<12}"
        for m_name in top_methods:
            type_scores = np.concatenate([
                np.array(methods[m_name])[id_mask],
                np.array(methods[m_name])[mask]
            ])
            auroc = roc_auc_score(type_labels, type_scores)
            per_scenario[ood_type][m_name] = float(auroc)
            row += f"{auroc:>18.3f}"
        print(row, flush=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'comprehensive_comparison',
        'experiment_number': 66,
        'timestamp': timestamp,
        'n_cal': len(cal_data),
        'n_test': len(test_data),
        'aurocs': {k: float(v) for k, v in aurocs.items()},
        'per_scenario': per_scenario,
    }
    output_path = os.path.join(RESULTS_DIR, f"comprehensive_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
