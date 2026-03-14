"""
Calibration-Free OOD Detection via Attention Patterns.

Tests whether last-layer attention statistics alone can detect OOD
inputs WITHOUT any calibration data. This would eliminate the need
for calibration entirely.

Experiment 64 in the CalibDrive series.
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


def extract_all_signals(model, processor, image, prompt):
    """Extract attention, hidden state, and output signals."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)

    # Forward pass for attention
    with torch.no_grad():
        out = model(**inputs, output_attentions=True, output_hidden_states=True)

    result = {}

    # Attention stats from last layer
    if hasattr(out, 'attentions') and out.attentions:
        attn = out.attentions[-1][0].float().cpu().numpy()  # (heads, seq, seq)
        n_heads, seq_len, _ = attn.shape
        last_attn = attn[:, -1, :]  # (heads, seq)

        entropies = []
        max_attns = []
        for h in range(n_heads):
            a = last_attn[h] + 1e-10
            entropies.append(float(-np.sum(a * np.log(a))))
            max_attns.append(float(np.max(last_attn[h])))

        result['attn_entropy'] = float(np.mean(entropies))
        result['attn_max'] = float(np.mean(max_attns))
        result['attn_entropy_std'] = float(np.std(entropies))

    # Hidden state from last layer
    if hasattr(out, 'hidden_states') and out.hidden_states:
        hidden = out.hidden_states[-1][0, -1, :].float().cpu().numpy()
        result['hidden'] = hidden

    # Also do a generate pass for action scores
    with torch.no_grad():
        gen_out = model.generate(
            **inputs, max_new_tokens=7, do_sample=False,
            output_scores=True, output_hidden_states=True,
            return_dict_in_generate=True,
        )

    # Action mass
    if hasattr(gen_out, 'scores') and gen_out.scores:
        vocab_size = gen_out.scores[0].shape[-1]
        action_start = vocab_size - 256
        masses = []
        for score in gen_out.scores[:7]:
            probs = torch.softmax(score[0].float(), dim=0)
            masses.append(float(probs[action_start:].sum()))
        result['action_mass'] = float(np.mean(masses))

    # Hidden from generate
    if hasattr(gen_out, 'hidden_states') and gen_out.hidden_states:
        last_step = gen_out.hidden_states[-1]
        if isinstance(last_step, tuple):
            gen_hidden = last_step[-1][0, -1, :].float().cpu().numpy()
        else:
            gen_hidden = last_step[0, -1, :].float().cpu().numpy()
        result['gen_hidden'] = gen_hidden

    return result


def cosine_dist(a, b):
    return 1.0 - float(np.dot(a / (np.linalg.norm(a) + 1e-10),
                               b / (np.linalg.norm(b) + 1e-10)))


def main():
    print("=" * 70, flush=True)
    print("CALIBRATION-FREE OOD DETECTION", flush=True)
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

    # Larger test set for robust evaluation
    test_fns = {
        'highway': (create_highway, False, 15),
        'urban': (create_urban, False, 15),
        'noise': (create_noise, True, 10),
        'indoor': (create_indoor, True, 10),
        'inverted': (create_inverted, True, 10),
        'blackout': (create_blackout, True, 10),
    }

    all_data = []
    cnt = 0
    total = sum(v[2] for v in test_fns.values())
    for scene, (fn, is_ood, n) in test_fns.items():
        for i in range(n):
            cnt += 1
            signals = extract_all_signals(model, processor,
                                          Image.fromarray(fn(i + 300)), prompt)
            signals['scenario'] = scene
            signals['is_ood'] = is_ood
            all_data.append(signals)
            if cnt % 10 == 0:
                print(f"  [{cnt}/{total}] {scene}_{i}", flush=True)

    labels = [1 if d['is_ood'] else 0 for d in all_data]

    print("\n" + "=" * 70, flush=True)
    print("CALIBRATION-FREE DETECTION (no cal data needed)", flush=True)
    print("=" * 70, flush=True)

    # Calibration-free signals
    calfree_signals = {}
    if all(d.get('attn_entropy') is not None for d in all_data):
        calfree_signals['attn_entropy'] = [-d['attn_entropy'] for d in all_data]  # Lower entropy = OOD
        calfree_signals['attn_max'] = [d['attn_max'] for d in all_data]  # Higher max = OOD
    if all(d.get('action_mass') is not None for d in all_data):
        calfree_signals['1-mass'] = [1 - d['action_mass'] for d in all_data]

    print(f"\n  {'Signal':<25} {'AUROC':>8} {'ID Mean':>10} {'OOD Mean':>10}", flush=True)
    print("  " + "-" * 55, flush=True)

    calfree_aurocs = {}
    for name, scores in calfree_signals.items():
        auroc = roc_auc_score(labels, scores)
        calfree_aurocs[name] = auroc
        id_vals = [s for s, l in zip(scores, labels) if l == 0]
        ood_vals = [s for s, l in zip(scores, labels) if l == 1]
        print(f"  {name:<25} {auroc:>8.3f} {np.mean(id_vals):>10.4f} {np.mean(ood_vals):>10.4f}",
              flush=True)

    # Calibrated signals (using first 10 ID samples as calibration)
    print("\n" + "=" * 70, flush=True)
    print("CALIBRATED DETECTION (10 cal samples)", flush=True)
    print("=" * 70, flush=True)

    cal_data = [d for d in all_data if not d['is_ood']][:10]
    test_data = [d for d in all_data if d['is_ood'] or d not in cal_data]
    test_labels = [1 if d['is_ood'] else 0 for d in test_data]

    # Cosine distance
    if all(d.get('gen_hidden') is not None for d in all_data):
        centroid = np.mean([d['gen_hidden'] for d in cal_data], axis=0)
        cosine_scores = [cosine_dist(d['gen_hidden'], centroid) for d in test_data]
        cos_auroc = roc_auc_score(test_labels, cosine_scores)
        print(f"  Cosine distance: AUROC={cos_auroc:.3f}", flush=True)

    # Multi-signal fusion
    print("\n" + "=" * 70, flush=True)
    print("MULTI-SIGNAL FUSION", flush=True)
    print("=" * 70, flush=True)

    # Normalize each signal to [0,1] and combine
    if calfree_signals and all(d.get('gen_hidden') is not None for d in all_data):
        # Use all data for fusion analysis
        all_cosine = [cosine_dist(d.get('gen_hidden', np.zeros(4096)),
                                   np.mean([d2['gen_hidden'] for d2 in cal_data], axis=0))
                      for d in all_data]

        signals_for_fusion = {
            'cosine': all_cosine,
        }
        signals_for_fusion.update(calfree_signals)

        # Normalize
        normalized = {}
        for name, vals in signals_for_fusion.items():
            arr = np.array(vals)
            normalized[name] = (arr - arr.min()) / (arr.max() - arr.min() + 1e-10)

        # Try combinations
        fusion_results = {}
        combos = [
            ('cos+attn_ent', {'cosine': 0.5, 'attn_entropy': 0.5}),
            ('cos+attn_max', {'cosine': 0.5, 'attn_max': 0.5}),
            ('0.7cos+0.3ent', {'cosine': 0.7, 'attn_entropy': 0.3}),
            ('0.7cos+0.3max', {'cosine': 0.7, 'attn_max': 0.3}),
            ('cos+mass+ent', {'cosine': 0.5, '1-mass': 0.25, 'attn_entropy': 0.25}),
            ('all_equal', {'cosine': 0.25, '1-mass': 0.25, 'attn_entropy': 0.25, 'attn_max': 0.25}),
        ]

        for combo_name, weights in combos:
            fused = np.zeros(len(all_data))
            for sig_name, w in weights.items():
                if sig_name in normalized:
                    fused += w * normalized[sig_name]
            auroc = roc_auc_score(labels, fused)
            fusion_results[combo_name] = float(auroc)
            print(f"  {combo_name:<25} AUROC={auroc:.3f}", flush=True)

    # Per-scenario analysis
    print("\n  Per-scenario calibration-free detection:", flush=True)
    for scene in sorted(set(d['scenario'] for d in all_data)):
        scene_mask = [d['scenario'] == scene for d in all_data]
        id_mask = [not d['is_ood'] for d in all_data]

        id_data_scene = [d for d in all_data if not d['is_ood']]
        ood_data_scene = [d for d in all_data if d['scenario'] == scene and d['is_ood']]
        if not ood_data_scene:
            continue

        type_labels = [0]*len(id_data_scene) + [1]*len(ood_data_scene)
        for sig_name in ['attn_entropy', 'attn_max']:
            if sig_name in calfree_signals:
                id_vals = [-d['attn_entropy'] if sig_name == 'attn_entropy' else d['attn_max']
                           for d in id_data_scene]
                ood_vals = [-d['attn_entropy'] if sig_name == 'attn_entropy' else d['attn_max']
                            for d in ood_data_scene]
                auroc = roc_auc_score(type_labels, id_vals + ood_vals)
                print(f"    {scene} ({sig_name}): AUROC={auroc:.3f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'attn_calibfree',
        'experiment_number': 64,
        'timestamp': timestamp,
        'n_test': len(all_data),
        'calfree_aurocs': {k: float(v) for k, v in calfree_aurocs.items()},
        'fusion_results': fusion_results if 'fusion_results' in dir() else {},
    }
    output_path = os.path.join(RESULTS_DIR, f"attn_calibfree_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
