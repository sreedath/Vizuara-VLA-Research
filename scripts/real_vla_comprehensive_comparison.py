"""
Comprehensive Method Comparison on Real OpenVLA-7B.

Produces the main results table for the paper: a clean head-to-head
comparison of ALL uncertainty/OOD detection methods evaluated on the
same samples with the same evaluation protocol.

Methods:
1. Action Mass (single pass, no calibration)
2. Entropy (single pass, no calibration)
3. MC Dropout Entropy (N=10, p=0.20)
4. Cosine Distance (single pass, 25 cal samples)
5. Cosine + Entropy combined
6. kNN Distance (k=3, 25 cal samples)

Evaluation:
- Proper 50/50 train/test split of easy samples
- Per-OOD-type and overall AUROC
- Bootstrap CIs (10 splits)
- Computational cost comparison

Experiment 32 in the CalibDrive series.
"""
import os
import json
import time
import datetime
import numpy as np
import torch
from PIL import Image, ImageDraw

RESULTS_DIR = "/workspace/Vizuara-VLA-Research/experiments"
os.makedirs(RESULTS_DIR, exist_ok=True)

SCENARIOS = {
    'highway': {'n': 30, 'speed': '30', 'difficulty': 'easy'},
    'urban': {'n': 30, 'speed': '15', 'difficulty': 'easy'},
    'ood_noise': {'n': 15, 'speed': '25', 'difficulty': 'ood'},
    'ood_blank': {'n': 15, 'speed': '25', 'difficulty': 'ood'},
    'ood_indoor': {'n': 15, 'speed': '25', 'difficulty': 'ood'},
    'ood_inverted': {'n': 15, 'speed': '30', 'difficulty': 'ood'},
    'ood_checker': {'n': 15, 'speed': '25', 'difficulty': 'ood'},
    'ood_blackout': {'n': 15, 'speed': '25', 'difficulty': 'ood'},
}


def create_scene_image(scenario, idx, size=(256, 256)):
    np.random.seed(idx * 3200 + hash(scenario) % 32000)
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
    elif scenario == 'ood_indoor':
        img = np.zeros((*size, 3), dtype=np.uint8)
        img[:size[0]//3] = [210, 180, 140]
        img[size[0]//3:2*size[0]//3] = [180, 120, 80]
        img[2*size[0]//3:] = [100, 70, 50]
    elif scenario == 'ood_inverted':
        img = np.zeros((*size, 3), dtype=np.uint8)
        img[:size[0]//2] = [135, 206, 235]
        img[size[0]//2:] = [80, 80, 80]
        img = 255 - img
    elif scenario == 'ood_checker':
        img = np.zeros((*size, 3), dtype=np.uint8)
        block = 32
        for y in range(0, size[0], block):
            for x in range(0, size[1], block):
                if (y // block + x // block) % 2 == 0:
                    img[y:y+block, x:x+block] = [255, 255, 255]
    elif scenario == 'ood_blackout':
        img = np.full((*size, 3), 5, dtype=np.uint8)
    else:
        img = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
    noise = np.random.randint(-3, 3, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def compute_auroc(pos_scores, neg_scores):
    n_correct = sum(1 for p in pos_scores for n in neg_scores if p > n)
    n_ties = sum(0.5 for p in pos_scores for n in neg_scores if p == n)
    n_total = len(pos_scores) * len(neg_scores)
    return (n_correct + n_ties) / n_total if n_total > 0 else 0.5


def enable_dropout(model, p=0.20):
    """Enable dropout in eval mode."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = p
            module.train()


def disable_dropout(model):
    """Disable dropout."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
            module.eval()


def main():
    print("=" * 70, flush=True)
    print("COMPREHENSIVE METHOD COMPARISON ON REAL OpenVLA-7B", flush=True)
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

    prompt = "In: What action should the robot take to drive forward at {speed} m/s safely?\nOut:"

    # ===================================================================
    # Phase 1: Single-pass inference (action mass, entropy, hidden states)
    # ===================================================================
    print("\n--- Phase 1: Single-pass inference ---", flush=True)
    all_samples = []
    all_hidden = []
    sample_idx = 0

    for scenario, config in SCENARIOS.items():
        for i in range(config['n']):
            sample_idx += 1
            image = create_scene_image(scenario, i)
            p = prompt.format(speed=config['speed'])
            inputs = processor(p, image).to(model.device, dtype=torch.bfloat16)

            t0 = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=7,
                    do_sample=False,
                    output_scores=True,
                    output_hidden_states=True,
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

            # Action mass and entropy
            vocab_size = outputs.scores[0].shape[-1]
            action_start = vocab_size - 256
            dim_masses = []
            dim_entropies = []
            for score in outputs.scores[:7]:
                full_logits = score[0].float()
                full_probs = torch.softmax(full_logits, dim=0).cpu().numpy()
                action_probs = full_probs[action_start:]
                dim_masses.append(float(action_probs.sum()))
                action_norm = action_probs / (action_probs.sum() + 1e-10)
                dim_entropies.append(float(-(action_norm * np.log(action_norm + 1e-10)).sum()))

            elapsed = time.time() - t0

            sample = {
                'scenario': scenario,
                'difficulty': config['difficulty'],
                'idx': i,
                'action_mass': float(np.mean(dim_masses)),
                'entropy': float(np.mean(dim_entropies)),
            }
            all_samples.append(sample)
            all_hidden.append(hidden)

            if sample_idx % 20 == 0 or sample_idx == total:
                print(f"  [{sample_idx}/{total}] {scenario}_{i}: "
                      f"mass={sample['action_mass']:.4f}, ent={sample['entropy']:.3f} "
                      f"({elapsed:.1f}s)", flush=True)

    # ===================================================================
    # Phase 2: MC Dropout (N=10, p=0.20)
    # ===================================================================
    print("\n--- Phase 2: MC Dropout (N=10, p=0.20) ---", flush=True)
    N_MC = 10
    enable_dropout(model, p=0.20)

    mc_entropies = []
    sample_idx = 0

    for scenario, config in SCENARIOS.items():
        for i in range(config['n']):
            sample_idx += 1
            image = create_scene_image(scenario, i)
            p = prompt.format(speed=config['speed'])
            inputs = processor(p, image).to(model.device, dtype=torch.bfloat16)

            mc_ents = []
            for mc_i in range(N_MC):
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
                dim_ents = []
                for score in outputs.scores[:7]:
                    full_logits = score[0].float()
                    full_probs = torch.softmax(full_logits, dim=0).cpu().numpy()
                    action_probs = full_probs[action_start:]
                    action_norm = action_probs / (action_probs.sum() + 1e-10)
                    dim_ents.append(float(-(action_norm * np.log(action_norm + 1e-10)).sum()))
                mc_ents.append(float(np.mean(dim_ents)))

            mc_entropy_mean = float(np.mean(mc_ents))
            mc_entropy_std = float(np.std(mc_ents))
            mc_entropies.append(mc_entropy_mean)
            all_samples[sample_idx - 1]['mc_entropy'] = mc_entropy_mean
            all_samples[sample_idx - 1]['mc_entropy_std'] = mc_entropy_std

            if sample_idx % 20 == 0 or sample_idx == total:
                print(f"  [{sample_idx}/{total}] {scenario}_{i}: "
                      f"mc_ent={mc_entropy_mean:.3f} ± {mc_entropy_std:.3f}", flush=True)

    disable_dropout(model)

    # ===================================================================
    # Analysis
    # ===================================================================
    print("\n" + "=" * 70, flush=True)
    print("COMPREHENSIVE COMPARISON", flush=True)
    print("=" * 70, flush=True)

    hidden_arr = np.array(all_hidden)
    easy_idxs = [i for i, s in enumerate(all_samples) if s['difficulty'] == 'easy']
    ood_idxs = [i for i, s in enumerate(all_samples) if s['difficulty'] == 'ood']

    # Bootstrap evaluation
    n_bootstrap = 10
    method_aurocs = {}
    ood_types = [s for s in SCENARIOS if s.startswith('ood_')]

    for boot_i in range(n_bootstrap):
        np.random.seed(boot_i * 42)
        shuffled = np.random.permutation(easy_idxs)
        cal_idxs = shuffled[:len(shuffled)//2]
        test_easy_idxs = shuffled[len(shuffled)//2:]

        # Compute calibration statistics
        cal_mean = np.mean(hidden_arr[cal_idxs], axis=0)
        cal_norm = cal_mean / (np.linalg.norm(cal_mean) + 1e-10)
        cal_hidden = hidden_arr[cal_idxs]

        # Compute all scores
        for s_idx in range(len(all_samples)):
            h = hidden_arr[s_idx]
            h_norm = h / (np.linalg.norm(h) + 1e-10)
            all_samples[s_idx][f'cos_dist_{boot_i}'] = 1.0 - float(np.dot(h_norm, cal_norm))

            # kNN (k=3)
            dists = np.linalg.norm(cal_hidden - h, axis=1)
            all_samples[s_idx][f'knn3_{boot_i}'] = float(np.mean(np.sort(dists)[:3]))

        methods = {
            'Action Mass': lambda i: -all_samples[i]['action_mass'],
            'Entropy': lambda i: all_samples[i]['entropy'],
            'MC Entropy': lambda i: all_samples[i]['mc_entropy'],
            'Cosine Dist': lambda i: all_samples[i][f'cos_dist_{boot_i}'],
            'kNN (k=3)': lambda i: all_samples[i][f'knn3_{boot_i}'],
        }

        # Also add cosine + entropy combined
        # Normalize for combination
        cos_vals = [all_samples[i][f'cos_dist_{boot_i}'] for i in range(len(all_samples))]
        ent_vals = [all_samples[i]['entropy'] for i in range(len(all_samples))]
        c_min, c_max = min(cos_vals), max(cos_vals)
        e_min, e_max = min(ent_vals), max(ent_vals)

        for s_idx in range(len(all_samples)):
            c_norm = (all_samples[s_idx][f'cos_dist_{boot_i}'] - c_min) / (c_max - c_min + 1e-10)
            e_norm = (all_samples[s_idx]['entropy'] - e_min) / (e_max - e_min + 1e-10)
            all_samples[s_idx][f'cos_ent_{boot_i}'] = 0.5 * c_norm + 0.5 * e_norm

        methods['Cos + Ent'] = lambda i: all_samples[i][f'cos_ent_{boot_i}']

        for method_name, score_fn in methods.items():
            test_easy_scores = [score_fn(i) for i in test_easy_idxs]
            ood_scores = [score_fn(i) for i in ood_idxs]
            overall_auroc = compute_auroc(ood_scores, test_easy_scores)

            if method_name not in method_aurocs:
                method_aurocs[method_name] = {'overall': [], 'per_type': {t: [] for t in ood_types}}

            method_aurocs[method_name]['overall'].append(overall_auroc)

            for ood_type in ood_types:
                type_idxs = [i for i in ood_idxs if all_samples[i]['scenario'] == ood_type]
                type_scores = [score_fn(i) for i in type_idxs]
                type_auroc = compute_auroc(type_scores, test_easy_scores)
                method_aurocs[method_name]['per_type'][ood_type].append(type_auroc)

    # Print main results table
    print("\n=== MAIN RESULTS TABLE ===", flush=True)
    print("-" * 100, flush=True)

    header = f"{'Method':<15} {'Passes':>6} {'Cal':>4} | {'Overall':>8}"
    for ood_type in ood_types:
        short_name = ood_type.replace('ood_', '')[:6]
        header += f" {short_name:>8}"
    print(header, flush=True)
    print("-" * 100, flush=True)

    method_meta = {
        'Action Mass': (1, 'No'),
        'Entropy': (1, 'No'),
        'MC Entropy': (10, 'No'),
        'Cosine Dist': (1, 'Yes'),
        'kNN (k=3)': (1, 'Yes'),
        'Cos + Ent': (1, 'Yes'),
    }

    for method_name in ['Cosine Dist', 'Cos + Ent', 'kNN (k=3)', 'MC Entropy', 'Action Mass', 'Entropy']:
        passes, cal = method_meta[method_name]
        mean_overall = np.mean(method_aurocs[method_name]['overall'])
        std_overall = np.std(method_aurocs[method_name]['overall'])

        row = f"{method_name:<15} {passes:>6} {cal:>4} | {mean_overall:>5.3f}±{std_overall:.3f}"

        for ood_type in ood_types:
            mean_type = np.mean(method_aurocs[method_name]['per_type'][ood_type])
            row += f" {mean_type:>8.3f}"

        print(row, flush=True)

    print("-" * 100, flush=True)

    # Computational cost comparison
    print("\n=== COMPUTATIONAL COST ===", flush=True)
    print(f"{'Method':<15} {'Passes':>6} {'Cal Data':>8} {'Storage':>8} {'Cost/sample':>11}", flush=True)
    print("-" * 55, flush=True)
    costs = [
        ('Cosine Dist', 1, '25 imgs', '16 KB', '1 dot prod'),
        ('Cos + Ent', 1, '25 imgs', '16 KB', '1 dot + ent'),
        ('kNN (k=3)', 1, '25 imgs', '400 KB', '25 L2 dists'),
        ('MC Entropy', 10, 'None', '0', '10 passes'),
        ('Action Mass', 1, 'None', '0', 'Free'),
        ('Entropy', 1, 'None', '0', 'Free'),
    ]
    for name, passes, cal, storage, cost in costs:
        print(f"{name:<15} {passes:>6} {cal:>8} {storage:>8} {cost:>11}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_summary = {}
    for method_name in method_aurocs:
        results_summary[method_name] = {
            'mean': float(np.mean(method_aurocs[method_name]['overall'])),
            'std': float(np.std(method_aurocs[method_name]['overall'])),
            'per_type': {t: float(np.mean(v)) for t, v in
                        method_aurocs[method_name]['per_type'].items()},
        }

    output = {
        'timestamp': timestamp,
        'n_bootstrap': n_bootstrap,
        'n_mc': N_MC,
        'results': results_summary,
        'samples': [{k: v for k, v in s.items()
                    if not k.startswith('cos_dist_') and not k.startswith('knn3_')
                    and not k.startswith('cos_ent_')} for s in all_samples],
    }

    output_path = os.path.join(RESULTS_DIR, f"comprehensive_comparison_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
