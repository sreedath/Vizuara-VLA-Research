"""
Attention Pattern Analysis for OOD Detection on Real OpenVLA-7B.

Examines how attention patterns differ between in-distribution and OOD
inputs. Provides mechanistic insight into why cosine distance works.

Tests:
1. Attention entropy: are OOD attention patterns more diffuse or focused?
2. Image vs text token attention: do OOD inputs attend differently to image tokens?
3. Last-token attention pattern: what does the final generated token attend to?
4. Attention-based OOD detection: can attention statistics alone detect OOD?

Experiment 39 in the CalibDrive series.
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

SCENARIOS = {
    'highway': {'n': 15, 'speed': '30', 'difficulty': 'easy'},
    'urban': {'n': 15, 'speed': '15', 'difficulty': 'easy'},
    'ood_noise': {'n': 8, 'speed': '25', 'difficulty': 'ood'},
    'ood_blank': {'n': 8, 'speed': '25', 'difficulty': 'ood'},
    'ood_indoor': {'n': 8, 'speed': '25', 'difficulty': 'ood'},
    'ood_inverted': {'n': 8, 'speed': '30', 'difficulty': 'ood'},
    'ood_blackout': {'n': 8, 'speed': '25', 'difficulty': 'ood'},
}


def create_scene_image(scenario, idx, size=(256, 256)):
    np.random.seed(idx * 3900 + hash(scenario) % 39000)
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
    elif scenario == 'ood_blackout':
        img = np.full((*size, 3), 5, dtype=np.uint8)
    else:
        img = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
    noise = np.random.randint(-3, 3, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def main():
    print("=" * 70, flush=True)
    print("ATTENTION PATTERN ANALYSIS", flush=True)
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

    prompt = "In: What action should the robot take to drive forward at {speed} m/s safely?\nOut:"
    total = sum(s['n'] for s in SCENARIOS.values())
    print(f"Total samples: {total}", flush=True)

    # Calibration centroid
    cal_hidden = []
    print("\nCalibrating centroid...", flush=True)
    for scene in ['highway', 'urban']:
        for i in range(10):
            img = create_scene_image(scene, i + 100)
            p = prompt.format(speed='25')
            inputs = processor(p, img).to(model.device, dtype=torch.bfloat16)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=7, do_sample=False,
                    output_hidden_states=True, return_dict_in_generate=True,
                )
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                last_step = outputs.hidden_states[-1]
                if isinstance(last_step, tuple):
                    hidden = last_step[-1][0, -1, :].float().cpu().numpy()
                else:
                    hidden = last_step[0, -1, :].float().cpu().numpy()
            else:
                hidden = np.zeros(4096)
            cal_hidden.append(hidden)
    cal_mean = np.mean(np.array(cal_hidden), axis=0)
    cal_norm = cal_mean / (np.linalg.norm(cal_mean) + 1e-10)
    print(f"  Centroid from {len(cal_hidden)} samples.", flush=True)

    # Main analysis
    all_results = []
    sample_idx = 0

    for scenario, config in SCENARIOS.items():
        for i in range(config['n']):
            sample_idx += 1
            image = create_scene_image(scenario, i)
            p = prompt.format(speed=config['speed'])
            inputs = processor(p, image).to(model.device, dtype=torch.bfloat16)
            input_len = inputs['input_ids'].shape[1]

            with torch.no_grad():
                # Forward pass with attention output
                forward_out = model.forward(
                    **inputs,
                    output_attentions=True,
                    return_dict=True,
                )

                # Attention from last layer
                last_layer_attn = forward_out.attentions[-1][0]  # (heads, seq, seq)
                n_heads = last_layer_attn.shape[0]
                seq_len = last_layer_attn.shape[1]

                # Last token attention across heads
                last_token_attn = last_layer_attn[:, -1, :].float().cpu().numpy()
                mean_last_attn = np.mean(last_token_attn, axis=0)

                # Attention entropy per head
                attn_entropies = []
                for h in range(n_heads):
                    a = last_token_attn[h]
                    a = np.clip(a, 1e-10, 1.0)
                    ent = -np.sum(a * np.log(a))
                    attn_entropies.append(ent)

                mean_attn_entropy = float(np.mean(attn_entropies))
                max_attn_entropy = float(np.max(attn_entropies))
                std_attn_entropy = float(np.std(attn_entropies))

                # Attention concentration
                max_attn_val = float(np.max(mean_last_attn))
                top5_attn = float(np.sum(np.sort(mean_last_attn)[-5:]))

                # Image vs text attention
                n_image_tokens = max(1, seq_len - 30)
                image_attn = float(np.sum(mean_last_attn[:n_image_tokens]))
                text_attn = float(np.sum(mean_last_attn[n_image_tokens:]))

                # Generate for hidden state + action mass
                gen_out = model.generate(
                    **inputs, max_new_tokens=7, do_sample=False,
                    output_hidden_states=True, output_scores=True,
                    return_dict_in_generate=True,
                )

            # Hidden state
            if hasattr(gen_out, 'hidden_states') and gen_out.hidden_states:
                last_step = gen_out.hidden_states[-1]
                if isinstance(last_step, tuple):
                    hidden = last_step[-1][0, -1, :].float().cpu().numpy()
                else:
                    hidden = last_step[0, -1, :].float().cpu().numpy()
            else:
                hidden = np.zeros(4096)
            h_norm = hidden / (np.linalg.norm(hidden) + 1e-10)
            cos_dist = 1.0 - float(np.dot(h_norm, cal_norm))

            # Action mass
            vocab_size = gen_out.scores[0].shape[-1]
            action_start = vocab_size - 256
            dim_masses = []
            for score in gen_out.scores[:7]:
                full_logits = score[0].float()
                full_probs = torch.softmax(full_logits, dim=0).cpu().numpy()
                dim_masses.append(float(full_probs[action_start:].sum()))

            result = {
                'scenario': scenario,
                'difficulty': config['difficulty'],
                'idx': i,
                'cos_dist': cos_dist,
                'action_mass': float(np.mean(dim_masses)),
                'mean_attn_entropy': mean_attn_entropy,
                'max_attn_entropy': max_attn_entropy,
                'std_attn_entropy': std_attn_entropy,
                'max_attn_val': max_attn_val,
                'top5_attn': top5_attn,
                'image_attn_frac': image_attn,
                'text_attn_frac': text_attn,
                'seq_len': seq_len,
                'n_heads': n_heads,
            }
            all_results.append(result)

            if sample_idx % 10 == 0 or sample_idx == total:
                print(f"  [{sample_idx}/{total}] {scenario}_{i}: "
                      f"cos={cos_dist:.4f}, attn_ent={mean_attn_entropy:.2f}, "
                      f"img_attn={image_attn:.3f}", flush=True)

    # ===================================================================
    # Analysis
    # ===================================================================
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    easy = [r for r in all_results if r['difficulty'] == 'easy']
    ood = [r for r in all_results if r['difficulty'] == 'ood']

    # 1. Attention statistics comparison
    print("\n1. Attention Statistics: Easy vs OOD", flush=True)
    print("-" * 80, flush=True)
    metrics = ['mean_attn_entropy', 'max_attn_entropy', 'std_attn_entropy',
               'max_attn_val', 'top5_attn', 'image_attn_frac', 'text_attn_frac']

    for m in metrics:
        easy_vals = [r[m] for r in easy]
        ood_vals = [r[m] for r in ood]
        print(f"  {m:<22}: easy={np.mean(easy_vals):.4f}±{np.std(easy_vals):.4f}, "
              f"OOD={np.mean(ood_vals):.4f}±{np.std(ood_vals):.4f}, "
              f"Δ={np.mean(ood_vals)-np.mean(easy_vals):+.4f}", flush=True)

    # 2. Per-scenario
    print("\n2. Per-Scenario Mean Attention Entropy", flush=True)
    print("-" * 80, flush=True)
    for scenario in SCENARIOS:
        s_results = [r for r in all_results if r['scenario'] == scenario]
        mean_ent = np.mean([r['mean_attn_entropy'] for r in s_results])
        mean_img = np.mean([r['image_attn_frac'] for r in s_results])
        print(f"  {scenario:<15}: attn_ent={mean_ent:.2f}, img_frac={mean_img:.3f}", flush=True)

    # 3. AUROC
    print("\n3. Attention-Based OOD Detection AUROC", flush=True)
    print("-" * 80, flush=True)
    easy_idxs = [i for i, r in enumerate(all_results) if r['difficulty'] == 'easy']
    ood_idxs = [i for i, r in enumerate(all_results) if r['difficulty'] == 'ood']
    eval_labels = [0] * len(easy_idxs) + [1] * len(ood_idxs)

    signals = {
        'Cosine distance': [all_results[i]['cos_dist'] for i in easy_idxs + ood_idxs],
        'Attn entropy (mean)': [all_results[i]['mean_attn_entropy'] for i in easy_idxs + ood_idxs],
        'Attn entropy (max)': [all_results[i]['max_attn_entropy'] for i in easy_idxs + ood_idxs],
        'Attn entropy (std)': [all_results[i]['std_attn_entropy'] for i in easy_idxs + ood_idxs],
        'Max attn value': [-all_results[i]['max_attn_val'] for i in easy_idxs + ood_idxs],
        'Top-5 attn': [-all_results[i]['top5_attn'] for i in easy_idxs + ood_idxs],
        'Image attn frac': [-all_results[i]['image_attn_frac'] for i in easy_idxs + ood_idxs],
        'Action mass': [-all_results[i]['action_mass'] for i in easy_idxs + ood_idxs],
    }

    for name, scores in signals.items():
        auroc = roc_auc_score(eval_labels, scores)
        print(f"  {name:<22}: AUROC = {auroc:.3f}", flush=True)

    # 4. Correlation
    print("\n4. Correlation: Attention Metrics ↔ Cosine Distance", flush=True)
    print("-" * 80, flush=True)
    cos_vals = [r['cos_dist'] for r in all_results]
    for m in metrics:
        m_vals = [r[m] for r in all_results]
        r = np.corrcoef(cos_vals, m_vals)[0, 1]
        print(f"  Cosine ↔ {m:<22}: r = {r:+.3f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'attention_analysis',
        'experiment_number': 39,
        'timestamp': timestamp,
        'results': all_results,
    }

    output_path = os.path.join(RESULTS_DIR, f"attention_analysis_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
