"""
Attention Pattern Analysis on Real OpenVLA-7B.

Analyzes attention patterns during action token generation:
1. Attention entropy per head and layer
2. Image vs text attention ratio
3. Per-scenario attention differences
4. Attention-uncertainty correlation
5. Which image patches receive most attention across scenarios

Experiment 13 in the CalibDrive series.
"""
import os
import json
import time
import datetime
import numpy as np
import torch
from PIL import Image

RESULTS_DIR = "/workspace/Vizuara-VLA-Research/experiments"
os.makedirs(RESULTS_DIR, exist_ok=True)

SCENARIOS = {
    'highway': {'n': 15, 'speed': '30', 'difficulty': 'easy'},
    'urban': {'n': 15, 'speed': '15', 'difficulty': 'easy'},
    'night': {'n': 10, 'speed': '25', 'difficulty': 'hard'},
    'rain': {'n': 10, 'speed': '20', 'difficulty': 'hard'},
    'ood_noise': {'n': 15, 'speed': '25', 'difficulty': 'ood'},
    'ood_blank': {'n': 15, 'speed': '25', 'difficulty': 'ood'},
}


def create_scene_image(scenario, idx, size=(256, 256)):
    np.random.seed(idx * 700 + hash(scenario) % 7000)
    if scenario == 'highway':
        img = np.zeros((*size, 3), dtype=np.uint8)
        img[:size[0]//2] = [135, 206, 235]
        img[size[0]//2:] = [80, 80, 80]
    elif scenario == 'urban':
        img = np.zeros((*size, 3), dtype=np.uint8)
        img[:size[0]//3] = [135, 206, 235]
        img[size[0]//3:size[0]//2] = [139, 119, 101]
        img[size[0]//2:] = [80, 80, 80]
    elif scenario == 'night':
        img = np.full((*size, 3), 15, dtype=np.uint8)
        img[size[0]//2:] = [30, 30, 35]
    elif scenario == 'rain':
        img = np.zeros((*size, 3), dtype=np.uint8)
        img[:size[0]//2] = [100, 100, 110]
        img[size[0]//2:] = [60, 60, 65]
    elif scenario == 'ood_noise':
        img = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
    elif scenario == 'ood_blank':
        img = np.full((*size, 3), 128, dtype=np.uint8)
    else:
        img = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)

    noise = np.random.randint(-10, 10, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def main():
    print("=" * 70, flush=True)
    print("ATTENTION PATTERN ANALYSIS ON REAL OpenVLA-7B", flush=True)
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

    # Determine architecture details
    n_layers = model.config.num_hidden_layers if hasattr(model.config, 'num_hidden_layers') else 32
    n_heads = model.config.num_attention_heads if hasattr(model.config, 'num_attention_heads') else 32
    print(f"Layers: {n_layers}, Heads: {n_heads}", flush=True)

    # Pre-generate images
    images = {}
    prompts = {}
    for scenario, config in SCENARIOS.items():
        for i in range(config['n']):
            key = f"{scenario}_{i}"
            images[key] = create_scene_image(scenario, i)
            prompts[key] = f"In: What action should the robot take to drive forward at {config['speed']} m/s safely?\nOut:"

    total = sum(s['n'] for s in SCENARIOS.values())
    print(f"Total samples: {total}", flush=True)
    print(flush=True)

    # We'll sample a few layers for attention analysis (every 8th + first + last)
    layers_to_analyze = sorted(set([0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]))
    print(f"Analyzing layers: {layers_to_analyze}", flush=True)

    all_results = {}
    sample_idx = 0

    for scenario, config in SCENARIOS.items():
        scenario_results = []

        for i in range(config['n']):
            sample_idx += 1
            key = f"{scenario}_{i}"
            image = images[key]
            prompt = prompts[key]

            t0 = time.time()

            # Forward pass with attention outputs
            inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
            with torch.no_grad():
                # First get the encoding to count token types
                input_ids = inputs['input_ids']
                n_tokens = input_ids.shape[1]

                # Run forward pass with attention
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=7,
                    do_sample=False,
                    output_scores=True,
                    output_attentions=True,
                    return_dict_in_generate=True,
                )

            elapsed = time.time() - t0

            # Extract action logit metrics
            vocab_size = outputs.scores[0].shape[-1]
            action_start = vocab_size - 256
            dim_confs = []
            dim_entropies = []
            for score in outputs.scores[:7]:
                logits = score[0, action_start:].float()
                probs = torch.softmax(logits, dim=0)
                dim_confs.append(probs.max().item())
                dim_entropies.append(-(probs * torch.log(probs + 1e-10)).sum().item())

            geo_conf = np.exp(np.mean(np.log(np.array(dim_confs) + 1e-10)))
            mean_ent = np.mean(dim_entropies)

            # Extract attention patterns from sampled layers
            # outputs.attentions is a tuple of tuples: (step, layer) -> (batch, heads, seq, seq)
            attn_stats = {}

            # We analyze the FIRST generation step's attentions
            if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                first_step_attns = outputs.attentions[0]  # first generated token

                for layer_idx in layers_to_analyze:
                    if layer_idx < len(first_step_attns):
                        attn = first_step_attns[layer_idx]  # (batch, heads, seq_out, seq_in)
                        # We want the last token's attention (the generated token)
                        # attn shape: (1, n_heads, 1, seq_len) for the generated token
                        if attn.dim() == 4:
                            last_attn = attn[0, :, -1, :]  # (n_heads, seq_len)
                        else:
                            last_attn = attn[0, :, :]  # fallback

                        last_attn = last_attn.float().cpu().numpy()
                        seq_len = last_attn.shape[-1]

                        # Attention entropy per head
                        head_entropies = []
                        for h in range(last_attn.shape[0]):
                            p = last_attn[h] + 1e-10
                            p = p / p.sum()
                            ent = -(p * np.log(p)).sum()
                            head_entropies.append(float(ent))

                        # Average attention pattern across heads
                        avg_attn = last_attn.mean(axis=0)  # (seq_len,)

                        # Attention concentration: what fraction of attention goes to top-10 tokens
                        sorted_attn = np.sort(avg_attn)[::-1]
                        top10_attn_mass = float(sorted_attn[:10].sum())
                        top50_attn_mass = float(sorted_attn[:50].sum())

                        # Attention entropy (averaged across heads)
                        avg_entropy = float(np.mean(head_entropies))
                        max_entropy = float(np.max(head_entropies))
                        min_entropy = float(np.min(head_entropies))

                        attn_stats[f"layer_{layer_idx}"] = {
                            'avg_entropy': avg_entropy,
                            'max_entropy': max_entropy,
                            'min_entropy': min_entropy,
                            'entropy_std': float(np.std(head_entropies)),
                            'top10_mass': top10_attn_mass,
                            'top50_mass': top50_attn_mass,
                            'seq_len': int(seq_len),
                        }

            # Clear GPU memory
            del outputs
            torch.cuda.empty_cache()

            sample_result = {
                'scenario': scenario,
                'difficulty': config['difficulty'],
                'idx': i,
                'geo_conf': float(geo_conf),
                'mean_entropy': float(mean_ent),
                'attention': attn_stats,
                'elapsed': float(elapsed),
            }
            scenario_results.append(sample_result)

            if i % 5 == 0 or i == config['n'] - 1:
                # Summarize attention
                if attn_stats:
                    mid_layer = f"layer_{layers_to_analyze[len(layers_to_analyze)//2]}"
                    if mid_layer in attn_stats:
                        a = attn_stats[mid_layer]
                        print(f"  [{sample_idx}/{total}] {key}: "
                              f"conf={geo_conf:.3f}, ent={mean_ent:.3f}, "
                              f"attn_ent={a['avg_entropy']:.2f}, "
                              f"top10={a['top10_mass']:.3f} ({elapsed:.1f}s)", flush=True)
                    else:
                        print(f"  [{sample_idx}/{total}] {key}: "
                              f"conf={geo_conf:.3f}, ent={mean_ent:.3f} ({elapsed:.1f}s)", flush=True)
                else:
                    print(f"  [{sample_idx}/{total}] {key}: "
                          f"conf={geo_conf:.3f}, ent={mean_ent:.3f}, "
                          f"no attention ({elapsed:.1f}s)", flush=True)

        all_results[scenario] = scenario_results

    # ===================================================================
    # Analysis
    # ===================================================================
    print("\n" + "=" * 70, flush=True)
    print("ATTENTION PATTERN ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    # 1. Per-scenario attention statistics
    print("\n1. Per-Scenario Attention Statistics (averaged over layers)", flush=True)
    print("-" * 80, flush=True)
    print(f"{'Scenario':<15} {'Conf':>6} {'ActEnt':>7} {'AttnEnt':>8} {'Top10%':>7} {'Top50%':>7}", flush=True)
    print("-" * 80, flush=True)

    scenario_attn_data = {}
    for scenario in SCENARIOS:
        confs = []
        act_ents = []
        attn_ents = []
        top10s = []
        top50s = []

        for s in all_results[scenario]:
            confs.append(s['geo_conf'])
            act_ents.append(s['mean_entropy'])

            if s['attention']:
                layer_ents = [v['avg_entropy'] for v in s['attention'].values()]
                layer_top10 = [v['top10_mass'] for v in s['attention'].values()]
                layer_top50 = [v['top50_mass'] for v in s['attention'].values()]
                attn_ents.append(np.mean(layer_ents))
                top10s.append(np.mean(layer_top10))
                top50s.append(np.mean(layer_top50))

        scenario_attn_data[scenario] = {
            'conf': np.mean(confs),
            'act_ent': np.mean(act_ents),
            'attn_ent': np.mean(attn_ents) if attn_ents else 0,
            'top10': np.mean(top10s) if top10s else 0,
            'top50': np.mean(top50s) if top50s else 0,
        }

        print(f"{scenario:<15} {np.mean(confs):>6.3f} {np.mean(act_ents):>7.3f} "
              f"{np.mean(attn_ents) if attn_ents else 0:>8.3f} "
              f"{np.mean(top10s) if top10s else 0:>7.3f} "
              f"{np.mean(top50s) if top50s else 0:>7.3f}", flush=True)

    # 2. Per-layer attention entropy
    print("\n2. Attention Entropy by Layer (averaged over scenarios)", flush=True)
    print("-" * 60, flush=True)

    for layer_idx in layers_to_analyze:
        layer_key = f"layer_{layer_idx}"
        ents = []
        for scenario in SCENARIOS:
            for s in all_results[scenario]:
                if layer_key in s.get('attention', {}):
                    ents.append(s['attention'][layer_key]['avg_entropy'])
        if ents:
            print(f"  Layer {layer_idx:>3}: entropy={np.mean(ents):.3f} ± {np.std(ents):.3f}", flush=True)

    # 3. Attention-Uncertainty Correlation
    print("\n3. Attention-Uncertainty Correlation", flush=True)
    print("-" * 60, flush=True)

    all_act_ents = []
    all_attn_ents = []
    all_confs = []
    all_top10s = []

    for scenario in SCENARIOS:
        for s in all_results[scenario]:
            if s['attention']:
                all_act_ents.append(s['mean_entropy'])
                all_confs.append(s['geo_conf'])
                layer_ents = [v['avg_entropy'] for v in s['attention'].values()]
                layer_top10 = [v['top10_mass'] for v in s['attention'].values()]
                all_attn_ents.append(np.mean(layer_ents))
                all_top10s.append(np.mean(layer_top10))

    if len(all_act_ents) > 5:
        from scipy.stats import pearsonr, spearmanr
        r_act_attn, p_act_attn = pearsonr(all_act_ents, all_attn_ents)
        r_conf_attn, p_conf_attn = pearsonr(all_confs, all_attn_ents)
        r_conf_top10, p_conf_top10 = pearsonr(all_confs, all_top10s)
        rho_act_attn, _ = spearmanr(all_act_ents, all_attn_ents)

        print(f"  Action Entropy vs Attn Entropy: r={r_act_attn:.3f}, p={p_act_attn:.4f}", flush=True)
        print(f"  Confidence vs Attn Entropy:     r={r_conf_attn:.3f}, p={p_conf_attn:.4f}", flush=True)
        print(f"  Confidence vs Top-10 Mass:      r={r_conf_top10:.3f}, p={p_conf_top10:.4f}", flush=True)
        print(f"  Action Entropy vs Attn Entropy (Spearman): rho={rho_act_attn:.3f}", flush=True)

    # 4. AUROC using attention entropy as uncertainty signal
    print("\n4. AUROC Using Attention Entropy", flush=True)
    print("-" * 60, flush=True)

    easy_attn_ent = []
    ood_attn_ent = []
    hard_attn_ent = []
    easy_top10 = []
    ood_top10 = []

    for scenario in SCENARIOS:
        for s in all_results[scenario]:
            if s['attention']:
                mean_ae = np.mean([v['avg_entropy'] for v in s['attention'].values()])
                mean_t10 = np.mean([v['top10_mass'] for v in s['attention'].values()])
                if s['difficulty'] == 'easy':
                    easy_attn_ent.append(mean_ae)
                    easy_top10.append(mean_t10)
                elif s['difficulty'] == 'ood':
                    ood_attn_ent.append(mean_ae)
                    ood_top10.append(mean_t10)
                elif s['difficulty'] == 'hard':
                    hard_attn_ent.append(mean_ae)

    if easy_attn_ent and ood_attn_ent:
        # Higher attention entropy for OOD = positive signal
        n_correct = sum(1 for e in easy_attn_ent for o in ood_attn_ent if o > e)
        n_ties = sum(0.5 for e in easy_attn_ent for o in ood_attn_ent if o == e)
        n_total = len(easy_attn_ent) * len(ood_attn_ent)
        auroc_attn = (n_correct + n_ties) / n_total if n_total > 0 else 0.5
        print(f"  AUROC(attn entropy, easy vs ood) = {auroc_attn:.3f}", flush=True)

        # Lower top-10 mass for OOD = diffuse attention
        n_correct2 = sum(1 for e in easy_top10 for o in ood_top10 if e > o)
        n_ties2 = sum(0.5 for e in easy_top10 for o in ood_top10 if e == o)
        auroc_top10 = (n_correct2 + n_ties2) / n_total if n_total > 0 else 0.5
        print(f"  AUROC(neg top10 mass, easy vs ood) = {auroc_top10:.3f}", flush=True)

    if easy_attn_ent and hard_attn_ent:
        n_correct3 = sum(1 for e in easy_attn_ent for h in hard_attn_ent if h > e)
        n_ties3 = sum(0.5 for e in easy_attn_ent for h in hard_attn_ent if h == e)
        n_total3 = len(easy_attn_ent) * len(hard_attn_ent)
        auroc_hard = (n_correct3 + n_ties3) / n_total3 if n_total3 > 0 else 0.5
        print(f"  AUROC(attn entropy, easy vs hard) = {auroc_hard:.3f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'timestamp': timestamp,
        'layers_analyzed': layers_to_analyze,
        'scenario_summary': scenario_attn_data,
        'per_sample': {
            scenario: [
                {k: v for k, v in s.items()}
                for s in samples
            ]
            for scenario, samples in all_results.items()
        },
    }

    output_path = os.path.join(RESULTS_DIR, f"attention_analysis_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
