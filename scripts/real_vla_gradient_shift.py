"""
Gradient Distribution Shift on Real OpenVLA-7B.

Tests how uncertainty signals degrade under progressive distribution shift:
1. Start from highway (base distribution)
2. Progressively morph toward OOD noise via alpha-blending
3. Measure how confidence, entropy, and MC Dropout signals track the shift
4. Determine at what blend ratio each signal detects the shift

This reveals the detection frontier — how much shift is needed before
uncertainty signals fire.

Experiment 17 in the CalibDrive series.
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

ALPHA_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
N_SAMPLES = 15  # Per alpha level
N_MC_PASSES = 10
OPTIMAL_DROPOUT = 0.20


def create_highway_image(idx, size=(256, 256)):
    np.random.seed(idx * 1100)
    img = np.zeros((*size, 3), dtype=np.uint8)
    img[:size[0]//2] = [135, 206, 235]
    img[size[0]//2:] = [80, 80, 80]
    noise = np.random.randint(-5, 5, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def create_noise_image(idx, size=(256, 256)):
    np.random.seed(idx * 1200)
    return np.random.randint(0, 256, (*size, 3), dtype=np.uint8)


def blend_images(img1, img2, alpha):
    """Alpha-blend: result = (1-alpha)*img1 + alpha*img2."""
    return np.clip(
        (1 - alpha) * img1.astype(np.float32) + alpha * img2.astype(np.float32),
        0, 255
    ).astype(np.uint8)


def set_dropout_rate(model, rate):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = rate


def enable_mc_dropout(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


def single_forward(model, processor, image, prompt):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
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
    dim_confs = []
    dim_entropies = []
    dim_tokens = []
    for score in outputs.scores[:7]:
        logits = score[0, action_start:].float()
        probs = torch.softmax(logits, dim=0)
        dim_confs.append(probs.max().item())
        dim_entropies.append(-(probs * torch.log(probs + 1e-10)).sum().item())
        dim_tokens.append(probs.argmax().item())
    return {
        'geo_conf': float(np.exp(np.mean(np.log(np.array(dim_confs) + 1e-10)))),
        'mean_entropy': float(np.mean(dim_entropies)),
        'dim_tokens': dim_tokens,
    }


def main():
    print("=" * 70, flush=True)
    print("GRADIENT DISTRIBUTION SHIFT ON REAL OpenVLA-7B", flush=True)
    print("=" * 70, flush=True)
    print(f"Alpha levels: {ALPHA_LEVELS}", flush=True)
    print(f"Samples per alpha: {N_SAMPLES}", flush=True)
    print(f"MC passes: {N_MC_PASSES}", flush=True)
    print(flush=True)

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
    set_dropout_rate(model, OPTIMAL_DROPOUT)
    enable_mc_dropout(model)
    print("Model loaded.", flush=True)

    prompt = "In: What action should the robot take to drive forward at 30 m/s safely?\nOut:"
    total_inferences = len(ALPHA_LEVELS) * N_SAMPLES * N_MC_PASSES
    print(f"Total inferences: {total_inferences}", flush=True)
    print(flush=True)

    # Pre-generate base images
    highway_imgs = [create_highway_image(i) for i in range(N_SAMPLES)]
    noise_imgs = [create_noise_image(i) for i in range(N_SAMPLES)]

    # Get baseline tokens for highway (alpha=0)
    print("Getting baseline highway tokens...", flush=True)
    baseline_tokens = []
    for i in range(N_SAMPLES):
        img = Image.fromarray(highway_imgs[i])
        r = single_forward(model, processor, img, prompt)
        baseline_tokens.append(r['dim_tokens'])
    print("Baseline done.", flush=True)

    all_results = {}

    for alpha in ALPHA_LEVELS:
        print(f"\n  Alpha = {alpha:.1f}", flush=True)
        alpha_results = []

        for i in range(N_SAMPLES):
            blended = blend_images(highway_imgs[i], noise_imgs[i], alpha)
            img = Image.fromarray(blended)

            t0 = time.time()
            mc_results = []
            for mc in range(N_MC_PASSES):
                r = single_forward(model, processor, img, prompt)
                mc_results.append(r)
            elapsed = time.time() - t0

            confs = [r['geo_conf'] for r in mc_results]
            entropies = [r['mean_entropy'] for r in mc_results]

            # Token agreement with baseline
            first_tokens = mc_results[0]['dim_tokens']
            base_agree = sum(1 for a, b in zip(baseline_tokens[i], first_tokens) if a == b) / 7

            # MC token agreement
            tokens_arr = np.array([r['dim_tokens'] for r in mc_results])
            mc_token_agree = np.mean([len(set(tokens_arr[:, d])) == 1 for d in range(7)])

            alpha_results.append({
                'idx': i,
                'alpha': alpha,
                'conf_mean': float(np.mean(confs)),
                'conf_std': float(np.std(confs)),
                'entropy_mean': float(np.mean(entropies)),
                'entropy_std': float(np.std(entropies)),
                'baseline_token_agree': float(base_agree),
                'mc_token_agree': float(mc_token_agree),
            })

            if i % 5 == 0 or i == N_SAMPLES - 1:
                print(f"    [{i+1}/{N_SAMPLES}] α={alpha:.1f}: "
                      f"ent={np.mean(entropies):.3f}, conf={np.mean(confs):.3f}, "
                      f"base_agree={base_agree:.2f} ({elapsed:.1f}s)", flush=True)

        all_results[str(alpha)] = alpha_results

    # ===================================================================
    # Analysis
    # ===================================================================
    print("\n" + "=" * 70, flush=True)
    print("GRADIENT SHIFT ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    print("\n1. Signal Evolution vs Alpha (Highway→Noise Blend)", flush=True)
    print("-" * 80, flush=True)
    print(f"{'Alpha':>6} {'Entropy':>8} {'±':>6} {'Conf':>7} {'±':>6} "
          f"{'MCStd':>7} {'BaseAgree':>10} {'MCAgree':>8}", flush=True)
    print("-" * 80, flush=True)

    for alpha in ALPHA_LEVELS:
        samples = all_results[str(alpha)]
        ents = [s['entropy_mean'] for s in samples]
        confs = [s['conf_mean'] for s in samples]
        mc_stds = [s['conf_std'] for s in samples]
        base_agrees = [s['baseline_token_agree'] for s in samples]
        mc_agrees = [s['mc_token_agree'] for s in samples]

        print(f"{alpha:>6.1f} {np.mean(ents):>8.3f} {np.std(ents):>6.3f} "
              f"{np.mean(confs):>7.3f} {np.std(confs):>6.3f} "
              f"{np.mean(mc_stds):>7.4f} {np.mean(base_agrees):>10.3f} "
              f"{np.mean(mc_agrees):>8.3f}", flush=True)

    # 2. Detection threshold analysis
    print("\n2. At what alpha does each signal detect the shift?", flush=True)
    print("-" * 60, flush=True)

    base_ent = np.mean([s['entropy_mean'] for s in all_results['0.0']])
    base_conf = np.mean([s['conf_mean'] for s in all_results['0.0']])

    for alpha in ALPHA_LEVELS[1:]:
        samples = all_results[str(alpha)]
        ent = np.mean([s['entropy_mean'] for s in samples])
        conf = np.mean([s['conf_mean'] for s in samples])
        ent_shift = (ent - base_ent) / base_ent * 100
        conf_shift = (conf - base_conf) / base_conf * 100
        base_agree = np.mean([s['baseline_token_agree'] for s in samples])

        print(f"  α={alpha:.1f}: entropy shift={ent_shift:+.1f}%, "
              f"conf shift={conf_shift:+.1f}%, "
              f"token agree={base_agree:.3f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'timestamp': timestamp,
        'config': {
            'alpha_levels': ALPHA_LEVELS,
            'n_samples': N_SAMPLES,
            'n_mc_passes': N_MC_PASSES,
            'dropout_rate': OPTIMAL_DROPOUT,
        },
        'per_alpha': {
            str(alpha): samples
            for alpha, samples in all_results.items()
        },
    }

    output_path = os.path.join(RESULTS_DIR, f"gradient_shift_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
