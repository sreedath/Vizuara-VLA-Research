"""
Gradual Distribution Shift Analysis on Real OpenVLA-7B.

Tests whether cosine distance provides a continuous, calibrated uncertainty
signal as images gradually transition from in-distribution to OOD.

Corruption types tested at severity levels 0%, 25%, 50%, 75%, 100%:
1. Gaussian noise injection
2. Brightness reduction (toward blackout)
3. Color inversion (partial)
4. Gaussian blur
5. Occlusion (random rectangles)

Key questions:
- Does cosine distance increase monotonically with corruption severity?
- Is there a "phase transition" or is the increase smooth?
- How does action mass compare for detecting gradual shifts?
- At what severity does cosine distance cross the conformal threshold?

Experiment 34 in the CalibDrive series.
"""
import os
import json
import time
import datetime
import numpy as np
import torch
from PIL import Image, ImageFilter

RESULTS_DIR = "/workspace/Vizuara-VLA-Research/experiments"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Base scenarios (in-distribution)
BASE_SCENES = ['highway', 'urban']
# Corruption types
CORRUPTIONS = ['noise', 'darken', 'invert', 'blur', 'occlude']
# Severity levels (0.0 = clean, 1.0 = fully corrupted)
SEVERITIES = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]
# Samples per (base_scene, corruption, severity)
N_PER = 5


def create_base_image(scene, idx, size=(256, 256)):
    """Create a clean in-distribution driving image."""
    np.random.seed(idx * 3400 + hash(scene) % 34000)
    if scene == 'highway':
        img = np.zeros((*size, 3), dtype=np.uint8)
        img[:size[0]//2] = [135, 206, 235]  # sky
        img[size[0]//2:] = [80, 80, 80]     # road
    elif scene == 'urban':
        img = np.zeros((*size, 3), dtype=np.uint8)
        img[:size[0]//3] = [135, 206, 235]        # sky
        img[size[0]//3:size[0]//2] = [139, 119, 101]  # buildings
        img[size[0]//2:] = [80, 80, 80]           # road
    else:
        img = np.zeros((*size, 3), dtype=np.uint8)
        img[:size[0]//2] = [135, 206, 235]
        img[size[0]//2:] = [80, 80, 80]
    # Minor noise for realism
    noise = np.random.randint(-2, 3, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def apply_corruption(img_arr, corruption, severity, rng):
    """Apply corruption at given severity (0=clean, 1=full corruption)."""
    img = img_arr.astype(np.float64)
    h, w, c = img.shape

    if corruption == 'noise':
        # Gaussian noise with increasing std
        std = severity * 128.0  # 0 to 128
        noise = rng.normal(0, max(std, 1e-6), img.shape)
        img = np.clip(img + noise, 0, 255)

    elif corruption == 'darken':
        # Reduce brightness toward blackout
        factor = 1.0 - severity  # 1.0 (full brightness) to 0.0 (black)
        img = img * factor

    elif corruption == 'invert':
        # Partial color inversion: blend original with inverted
        inverted = 255.0 - img
        img = (1.0 - severity) * img + severity * inverted

    elif corruption == 'blur':
        # Gaussian blur with increasing kernel
        if severity > 0:
            # Convert to PIL, apply blur, convert back
            pil_img = Image.fromarray(img.astype(np.uint8))
            radius = severity * 20.0  # 0 to 20 pixel radius
            pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=radius))
            img = np.array(pil_img).astype(np.float64)

    elif corruption == 'occlude':
        # Random rectangular occlusion covering severity fraction of image
        if severity > 0:
            area_fraction = severity * 0.8  # up to 80% occluded
            n_rects = max(1, int(severity * 5))
            for _ in range(n_rects):
                rect_h = int(h * np.sqrt(area_fraction / n_rects))
                rect_w = int(w * np.sqrt(area_fraction / n_rects))
                y = rng.integers(0, max(h - rect_h, 1))
                x = rng.integers(0, max(w - rect_w, 1))
                # Gray occlusion
                img[y:y+rect_h, x:x+rect_w] = 128.0

    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    print("=" * 70, flush=True)
    print("GRADUAL DISTRIBUTION SHIFT ANALYSIS", flush=True)
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

    prompt = "In: What action should the robot take to drive forward at 25 m/s safely?\nOut:"

    # Total samples
    total = len(BASE_SCENES) * len(CORRUPTIONS) * len(SEVERITIES) * N_PER
    print(f"Total samples: {total} ({len(BASE_SCENES)} scenes × "
          f"{len(CORRUPTIONS)} corruptions × {len(SEVERITIES)} severities × "
          f"{N_PER} per)", flush=True)

    # First collect calibration data (severity=0 samples)
    print("\nPhase 1: Collecting calibration hidden states (clean images)...", flush=True)
    cal_hidden = []
    cal_count = 0

    for scene in BASE_SCENES:
        for i in range(15):  # 15 clean calibration images per scene
            base_img = create_base_image(scene, i + 100)  # offset seed
            image = Image.fromarray(base_img)
            inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=7,
                    do_sample=False,
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

            cal_hidden.append(hidden)
            cal_count += 1
            if cal_count % 10 == 0:
                print(f"  Calibration: {cal_count}/30", flush=True)

    # Compute calibration centroid
    cal_arr = np.array(cal_hidden)
    cal_mean = np.mean(cal_arr, axis=0)
    cal_norm = cal_mean / (np.linalg.norm(cal_mean) + 1e-10)
    print(f"  Calibration centroid computed from {len(cal_hidden)} samples.", flush=True)

    # Compute conformal threshold from calibration set (alpha=0.10)
    cal_cos_dists = []
    for h in cal_hidden:
        h_norm = h / (np.linalg.norm(h) + 1e-10)
        cal_cos_dists.append(1.0 - float(np.dot(h_norm, cal_norm)))
    cal_cos_sorted = sorted(cal_cos_dists)
    alpha = 0.10
    q_idx = int(np.ceil((1 - alpha) * (len(cal_cos_sorted) + 1))) - 1
    q_idx = min(q_idx, len(cal_cos_sorted) - 1)
    conformal_threshold = cal_cos_sorted[q_idx]
    print(f"  Conformal threshold (α={alpha}): {conformal_threshold:.6f}", flush=True)

    # Phase 2: Run all corruption experiments
    print(f"\nPhase 2: Running corruption experiments ({total} samples)...", flush=True)
    all_results = []
    sample_idx = 0

    for scene in BASE_SCENES:
        for corruption in CORRUPTIONS:
            for severity in SEVERITIES:
                for i in range(N_PER):
                    sample_idx += 1
                    rng = np.random.default_rng(seed=i * 3401 + hash(scene) % 3400
                                                + hash(corruption) % 3400
                                                + int(severity * 1000))

                    base_img = create_base_image(scene, i)
                    corrupted = apply_corruption(base_img, corruption, severity, rng)
                    image = Image.fromarray(corrupted)

                    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)

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

                    # Cosine distance
                    h_norm = hidden / (np.linalg.norm(hidden) + 1e-10)
                    cos_dist = 1.0 - float(np.dot(h_norm, cal_norm))

                    # Action mass
                    vocab_size = outputs.scores[0].shape[-1]
                    action_start = vocab_size - 256
                    dim_masses = []
                    for score in outputs.scores[:7]:
                        full_logits = score[0].float()
                        full_probs = torch.softmax(full_logits, dim=0).cpu().numpy()
                        action_probs = full_probs[action_start:]
                        dim_masses.append(float(action_probs.sum()))

                    # Entropy
                    entropies = []
                    for score in outputs.scores[:7]:
                        full_logits = score[0].float()
                        full_probs = torch.softmax(full_logits, dim=0)
                        ent = -torch.sum(full_probs * torch.log(full_probs + 1e-10)).item()
                        entropies.append(ent)

                    result = {
                        'scene': scene,
                        'corruption': corruption,
                        'severity': severity,
                        'idx': i,
                        'cos_dist': cos_dist,
                        'action_mass': float(np.mean(dim_masses)),
                        'entropy': float(np.mean(entropies)),
                        'hidden_norm': float(np.linalg.norm(hidden)),
                        'flagged': cos_dist > conformal_threshold,
                    }
                    all_results.append(result)

                    if sample_idx % 30 == 0 or sample_idx == total:
                        print(f"  [{sample_idx}/{total}] {scene}/{corruption}/"
                              f"sev={severity:.2f}: cos={cos_dist:.4f}, "
                              f"mass={result['action_mass']:.4f}, "
                              f"{'FLAGGED' if result['flagged'] else 'ok'}",
                              flush=True)

    # ===================================================================
    # Analysis
    # ===================================================================
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    # 1. Mean cosine distance by severity for each corruption
    print("\n1. Mean Cosine Distance by Severity", flush=True)
    print("-" * 80, flush=True)
    header = "Severity"
    print(f"  {header:>10}", end="", flush=True)
    for c in CORRUPTIONS:
        print(f" | {c:>10}", end="", flush=True)
    print("", flush=True)
    print("  " + "-" * (12 + 13 * len(CORRUPTIONS)), flush=True)

    for sev in SEVERITIES:
        print(f"  {sev:>10.2f}", end="", flush=True)
        for c in CORRUPTIONS:
            vals = [r['cos_dist'] for r in all_results
                    if r['corruption'] == c and r['severity'] == sev]
            mean_val = np.mean(vals) if vals else 0
            print(f" | {mean_val:>10.4f}", end="", flush=True)
        print("", flush=True)

    # 2. Mean action mass by severity for each corruption
    print("\n2. Mean Action Mass by Severity", flush=True)
    print("-" * 80, flush=True)
    print(f"  {header:>10}", end="", flush=True)
    for c in CORRUPTIONS:
        print(f" | {c:>10}", end="", flush=True)
    print("", flush=True)
    print("  " + "-" * (12 + 13 * len(CORRUPTIONS)), flush=True)

    for sev in SEVERITIES:
        print(f"  {sev:>10.2f}", end="", flush=True)
        for c in CORRUPTIONS:
            vals = [r['action_mass'] for r in all_results
                    if r['corruption'] == c and r['severity'] == sev]
            mean_val = np.mean(vals) if vals else 0
            print(f" | {mean_val:>10.4f}", end="", flush=True)
        print("", flush=True)

    # 3. Monotonicity check
    print("\n3. Monotonicity Check (cosine distance)", flush=True)
    print("-" * 80, flush=True)
    for c in CORRUPTIONS:
        means = []
        for sev in SEVERITIES:
            vals = [r['cos_dist'] for r in all_results
                    if r['corruption'] == c and r['severity'] == sev]
            means.append(np.mean(vals))
        # Check if monotonically increasing
        diffs = [means[i+1] - means[i] for i in range(len(means)-1)]
        monotonic = all(d >= -0.001 for d in diffs)  # small tolerance
        trend = "MONOTONIC" if monotonic else "NON-MONOTONIC"
        increase = means[-1] - means[0]
        print(f"  {c:<10}: {trend} | "
              f"range [{means[0]:.4f} → {means[-1]:.4f}] | "
              f"Δ={increase:+.4f}", flush=True)

    # 4. Flag rate by severity
    print("\n4. OOD Flag Rate by Severity (conformal α=0.10)", flush=True)
    print("-" * 80, flush=True)
    print(f"  {header:>10}", end="", flush=True)
    for c in CORRUPTIONS:
        print(f" | {c:>10}", end="", flush=True)
    print("", flush=True)
    print("  " + "-" * (12 + 13 * len(CORRUPTIONS)), flush=True)

    for sev in SEVERITIES:
        print(f"  {sev:>10.2f}", end="", flush=True)
        for c in CORRUPTIONS:
            samples = [r for r in all_results
                       if r['corruption'] == c and r['severity'] == sev]
            flagged = sum(1 for r in samples if r['flagged'])
            rate = flagged / len(samples) if samples else 0
            print(f" | {rate:>9.0%} ", end="", flush=True)
        print("", flush=True)

    # 5. Severity threshold: at what severity does >50% get flagged?
    print("\n5. Critical Severity (>50% flagged)", flush=True)
    print("-" * 80, flush=True)
    for c in CORRUPTIONS:
        critical_sev = None
        for sev in SEVERITIES:
            samples = [r for r in all_results
                       if r['corruption'] == c and r['severity'] == sev]
            flagged = sum(1 for r in samples if r['flagged'])
            rate = flagged / len(samples) if samples else 0
            if rate > 0.5 and critical_sev is None:
                critical_sev = sev
        if critical_sev is not None:
            print(f"  {c:<10}: severity ≥ {critical_sev:.2f}", flush=True)
        else:
            print(f"  {c:<10}: never reached >50% flagging", flush=True)

    # 6. Correlation between cosine distance and action mass
    print("\n6. Cosine Distance vs Action Mass Correlation", flush=True)
    print("-" * 80, flush=True)
    cos_vals = [r['cos_dist'] for r in all_results]
    mass_vals = [r['action_mass'] for r in all_results]
    ent_vals = [r['entropy'] for r in all_results]
    corr_cos_mass = np.corrcoef(cos_vals, mass_vals)[0, 1]
    corr_cos_ent = np.corrcoef(cos_vals, ent_vals)[0, 1]
    corr_mass_ent = np.corrcoef(mass_vals, ent_vals)[0, 1]
    print(f"  Cosine ↔ Action Mass:  r = {corr_cos_mass:+.3f}", flush=True)
    print(f"  Cosine ↔ Entropy:      r = {corr_cos_ent:+.3f}", flush=True)
    print(f"  Action Mass ↔ Entropy: r = {corr_mass_ent:+.3f}", flush=True)

    # 7. Sensitivity: which corruption causes largest cosine shift?
    print("\n7. Corruption Sensitivity Ranking", flush=True)
    print("-" * 80, flush=True)
    sensitivity = []
    for c in CORRUPTIONS:
        clean = [r['cos_dist'] for r in all_results
                 if r['corruption'] == c and r['severity'] == 0.0]
        full = [r['cos_dist'] for r in all_results
                if r['corruption'] == c and r['severity'] == 1.0]
        delta = np.mean(full) - np.mean(clean)
        sensitivity.append((c, delta))
    sensitivity.sort(key=lambda x: -x[1])
    for rank, (c, delta) in enumerate(sensitivity, 1):
        print(f"  {rank}. {c:<10}: Δcos = {delta:+.4f}", flush=True)

    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'gradual_distribution_shift',
        'experiment_number': 34,
        'timestamp': timestamp,
        'n_calibration': len(cal_hidden),
        'conformal_threshold': conformal_threshold,
        'conformal_alpha': alpha,
        'severities': SEVERITIES,
        'corruptions': CORRUPTIONS,
        'base_scenes': BASE_SCENES,
        'n_per': N_PER,
        'total_samples': total,
        'results': all_results,
    }

    output_path = os.path.join(RESULTS_DIR, f"gradual_shift_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
