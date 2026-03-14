"""
Action Prediction Consistency Under Distribution Shift.

Tests whether cosine distance correlates with actual action quality degradation
by measuring action consistency: how much do predicted actions change when the
same base scene is corrupted at different severity levels?

Metrics:
1. Action token agreement: fraction of 7 action tokens matching clean prediction
2. Action L2 distance: L2 distance between clean and corrupted action vectors
3. Softmax entropy increase: how much more uncertain is the model on corrupted inputs?
4. Correlation of cosine distance with action deviation

Also tests cross-sample action consistency (do similar scenes get similar actions?)
and action drift across corruptions.

Experiment 35 in the CalibDrive series.
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

BASE_SCENES = ['highway', 'urban']
CORRUPTIONS = ['noise', 'darken', 'invert', 'blur', 'occlude']
SEVERITIES = [0.0, 0.25, 0.50, 0.75, 1.0]
N_PER = 8  # samples per base scene


def create_base_image(scene, idx, size=(256, 256)):
    np.random.seed(idx * 3500 + hash(scene) % 35000)
    if scene == 'highway':
        img = np.zeros((*size, 3), dtype=np.uint8)
        img[:size[0]//2] = [135, 206, 235]
        img[size[0]//2:] = [80, 80, 80]
    elif scene == 'urban':
        img = np.zeros((*size, 3), dtype=np.uint8)
        img[:size[0]//3] = [135, 206, 235]
        img[size[0]//3:size[0]//2] = [139, 119, 101]
        img[size[0]//2:] = [80, 80, 80]
    else:
        img = np.zeros((*size, 3), dtype=np.uint8)
        img[:size[0]//2] = [135, 206, 235]
        img[size[0]//2:] = [80, 80, 80]
    noise = np.random.randint(-2, 3, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def apply_corruption(img_arr, corruption, severity, rng):
    img = img_arr.astype(np.float64)
    h, w, c = img.shape

    if corruption == 'noise':
        std = severity * 128.0
        noise = rng.normal(0, max(std, 1e-6), img.shape)
        img = np.clip(img + noise, 0, 255)
    elif corruption == 'darken':
        factor = 1.0 - severity
        img = img * factor
    elif corruption == 'invert':
        inverted = 255.0 - img
        img = (1.0 - severity) * img + severity * inverted
    elif corruption == 'blur':
        if severity > 0:
            pil_img = Image.fromarray(img.astype(np.uint8))
            radius = severity * 20.0
            pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=radius))
            img = np.array(pil_img).astype(np.float64)
    elif corruption == 'occlude':
        if severity > 0:
            area_fraction = severity * 0.8
            n_rects = max(1, int(severity * 5))
            for _ in range(n_rects):
                rect_h = int(h * np.sqrt(area_fraction / n_rects))
                rect_w = int(w * np.sqrt(area_fraction / n_rects))
                y = rng.integers(0, max(h - rect_h, 1))
                x = rng.integers(0, max(w - rect_w, 1))
                img[y:y+rect_h, x:x+rect_w] = 128.0

    return np.clip(img, 0, 255).astype(np.uint8)


def extract_info(model, processor, image, prompt):
    """Run inference and extract hidden state, action tokens, scores."""
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

    # Action tokens
    generated_ids = outputs.sequences[0]
    input_len = inputs['input_ids'].shape[1]
    action_tokens = generated_ids[input_len:input_len+7].cpu().numpy().tolist()

    # Action mass, entropy
    vocab_size = outputs.scores[0].shape[-1]
    action_start = vocab_size - 256
    dim_masses = []
    dim_entropies = []
    dim_confidences = []
    action_bins = []

    for score in outputs.scores[:7]:
        full_logits = score[0].float()
        full_probs = torch.softmax(full_logits, dim=0)

        # Action mass
        action_probs = full_probs[action_start:].cpu().numpy()
        dim_masses.append(float(action_probs.sum()))

        # Entropy
        ent = -torch.sum(full_probs * torch.log(full_probs + 1e-10)).item()
        dim_entropies.append(ent)

        # Confidence (max action prob)
        dim_confidences.append(float(action_probs.max()))

        # Action bin index
        action_bins.append(int(np.argmax(action_probs)))

    return {
        'hidden': hidden,
        'action_tokens': action_tokens,
        'action_bins': action_bins,
        'action_mass': float(np.mean(dim_masses)),
        'entropy': float(np.mean(dim_entropies)),
        'confidence': float(np.mean(dim_confidences)),
    }


def main():
    print("=" * 70, flush=True)
    print("ACTION PREDICTION CONSISTENCY UNDER DISTRIBUTION SHIFT", flush=True)
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

    # Phase 1: Calibration centroid
    print("\nPhase 1: Calibration centroid...", flush=True)
    cal_hidden = []
    for scene in BASE_SCENES:
        for i in range(15):
            base_img = create_base_image(scene, i + 200)
            image = Image.fromarray(base_img)
            info = extract_info(model, processor, image, prompt)
            cal_hidden.append(info['hidden'])
    cal_arr = np.array(cal_hidden)
    cal_mean = np.mean(cal_arr, axis=0)
    cal_norm = cal_mean / (np.linalg.norm(cal_mean) + 1e-10)
    print(f"  Centroid computed from {len(cal_hidden)} samples.", flush=True)

    # Phase 2: Get clean baseline actions for each test image
    print("\nPhase 2: Clean baseline actions...", flush=True)
    clean_baselines = {}  # (scene, idx) -> info dict
    for scene in BASE_SCENES:
        for i in range(N_PER):
            base_img = create_base_image(scene, i)
            image = Image.fromarray(base_img)
            info = extract_info(model, processor, image, prompt)
            h_norm = info['hidden'] / (np.linalg.norm(info['hidden']) + 1e-10)
            info['cos_dist'] = 1.0 - float(np.dot(h_norm, cal_norm))
            clean_baselines[(scene, i)] = info
            print(f"  {scene}_{i}: tokens={info['action_bins']}, "
                  f"mass={info['action_mass']:.4f}, "
                  f"cos={info['cos_dist']:.4f}", flush=True)

    # Phase 3: Corrupted actions
    print("\nPhase 3: Corrupted actions...", flush=True)
    total = len(BASE_SCENES) * N_PER * len(CORRUPTIONS) * (len(SEVERITIES) - 1)
    all_results = []
    sample_idx = 0

    for scene in BASE_SCENES:
        for i in range(N_PER):
            baseline = clean_baselines[(scene, i)]
            for corruption in CORRUPTIONS:
                for severity in SEVERITIES:
                    if severity == 0.0:
                        continue  # already have clean baseline
                    sample_idx += 1

                    rng = np.random.default_rng(seed=i * 3501
                                                + hash(scene) % 3500
                                                + hash(corruption) % 3500
                                                + int(severity * 1000))
                    base_img = create_base_image(scene, i)
                    corrupted = apply_corruption(base_img, corruption, severity, rng)
                    image = Image.fromarray(corrupted)

                    info = extract_info(model, processor, image, prompt)
                    h_norm = info['hidden'] / (np.linalg.norm(info['hidden']) + 1e-10)
                    cos_dist = 1.0 - float(np.dot(h_norm, cal_norm))

                    # Compute action deviation from clean baseline
                    token_agreement = sum(
                        1 for a, b in zip(info['action_bins'], baseline['action_bins'])
                        if a == b
                    ) / 7.0

                    bin_diff = [abs(a - b) for a, b in zip(
                        info['action_bins'], baseline['action_bins'])]
                    action_l2 = float(np.sqrt(sum(d**2 for d in bin_diff)))

                    entropy_increase = info['entropy'] - baseline['entropy']

                    result = {
                        'scene': scene,
                        'idx': i,
                        'corruption': corruption,
                        'severity': severity,
                        'cos_dist': cos_dist,
                        'cos_dist_baseline': baseline['cos_dist'],
                        'cos_dist_increase': cos_dist - baseline['cos_dist'],
                        'action_mass': info['action_mass'],
                        'entropy': info['entropy'],
                        'token_agreement': token_agreement,
                        'action_l2': action_l2,
                        'entropy_increase': entropy_increase,
                        'action_bins': info['action_bins'],
                        'baseline_bins': baseline['action_bins'],
                    }
                    all_results.append(result)

                    if sample_idx % 20 == 0 or sample_idx == total:
                        print(f"  [{sample_idx}/{total}] {scene}/{corruption}/"
                              f"sev={severity:.2f}: agree={token_agreement:.2f}, "
                              f"L2={action_l2:.1f}, cos={cos_dist:.4f}",
                              flush=True)

    # ===================================================================
    # Analysis
    # ===================================================================
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    # 1. Token agreement by severity
    print("\n1. Mean Token Agreement by Severity (1.0 = all 7 tokens match clean)", flush=True)
    print("-" * 80, flush=True)
    header = "Severity"
    print(f"  {header:>10}", end="", flush=True)
    for c in CORRUPTIONS:
        print(f" | {c:>10}", end="", flush=True)
    print("", flush=True)
    print("  " + "-" * (12 + 13 * len(CORRUPTIONS)), flush=True)

    for sev in [s for s in SEVERITIES if s > 0]:
        print(f"  {sev:>10.2f}", end="", flush=True)
        for c in CORRUPTIONS:
            vals = [r['token_agreement'] for r in all_results
                    if r['corruption'] == c and r['severity'] == sev]
            print(f" | {np.mean(vals):>10.3f}", end="", flush=True)
        print("", flush=True)

    # 2. Action L2 by severity
    print("\n2. Mean Action L2 Distance by Severity (0 = identical to clean)", flush=True)
    print("-" * 80, flush=True)
    print(f"  {header:>10}", end="", flush=True)
    for c in CORRUPTIONS:
        print(f" | {c:>10}", end="", flush=True)
    print("", flush=True)
    print("  " + "-" * (12 + 13 * len(CORRUPTIONS)), flush=True)

    for sev in [s for s in SEVERITIES if s > 0]:
        print(f"  {sev:>10.2f}", end="", flush=True)
        for c in CORRUPTIONS:
            vals = [r['action_l2'] for r in all_results
                    if r['corruption'] == c and r['severity'] == sev]
            print(f" | {np.mean(vals):>10.1f}", end="", flush=True)
        print("", flush=True)

    # 3. Correlation: cosine distance vs action deviation
    print("\n3. Correlation Analysis", flush=True)
    print("-" * 80, flush=True)
    cos_dists = [r['cos_dist'] for r in all_results]
    agreements = [r['token_agreement'] for r in all_results]
    l2s = [r['action_l2'] for r in all_results]
    ent_incs = [r['entropy_increase'] for r in all_results]
    masses = [r['action_mass'] for r in all_results]

    r_cos_agree = np.corrcoef(cos_dists, agreements)[0, 1]
    r_cos_l2 = np.corrcoef(cos_dists, l2s)[0, 1]
    r_cos_ent = np.corrcoef(cos_dists, ent_incs)[0, 1]
    r_mass_agree = np.corrcoef(masses, agreements)[0, 1]
    r_mass_l2 = np.corrcoef(masses, l2s)[0, 1]

    print(f"  Cosine dist ↔ Token agreement: r = {r_cos_agree:+.3f}", flush=True)
    print(f"  Cosine dist ↔ Action L2:       r = {r_cos_l2:+.3f}", flush=True)
    print(f"  Cosine dist ↔ Entropy change:  r = {r_cos_ent:+.3f}", flush=True)
    print(f"  Action mass ↔ Token agreement: r = {r_mass_agree:+.3f}", flush=True)
    print(f"  Action mass ↔ Action L2:       r = {r_mass_l2:+.3f}", flush=True)

    # 4. Per-corruption correlation
    print("\n4. Per-Corruption: Cosine ↔ Token Agreement Correlation", flush=True)
    print("-" * 80, flush=True)
    for c in CORRUPTIONS:
        c_cos = [r['cos_dist'] for r in all_results if r['corruption'] == c]
        c_agree = [r['token_agreement'] for r in all_results if r['corruption'] == c]
        r = np.corrcoef(c_cos, c_agree)[0, 1]
        print(f"  {c:<10}: r = {r:+.3f}", flush=True)

    # 5. At what cosine distance does action quality collapse?
    print("\n5. Action Quality vs Cosine Distance Bins", flush=True)
    print("-" * 80, flush=True)
    bins = [(0.0, 0.55), (0.55, 0.65), (0.65, 0.75), (0.75, 0.85), (0.85, 1.0)]
    print(f"  {'Cos bin':>15} | {'N':>5} | {'Agree':>8} | {'L2':>8} | {'Entropy':>8}", flush=True)
    print("  " + "-" * 55, flush=True)
    for lo, hi in bins:
        bin_samples = [r for r in all_results if lo <= r['cos_dist'] < hi]
        if bin_samples:
            n = len(bin_samples)
            mean_agree = np.mean([r['token_agreement'] for r in bin_samples])
            mean_l2 = np.mean([r['action_l2'] for r in bin_samples])
            mean_ent = np.mean([r['entropy'] for r in bin_samples])
            print(f"  [{lo:.2f}, {hi:.2f}) | {n:>5} | {mean_agree:>8.3f} | "
                  f"{mean_l2:>8.1f} | {mean_ent:>8.2f}", flush=True)

    # 6. Cross-sample action consistency within clean
    print("\n6. Cross-Sample Action Consistency (Clean)", flush=True)
    print("-" * 80, flush=True)
    for scene in BASE_SCENES:
        scene_bins = [clean_baselines[(scene, i)]['action_bins']
                      for i in range(N_PER)]
        # Pairwise token agreement
        agreements_list = []
        for a in range(len(scene_bins)):
            for b in range(a+1, len(scene_bins)):
                agree = sum(1 for x, y in zip(scene_bins[a], scene_bins[b])
                           if x == y) / 7.0
                agreements_list.append(agree)
        print(f"  {scene}: mean pairwise agreement = "
              f"{np.mean(agreements_list):.3f} ± {np.std(agreements_list):.3f}",
              flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Convert baseline info for JSON serialization
    baselines_json = {}
    for (s, i), info in clean_baselines.items():
        key = f"{s}_{i}"
        baselines_json[key] = {
            'action_bins': info['action_bins'],
            'action_mass': info['action_mass'],
            'entropy': info['entropy'],
            'cos_dist': info['cos_dist'],
        }

    output = {
        'experiment': 'action_consistency',
        'experiment_number': 35,
        'timestamp': timestamp,
        'n_per': N_PER,
        'total_corrupted': len(all_results),
        'clean_baselines': baselines_json,
        'results': [{k: v for k, v in r.items()} for r in all_results],
    }

    output_path = os.path.join(RESULTS_DIR, f"action_consistency_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
