"""
Temporal Consistency of the Cosine-Distance OOD Detector on Real OpenVLA-7B.

Tests whether corruption detection produces temporally stable signals across
sequential frames that simulate driving scenarios with camera jitter.

Tests:
1. Frame-to-frame stability: 10 slightly varied clean frames (camera jitter)
   -> verify low inter-frame cosine-distance variance.
2. Corruption onset detection: 5 clean then 5 fog frames -> measure
   detection latency (first frame flagged after transition).
3. Gradual corruption: severity ramp 0.0->1.0 across 10 frames -> measure
   detection latency.
4. Intermittent corruption: alternating clean/corrupt frames -> measure
   oscillation in the detector signal.
5. Recovery detection: 5 corrupt then 5 clean frames -> measure recovery
   latency (first clean frame NOT flagged after corruption ends).
6. Multi-corruption sequence: clean->fog->night->blur->clean -> measure
   transition detection accuracy.

Experiment in the CalibDrive series.
"""
import os
import json
import datetime
import numpy as np
import torch
from PIL import Image, ImageFilter

RESULTS_DIR = "/workspace/Vizuara-VLA-Research/experiments"
os.makedirs(RESULTS_DIR, exist_ok=True)

IMAGE_SIZE = (256, 256)
NUM_FRAMES = 10
CALIBRATION_SAMPLES = 30


# ------------------------------------------------------------------
# Image generation helpers
# ------------------------------------------------------------------

def create_clean_driving_image(seed, size=IMAGE_SIZE):
    """Create a simple in-distribution driving scene (highway/sky)."""
    rng = np.random.RandomState(seed)
    img = np.zeros((*size, 3), dtype=np.uint8)
    img[:size[0] // 2] = [135, 206, 235]   # sky
    img[size[0] // 2:] = [80, 80, 80]      # road
    # Minor realistic noise
    noise = rng.randint(-2, 3, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def add_camera_jitter(image, sigma, rng):
    """Simulate frame-to-frame camera jitter by adding small pixel noise."""
    arr = np.array(image).astype(np.int16)
    noise = rng.normal(0, sigma, arr.shape).astype(np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def apply_corruption(image, ctype, severity=1.0):
    """Apply a corruption of the given type and severity to a PIL Image."""
    arr = np.array(image).astype(np.float32) / 255.0
    if ctype == 'fog':
        arr = arr * (1 - 0.6 * severity) + 0.6 * severity
    elif ctype == 'night':
        arr = arr * max(0.01, 1.0 - 0.95 * severity)
    elif ctype == 'noise':
        arr = arr + np.random.RandomState(42).randn(*arr.shape) * 0.3 * severity
        arr = np.clip(arr, 0, 1)
    elif ctype == 'blur':
        return image.filter(ImageFilter.GaussianBlur(radius=10 * severity))
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))


def cosine_dist(a, b):
    """Cosine distance between two vectors."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    return 1.0 - np.dot(a, b) / (na * nb)


# ------------------------------------------------------------------
# Model helpers
# ------------------------------------------------------------------

def extract_hidden(model, processor, image, prompt, layer=3):
    """Extract a hidden-state vector from the specified transformer layer."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return fwd.hidden_states[layer][0, -1, :].float().cpu().numpy()


# ------------------------------------------------------------------
# Sequence generators for each test
# ------------------------------------------------------------------

def generate_stability_sequence(base_seed):
    """Test 1: 10 clean frames with slight camera jitter (sigma 2-5)."""
    base_image = create_clean_driving_image(base_seed)
    frames = []
    for i in range(NUM_FRAMES):
        rng = np.random.RandomState(base_seed * 1000 + i)
        sigma = rng.uniform(2, 5)
        frames.append({
            'image': add_camera_jitter(base_image, sigma, rng),
            'label': 'clean',
            'corruption': None,
            'severity': 0.0,
        })
    return frames


def generate_onset_sequence(base_seed):
    """Test 2: 5 clean frames then 5 fog frames (severity=1.0)."""
    base_image = create_clean_driving_image(base_seed)
    frames = []
    for i in range(NUM_FRAMES):
        rng = np.random.RandomState(base_seed * 1000 + i)
        jittered = add_camera_jitter(base_image, 3.0, rng)
        if i < 5:
            frames.append({
                'image': jittered,
                'label': 'clean',
                'corruption': None,
                'severity': 0.0,
            })
        else:
            frames.append({
                'image': apply_corruption(jittered, 'fog', severity=1.0),
                'label': 'corrupt',
                'corruption': 'fog',
                'severity': 1.0,
            })
    return frames


def generate_gradual_sequence(base_seed):
    """Test 3: severity ramp from 0.0 to 1.0 across 10 frames (fog)."""
    base_image = create_clean_driving_image(base_seed)
    frames = []
    for i in range(NUM_FRAMES):
        severity = i / (NUM_FRAMES - 1)  # 0.0, 0.111, ..., 1.0
        rng = np.random.RandomState(base_seed * 1000 + i)
        jittered = add_camera_jitter(base_image, 3.0, rng)
        frames.append({
            'image': apply_corruption(jittered, 'fog', severity=severity),
            'label': 'clean' if severity < 0.01 else 'corrupt',
            'corruption': 'fog',
            'severity': severity,
        })
    return frames


def generate_intermittent_sequence(base_seed):
    """Test 4: alternating clean/corrupt frames (fog severity=1.0)."""
    base_image = create_clean_driving_image(base_seed)
    frames = []
    for i in range(NUM_FRAMES):
        rng = np.random.RandomState(base_seed * 1000 + i)
        jittered = add_camera_jitter(base_image, 3.0, rng)
        is_corrupt = (i % 2 == 1)
        if is_corrupt:
            frames.append({
                'image': apply_corruption(jittered, 'fog', severity=1.0),
                'label': 'corrupt',
                'corruption': 'fog',
                'severity': 1.0,
            })
        else:
            frames.append({
                'image': jittered,
                'label': 'clean',
                'corruption': None,
                'severity': 0.0,
            })
    return frames


def generate_recovery_sequence(base_seed):
    """Test 5: 5 corrupt (fog) then 5 clean frames."""
    base_image = create_clean_driving_image(base_seed)
    frames = []
    for i in range(NUM_FRAMES):
        rng = np.random.RandomState(base_seed * 1000 + i)
        jittered = add_camera_jitter(base_image, 3.0, rng)
        if i < 5:
            frames.append({
                'image': apply_corruption(jittered, 'fog', severity=1.0),
                'label': 'corrupt',
                'corruption': 'fog',
                'severity': 1.0,
            })
        else:
            frames.append({
                'image': jittered,
                'label': 'clean',
                'corruption': None,
                'severity': 0.0,
            })
    return frames


def generate_multi_corruption_sequence(base_seed):
    """Test 6: clean(2)->fog(2)->night(2)->blur(2)->clean(2)."""
    base_image = create_clean_driving_image(base_seed)
    # Sequence definition: (corruption_type_or_None, severity, label)
    plan = [
        (None, 0.0, 'clean'),      # frame 0
        (None, 0.0, 'clean'),      # frame 1
        ('fog', 1.0, 'fog'),       # frame 2
        ('fog', 1.0, 'fog'),       # frame 3
        ('night', 1.0, 'night'),   # frame 4
        ('night', 1.0, 'night'),   # frame 5
        ('blur', 1.0, 'blur'),     # frame 6
        ('blur', 1.0, 'blur'),     # frame 7
        (None, 0.0, 'clean'),      # frame 8
        (None, 0.0, 'clean'),      # frame 9
    ]
    frames = []
    for i, (ctype, severity, label) in enumerate(plan):
        rng = np.random.RandomState(base_seed * 1000 + i)
        jittered = add_camera_jitter(base_image, 3.0, rng)
        if ctype is not None:
            frames.append({
                'image': apply_corruption(jittered, ctype, severity=severity),
                'label': label,
                'corruption': ctype,
                'severity': severity,
            })
        else:
            frames.append({
                'image': jittered,
                'label': label,
                'corruption': None,
                'severity': 0.0,
            })
    return frames


# ------------------------------------------------------------------
# Analysis helpers
# ------------------------------------------------------------------

def compute_detection_latency(flags, transition_idx):
    """Frames after transition_idx until first flag is True. None if never."""
    for offset, flag in enumerate(flags[transition_idx:]):
        if flag:
            return offset
    return None


def compute_recovery_latency(flags, recovery_idx):
    """Frames after recovery_idx until first flag is False. None if never."""
    for offset, flag in enumerate(flags[recovery_idx:]):
        if not flag:
            return offset
    return None


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    print("=" * 70, flush=True)
    print("TEMPORAL CONSISTENCY OF OOD DETECTOR", flush=True)
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

    prompt = "In: What action should the robot take to pick up the object?\nOut:"

    # =================================================================
    # Phase 1: Calibration -- build centroid from clean driving images
    # =================================================================
    print("\nPhase 1: Calibration centroid...", flush=True)
    cal_embeddings = []
    for i in range(CALIBRATION_SAMPLES):
        img = create_clean_driving_image(seed=9000 + i)
        rng = np.random.RandomState(9000 + i)
        img = add_camera_jitter(img, sigma=3.0, rng=rng)
        emb = extract_hidden(model, processor, img, prompt)
        cal_embeddings.append(emb)
        if (i + 1) % 10 == 0:
            print(f"  Calibration: {i + 1}/{CALIBRATION_SAMPLES}", flush=True)

    cal_arr = np.array(cal_embeddings)
    centroid = np.mean(cal_arr, axis=0)

    # Conformal threshold (alpha=0.10)
    cal_dists = sorted([cosine_dist(emb, centroid) for emb in cal_embeddings])
    alpha = 0.10
    q_idx = min(
        int(np.ceil((1 - alpha) * (len(cal_dists) + 1))) - 1,
        len(cal_dists) - 1,
    )
    threshold = cal_dists[q_idx]
    cal_mean_dist = float(np.mean(cal_dists))
    cal_std_dist = float(np.std(cal_dists))
    print(f"  Centroid from {CALIBRATION_SAMPLES} samples.", flush=True)
    print(f"  Calibration cosine dist: mean={cal_mean_dist:.6f}, "
          f"std={cal_std_dist:.6f}", flush=True)
    print(f"  Conformal threshold (alpha={alpha}): {threshold:.6f}", flush=True)

    # =================================================================
    # Phase 2: Run all temporal tests
    # =================================================================
    N_TRIALS = 3  # repeat each test with different base seeds

    tests = {
        'stability': {
            'description': 'Frame-to-frame stability (clean + jitter)',
            'generator': generate_stability_sequence,
        },
        'onset': {
            'description': 'Corruption onset (clean->fog)',
            'generator': generate_onset_sequence,
        },
        'gradual': {
            'description': 'Gradual corruption (severity ramp 0->1)',
            'generator': generate_gradual_sequence,
        },
        'intermittent': {
            'description': 'Intermittent corruption (alternating)',
            'generator': generate_intermittent_sequence,
        },
        'recovery': {
            'description': 'Recovery detection (fog->clean)',
            'generator': generate_recovery_sequence,
        },
        'multi_corruption': {
            'description': 'Multi-corruption (clean->fog->night->blur->clean)',
            'generator': generate_multi_corruption_sequence,
        },
    }

    all_test_results = {}
    total_frames = len(tests) * N_TRIALS * NUM_FRAMES
    frame_counter = 0

    for test_name, test_cfg in tests.items():
        print(f"\n{'=' * 70}", flush=True)
        print(f"Test: {test_cfg['description']}", flush=True)
        print(f"{'=' * 70}", flush=True)

        trial_results = []
        for trial in range(N_TRIALS):
            base_seed = 5000 + trial * 100 + hash(test_name) % 1000
            sequence = test_cfg['generator'](base_seed)

            frame_data = []
            embeddings = []
            for fi, frame_info in enumerate(sequence):
                frame_counter += 1
                emb = extract_hidden(
                    model, processor, frame_info['image'], prompt,
                )
                dist = cosine_dist(emb, centroid)
                flagged = dist > threshold

                frame_data.append({
                    'frame': fi,
                    'label': frame_info['label'],
                    'corruption': frame_info['corruption'],
                    'severity': frame_info['severity'],
                    'cosine_dist': float(dist),
                    'flagged': bool(flagged),
                })
                embeddings.append(emb)

                if frame_counter % 20 == 0 or frame_counter == total_frames:
                    print(f"  [{frame_counter}/{total_frames}] "
                          f"{test_name} trial={trial} frame={fi}: "
                          f"cos={dist:.6f} {'FLAGGED' if flagged else 'ok'}",
                          flush=True)

            # Inter-frame cosine distances (consecutive pairs)
            inter_frame_dists = [
                float(cosine_dist(embeddings[i], embeddings[i + 1]))
                for i in range(len(embeddings) - 1)
            ]

            trial_results.append({
                'trial': trial,
                'base_seed': base_seed,
                'frames': frame_data,
                'inter_frame_dists': inter_frame_dists,
            })

        all_test_results[test_name] = trial_results

    # =================================================================
    # Phase 3: Analysis
    # =================================================================
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    analysis = {}

    # -----------------------------------------------------------------
    # Test 1: Frame-to-frame stability
    # -----------------------------------------------------------------
    print("\n1. Frame-to-Frame Stability (Clean + Jitter)", flush=True)
    print("-" * 70, flush=True)

    stability_trials = all_test_results['stability']
    all_cos_dists = []
    all_inter_dists = []
    all_variances = []
    for tr in stability_trials:
        dists = [f['cosine_dist'] for f in tr['frames']]
        all_cos_dists.extend(dists)
        all_inter_dists.extend(tr['inter_frame_dists'])
        all_variances.append(float(np.var(dists)))
        print(f"  Trial {tr['trial']}: "
              f"mean_cos={np.mean(dists):.6f}, "
              f"std_cos={np.std(dists):.6f}, "
              f"max_cos={np.max(dists):.6f}, "
              f"flags={sum(f['flagged'] for f in tr['frames'])}/{NUM_FRAMES}",
              flush=True)

    mean_inter_frame = float(np.mean(all_inter_dists))
    std_inter_frame = float(np.std(all_inter_dists))
    mean_variance = float(np.mean(all_variances))
    false_flag_rate = sum(
        f['flagged'] for tr in stability_trials for f in tr['frames']
    ) / (len(stability_trials) * NUM_FRAMES)

    print(f"\n  Summary:", flush=True)
    print(f"    Mean inter-frame cosine dist: {mean_inter_frame:.6f}", flush=True)
    print(f"    Std inter-frame cosine dist:  {std_inter_frame:.6f}", flush=True)
    print(f"    Mean within-sequence variance: {mean_variance:.8f}", flush=True)
    print(f"    False flag rate:              {false_flag_rate:.1%}", flush=True)

    analysis['stability'] = {
        'mean_inter_frame_dist': mean_inter_frame,
        'std_inter_frame_dist': std_inter_frame,
        'mean_within_sequence_variance': mean_variance,
        'false_flag_rate': false_flag_rate,
    }

    # -----------------------------------------------------------------
    # Test 2: Corruption onset detection
    # -----------------------------------------------------------------
    print("\n2. Corruption Onset Detection (Clean->Fog)", flush=True)
    print("-" * 70, flush=True)

    onset_trials = all_test_results['onset']
    onset_latencies = []
    for tr in onset_trials:
        flags = [f['flagged'] for f in tr['frames']]
        dists = [f['cosine_dist'] for f in tr['frames']]
        latency = compute_detection_latency(flags, transition_idx=5)
        onset_latencies.append(latency)
        clean_mean = np.mean(dists[:5])
        corrupt_mean = np.mean(dists[5:])
        print(f"  Trial {tr['trial']}: "
              f"clean_mean={clean_mean:.6f}, corrupt_mean={corrupt_mean:.6f}, "
              f"latency={latency if latency is not None else 'NEVER'} frames, "
              f"flags={['F' if f else '.' for f in flags]}",
              flush=True)

    valid_latencies = [l for l in onset_latencies if l is not None]
    mean_onset = float(np.mean(valid_latencies)) if valid_latencies else None
    detection_rate = len(valid_latencies) / len(onset_latencies)

    print(f"\n  Summary:", flush=True)
    print(f"    Detection rate:  {detection_rate:.1%}", flush=True)
    print(f"    Mean latency:    {mean_onset if mean_onset is not None else 'N/A'} frames",
          flush=True)

    analysis['onset'] = {
        'latencies': onset_latencies,
        'mean_latency': mean_onset,
        'detection_rate': detection_rate,
    }

    # -----------------------------------------------------------------
    # Test 3: Gradual corruption
    # -----------------------------------------------------------------
    print("\n3. Gradual Corruption (Severity Ramp 0->1)", flush=True)
    print("-" * 70, flush=True)

    gradual_trials = all_test_results['gradual']
    gradual_latencies = []
    for tr in gradual_trials:
        flags = [f['flagged'] for f in tr['frames']]
        dists = [f['cosine_dist'] for f in tr['frames']]
        sevs = [f['severity'] for f in tr['frames']]
        # Detection latency from frame 0 (everything starts clean)
        first_flag = None
        first_flag_severity = None
        for fi, f in enumerate(tr['frames']):
            if f['flagged']:
                first_flag = fi
                first_flag_severity = f['severity']
                break
        gradual_latencies.append({
            'first_flag_frame': first_flag,
            'first_flag_severity': first_flag_severity,
        })
        dist_str = " ".join(f"{d:.4f}" for d in dists)
        print(f"  Trial {tr['trial']}: "
              f"first_flag=frame {first_flag} (sev={first_flag_severity}), "
              f"flags={['F' if f else '.' for f in flags]}",
              flush=True)
        print(f"    dists: {dist_str}", flush=True)

    # Check monotonicity of cosine distance with severity
    mono_checks = []
    for tr in gradual_trials:
        dists = [f['cosine_dist'] for f in tr['frames']]
        diffs = [dists[i + 1] - dists[i] for i in range(len(dists) - 1)]
        monotonic = all(d >= -1e-4 for d in diffs)
        mono_checks.append(monotonic)

    valid_flag_frames = [
        gl['first_flag_frame'] for gl in gradual_latencies
        if gl['first_flag_frame'] is not None
    ]
    valid_flag_sevs = [
        gl['first_flag_severity'] for gl in gradual_latencies
        if gl['first_flag_severity'] is not None
    ]
    avg_flag_frame = float(np.mean(valid_flag_frames)) if valid_flag_frames else None
    avg_flag_sev = float(np.mean(valid_flag_sevs)) if valid_flag_sevs else None

    print(f"\n  Summary:", flush=True)
    print(f"    Monotonic trials: {sum(mono_checks)}/{len(mono_checks)}", flush=True)
    print(f"    Mean first-flag frame:    "
          f"{avg_flag_frame if avg_flag_frame is not None else 'NEVER'}",
          flush=True)
    if avg_flag_sev is not None:
        print(f"    Mean first-flag severity: {avg_flag_sev:.3f}", flush=True)
    else:
        print(f"    Mean first-flag severity: N/A", flush=True)

    analysis['gradual'] = {
        'latencies': gradual_latencies,
        'mean_first_flag_frame': avg_flag_frame,
        'mean_first_flag_severity': avg_flag_sev,
        'monotonic_fraction': sum(mono_checks) / len(mono_checks),
    }

    # -----------------------------------------------------------------
    # Test 4: Intermittent corruption
    # -----------------------------------------------------------------
    print("\n4. Intermittent Corruption (Alternating Clean/Fog)", flush=True)
    print("-" * 70, flush=True)

    intermittent_trials = all_test_results['intermittent']
    signal_oscillations = []
    for tr in intermittent_trials:
        dists = [f['cosine_dist'] for f in tr['frames']]
        flags = [f['flagged'] for f in tr['frames']]
        labels = [f['label'] for f in tr['frames']]

        # Measure signal oscillation: consecutive sign changes in (dist - threshold)
        deviations = [d - threshold for d in dists]
        sign_changes = sum(
            1 for i in range(len(deviations) - 1)
            if (deviations[i] > 0) != (deviations[i + 1] > 0)
        )
        signal_oscillations.append(sign_changes)

        # Classification accuracy per frame
        correct = sum(
            1 for f in tr['frames']
            if (f['flagged'] and f['label'] == 'corrupt')
            or (not f['flagged'] and f['label'] == 'clean')
        )
        accuracy = correct / NUM_FRAMES

        print(f"  Trial {tr['trial']}: "
              f"sign_changes={sign_changes}, accuracy={accuracy:.1%}, "
              f"flags={['F' if fl else '.' for fl in flags]}, "
              f"labels={['C' if l == 'corrupt' else '.' for l in labels]}",
              flush=True)

    mean_oscillations = float(np.mean(signal_oscillations))
    # Expected oscillations for perfect tracking of alternating pattern: ~9
    # (every consecutive pair differs)
    per_frame_accuracies = []
    for tr in intermittent_trials:
        for f in tr['frames']:
            is_correct = (
                (f['flagged'] and f['label'] == 'corrupt')
                or (not f['flagged'] and f['label'] == 'clean')
            )
            per_frame_accuracies.append(is_correct)
    overall_accuracy = float(np.mean(per_frame_accuracies))

    print(f"\n  Summary:", flush=True)
    print(f"    Mean signal oscillations:  {mean_oscillations:.1f} "
          f"(max possible={NUM_FRAMES - 1})", flush=True)
    print(f"    Per-frame accuracy:        {overall_accuracy:.1%}", flush=True)
    print(f"    Detector behaviour:        "
          f"{'Oscillating (tracks corruption)' if mean_oscillations > 3 else 'Stable (smoothed)'}",
          flush=True)

    analysis['intermittent'] = {
        'mean_oscillations': mean_oscillations,
        'max_possible_oscillations': NUM_FRAMES - 1,
        'per_frame_accuracy': overall_accuracy,
    }

    # -----------------------------------------------------------------
    # Test 5: Recovery detection
    # -----------------------------------------------------------------
    print("\n5. Recovery Detection (Fog->Clean)", flush=True)
    print("-" * 70, flush=True)

    recovery_trials = all_test_results['recovery']
    recovery_latencies = []
    for tr in recovery_trials:
        flags = [f['flagged'] for f in tr['frames']]
        dists = [f['cosine_dist'] for f in tr['frames']]
        latency = compute_recovery_latency(flags, recovery_idx=5)
        recovery_latencies.append(latency)
        corrupt_mean = np.mean(dists[:5])
        clean_mean = np.mean(dists[5:])
        print(f"  Trial {tr['trial']}: "
              f"corrupt_mean={corrupt_mean:.6f}, clean_mean={clean_mean:.6f}, "
              f"recovery_latency={latency if latency is not None else 'NEVER'} frames, "
              f"flags={['F' if f else '.' for f in flags]}",
              flush=True)

    valid_recovery = [l for l in recovery_latencies if l is not None]
    mean_recovery = float(np.mean(valid_recovery)) if valid_recovery else None
    recovery_rate = len(valid_recovery) / len(recovery_latencies)

    print(f"\n  Summary:", flush=True)
    print(f"    Recovery rate:   {recovery_rate:.1%}", flush=True)
    print(f"    Mean latency:    "
          f"{mean_recovery if mean_recovery is not None else 'NEVER'} frames", flush=True)

    analysis['recovery'] = {
        'latencies': recovery_latencies,
        'mean_latency': mean_recovery,
        'recovery_rate': recovery_rate,
    }

    # -----------------------------------------------------------------
    # Test 6: Multi-corruption sequence
    # -----------------------------------------------------------------
    print("\n6. Multi-Corruption Sequence (clean->fog->night->blur->clean)", flush=True)
    print("-" * 70, flush=True)

    multi_trials = all_test_results['multi_corruption']
    # Expected pattern: frames 0-1 clean, 2-3 fog, 4-5 night, 6-7 blur, 8-9 clean
    transition_points = {
        'clean->fog': 2,
        'fog->night': 4,
        'night->blur': 6,
        'blur->clean': 8,
    }

    transition_detections = {t: [] for t in transition_points}
    per_segment_flag_rates = {
        'clean_start': [],
        'fog': [],
        'night': [],
        'blur': [],
        'clean_end': [],
    }

    for tr in multi_trials:
        dists = [f['cosine_dist'] for f in tr['frames']]
        flags = [f['flagged'] for f in tr['frames']]
        labels = [f['label'] for f in tr['frames']]

        # Per-segment flag rates
        per_segment_flag_rates['clean_start'].append(
            np.mean([float(flags[i]) for i in range(0, 2)]))
        per_segment_flag_rates['fog'].append(
            np.mean([float(flags[i]) for i in range(2, 4)]))
        per_segment_flag_rates['night'].append(
            np.mean([float(flags[i]) for i in range(4, 6)]))
        per_segment_flag_rates['blur'].append(
            np.mean([float(flags[i]) for i in range(6, 8)]))
        per_segment_flag_rates['clean_end'].append(
            np.mean([float(flags[i]) for i in range(8, 10)]))

        # Check if transitions produce detectable jumps in cosine distance
        for t_name, t_idx in transition_points.items():
            pre_mean = np.mean(dists[max(0, t_idx - 2):t_idx])
            post_mean = np.mean(dists[t_idx:min(NUM_FRAMES, t_idx + 2)])
            delta = post_mean - pre_mean
            transition_detections[t_name].append(float(delta))

        dist_str = " ".join(f"{d:.4f}" for d in dists)
        flag_str = "".join(['F' if f else '.' for f in flags])
        label_str = "".join([l[0].upper() for l in labels])
        print(f"  Trial {tr['trial']}:", flush=True)
        print(f"    labels: {label_str}", flush=True)
        print(f"    flags:  {flag_str}", flush=True)
        print(f"    dists:  {dist_str}", flush=True)

    print(f"\n  Per-segment flag rates:", flush=True)
    for seg, rates in per_segment_flag_rates.items():
        print(f"    {seg:<12}: {np.mean(rates):.1%}", flush=True)

    print(f"\n  Transition cosine-distance jumps (mean delta):", flush=True)
    for t_name, deltas in transition_detections.items():
        mean_delta = float(np.mean(deltas))
        print(f"    {t_name:<16}: {mean_delta:+.6f}", flush=True)

    analysis['multi_corruption'] = {
        'per_segment_flag_rates': {
            k: float(np.mean(v)) for k, v in per_segment_flag_rates.items()
        },
        'transition_deltas': {
            k: float(np.mean(v)) for k, v in transition_detections.items()
        },
    }

    # =================================================================
    # Overall summary table
    # =================================================================
    print("\n" + "=" * 70, flush=True)
    print("OVERALL SUMMARY", flush=True)
    print("=" * 70, flush=True)

    print(f"\n  {'Test':<30} | {'Key Metric':<30} | {'Value':>10}", flush=True)
    print("  " + "-" * 75, flush=True)
    print(f"  {'Stability':<30} | {'False flag rate':<30} | "
          f"{false_flag_rate:>9.1%}", flush=True)
    print(f"  {'Stability':<30} | {'Inter-frame dist std':<30} | "
          f"{std_inter_frame:>10.6f}", flush=True)
    print(f"  {'Onset (clean->fog)':<30} | {'Detection rate':<30} | "
          f"{detection_rate:>9.1%}", flush=True)
    onset_val = f"{mean_onset:.1f}" if mean_onset is not None else "N/A"
    print(f"  {'Onset (clean->fog)':<30} | {'Mean latency (frames)':<30} | "
          f"{onset_val:>10}", flush=True)
    print(f"  {'Gradual ramp':<30} | {'Monotonic fraction':<30} | "
          f"{analysis['gradual']['monotonic_fraction']:>9.1%}", flush=True)
    gff = analysis['gradual']['mean_first_flag_frame']
    gff_val = f"{gff:.1f}" if gff is not None else "NEVER"
    print(f"  {'Gradual ramp':<30} | {'Mean first-flag frame':<30} | "
          f"{gff_val:>10}", flush=True)
    print(f"  {'Intermittent':<30} | {'Per-frame accuracy':<30} | "
          f"{overall_accuracy:>9.1%}", flush=True)
    print(f"  {'Intermittent':<30} | {'Signal oscillations':<30} | "
          f"{mean_oscillations:>10.1f}", flush=True)
    print(f"  {'Recovery (fog->clean)':<30} | {'Recovery rate':<30} | "
          f"{recovery_rate:>9.1%}", flush=True)
    rec_val = f"{mean_recovery:.1f}" if mean_recovery is not None else "N/A"
    print(f"  {'Recovery (fog->clean)':<30} | {'Mean latency (frames)':<30} | "
          f"{rec_val:>10}", flush=True)

    # =================================================================
    # Save results
    # =================================================================
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'temporal_consistency',
        'timestamp': timestamp,
        'config': {
            'num_frames': NUM_FRAMES,
            'n_trials': N_TRIALS,
            'calibration_samples': CALIBRATION_SAMPLES,
            'conformal_alpha': alpha,
            'image_size': list(IMAGE_SIZE),
        },
        'calibration': {
            'threshold': threshold,
            'mean_dist': cal_mean_dist,
            'std_dist': cal_std_dist,
        },
        'analysis': analysis,
        'raw_results': {
            test_name: [
                {
                    'trial': tr['trial'],
                    'base_seed': tr['base_seed'],
                    'frames': tr['frames'],
                    'inter_frame_dists': tr['inter_frame_dists'],
                }
                for tr in trials
            ]
            for test_name, trials in all_test_results.items()
        },
    }

    output_path = os.path.join(
        RESULTS_DIR, f"temporal_consistency_{timestamp}.json",
    )
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
