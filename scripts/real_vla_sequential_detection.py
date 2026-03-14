"""
Sequential Inference Detection.

Simulates a driving sequence with frame-by-frame inference and
tests whether monitoring hidden state drift over consecutive frames
enables earlier OOD detection. Models gradual scene transitions
(highway → OOD) and abrupt changes.

Experiment 99 in the CalibDrive series.
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

def create_indoor(idx):
    rng = np.random.default_rng(idx * 5004)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:] = [200, 180, 160]
    img[SIZE[0]//2:, :] = [100, 80, 60]
    img[:SIZE[0]//3, SIZE[1]//3:2*SIZE[1]//3] = [150, 200, 255]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_noise(idx):
    rng = np.random.default_rng(idx * 5003)
    return rng.integers(0, 256, (*SIZE, 3), dtype=np.uint8)

def create_interpolated(idx, alpha):
    """Interpolate between highway and indoor at pixel level."""
    hw = create_highway(idx).astype(np.float32)
    indoor = create_indoor(idx).astype(np.float32)
    interp = (1 - alpha) * hw + alpha * indoor
    return np.clip(interp, 0, 255).astype(np.uint8)


def extract_hidden(model, processor, image, prompt):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    if hasattr(fwd, 'hidden_states') and fwd.hidden_states:
        return fwd.hidden_states[-1][0, -1, :].float().cpu().numpy()
    return None


def cosine_dist(a, b):
    return 1.0 - float(np.dot(a / (np.linalg.norm(a) + 1e-10),
                               b / (np.linalg.norm(b) + 1e-10)))


def main():
    print("=" * 70, flush=True)
    print("SEQUENTIAL INFERENCE DETECTION", flush=True)
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

    # Build calibration centroid
    print("\nBuilding calibration centroid...", flush=True)
    cal_hidden = []
    for i in range(15):
        h = extract_hidden(model, processor,
                          Image.fromarray(create_highway(i + 9000)), prompt)
        if h is not None:
            cal_hidden.append(h)
    centroid = np.mean(cal_hidden, axis=0)
    cal_scores = [cosine_dist(h, centroid) for h in cal_hidden]
    threshold = np.mean(cal_scores) + 3 * np.std(cal_scores)
    print(f"  Centroid from {len(cal_hidden)} samples, threshold={threshold:.4f}", flush=True)

    # Scenario 1: Gradual transition (highway → indoor over 20 frames)
    print("\n--- Scenario 1: Gradual transition ---", flush=True)
    gradual_scores = []
    gradual_alphas = np.linspace(0, 1, 20)
    for i, alpha in enumerate(gradual_alphas):
        img = create_interpolated(i + 1000, alpha)
        h = extract_hidden(model, processor, Image.fromarray(img), prompt)
        if h is not None:
            score = cosine_dist(h, centroid)
            detected = score > threshold
            gradual_scores.append({
                'frame': i,
                'alpha': float(alpha),
                'score': float(score),
                'detected': bool(detected),
            })
            status = "OOD" if detected else "ID"
            print(f"  Frame {i:2d} (α={alpha:.2f}): score={score:.4f} [{status}]", flush=True)

    # Find first detection frame
    first_detect_gradual = None
    for s in gradual_scores:
        if s['detected']:
            first_detect_gradual = s['frame']
            break

    # Scenario 2: Abrupt change (10 highway frames → 10 indoor frames)
    print("\n--- Scenario 2: Abrupt transition ---", flush=True)
    abrupt_scores = []
    for i in range(20):
        if i < 10:
            img = create_highway(i + 2000)
            scene = 'highway'
        else:
            img = create_indoor(i + 2000)
            scene = 'indoor'

        h = extract_hidden(model, processor, Image.fromarray(img), prompt)
        if h is not None:
            score = cosine_dist(h, centroid)
            detected = score > threshold
            abrupt_scores.append({
                'frame': i,
                'scene': scene,
                'score': float(score),
                'detected': bool(detected),
            })
            status = "OOD" if detected else "ID"
            print(f"  Frame {i:2d} ({scene:<8}): score={score:.4f} [{status}]", flush=True)

    first_detect_abrupt = None
    for s in abrupt_scores:
        if s['detected'] and s['scene'] == 'indoor':
            first_detect_abrupt = s['frame']
            break

    # Scenario 3: Intermittent noise (alternating highway/noise)
    print("\n--- Scenario 3: Intermittent noise ---", flush=True)
    intermittent_scores = []
    for i in range(20):
        if i % 3 == 0:
            img = create_noise(i + 3000)
            scene = 'noise'
        else:
            img = create_highway(i + 3000)
            scene = 'highway'

        h = extract_hidden(model, processor, Image.fromarray(img), prompt)
        if h is not None:
            score = cosine_dist(h, centroid)
            detected = score > threshold
            intermittent_scores.append({
                'frame': i,
                'scene': scene,
                'score': float(score),
                'detected': bool(detected),
            })
            status = "OOD" if detected else "ID"
            print(f"  Frame {i:2d} ({scene:<8}): score={score:.4f} [{status}]", flush=True)

    # Compute frame-to-frame drift
    print("\n--- Frame-to-frame drift analysis ---", flush=True)
    drift_gradual = []
    for i in range(1, len(gradual_scores)):
        # We need consecutive hidden states for this — approximate with score diff
        drift = abs(gradual_scores[i]['score'] - gradual_scores[i-1]['score'])
        drift_gradual.append(float(drift))
    print(f"  Gradual: mean drift={np.mean(drift_gradual):.4f}, max={np.max(drift_gradual):.4f}", flush=True)

    drift_abrupt = []
    for i in range(1, len(abrupt_scores)):
        drift = abs(abrupt_scores[i]['score'] - abrupt_scores[i-1]['score'])
        drift_abrupt.append(float(drift))
    print(f"  Abrupt: mean drift={np.mean(drift_abrupt):.4f}, max={np.max(drift_abrupt):.4f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'sequential_detection',
        'experiment_number': 99,
        'timestamp': timestamp,
        'threshold': float(threshold),
        'n_cal': len(cal_hidden),
        'gradual_transition': gradual_scores,
        'abrupt_transition': abrupt_scores,
        'intermittent_noise': intermittent_scores,
        'first_detect_gradual': first_detect_gradual,
        'first_detect_abrupt': first_detect_abrupt,
        'drift_gradual': drift_gradual,
        'drift_abrupt': drift_abrupt,
    }
    output_path = os.path.join(RESULTS_DIR, f"sequential_detection_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
