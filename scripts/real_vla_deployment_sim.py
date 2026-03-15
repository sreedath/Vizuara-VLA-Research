#!/usr/bin/env python3
"""Experiment 280: Deployment Simulation with Realistic Degradation
Simulates a realistic autonomous driving deployment scenario with:
- Temporal corruption sequences (clear->fog->rain->clear transitions)
- Gradual severity ramps (dawn-to-night)
- JPEG compression chain (camera->transmission->decoder)
- Combined real-world degradation (motion blur + noise + JPEG)
Tests whether the detector provides reliable real-time alerts.
"""
import torch, json, numpy as np
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor
from datetime import datetime
import io

def apply_corruption(img, ctype, severity=1.0):
    arr = np.array(img).astype(np.float32) / 255.0
    if ctype == 'fog':
        arr = arr * (1 - 0.6 * severity) + 0.6 * severity
    elif ctype == 'night':
        arr = arr * max(0.01, 1.0 - 0.95 * severity)
    elif ctype == 'noise':
        arr = arr + np.random.RandomState(42).randn(*arr.shape) * 0.3 * severity
        arr = np.clip(arr, 0, 1)
    elif ctype == 'blur':
        return img.filter(ImageFilter.GaussianBlur(radius=10 * severity))
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))

def extract_hidden(model, processor, image, prompt, layer=3):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return fwd.hidden_states[layer][0, -1, :].float().cpu().numpy()

print("=" * 60)
print("Experiment 280: Deployment Simulation")
print("=" * 60)

print("Loading OpenVLA-7B...")
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", torch_dtype=torch.bfloat16,
    device_map="auto", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
model.eval()

prompt = "In: What action should the robot take to pick up the object?\nOut:"
np.random.seed(42)
pixels = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
base_img = Image.fromarray(pixels)

centroid = extract_hidden(model, processor, base_img, prompt)

def compute_distance(embedding, centroid):
    return float(1.0 - np.dot(embedding, centroid) / (np.linalg.norm(embedding) * np.linalg.norm(centroid) + 1e-30))

# Scenario 1: Dawn-to-Night (30 frames)
print("\n=== SCENARIO 1: Dawn-to-Night (30 frames) ===")
dawn_night = []
for i in range(30):
    severity = i / 29.0
    frame = apply_corruption(base_img, 'night', severity)
    h = extract_hidden(model, processor, frame, prompt)
    d = compute_distance(h, centroid)
    dawn_night.append({'frame': i, 'severity': float(severity), 'distance': d})
    status = "SAFE" if d < 1e-4 else "WARNING" if d < 1e-3 else "OOD"
    print(f"  Frame {i:2d}: sev={severity:.2f}, d={d:.6f} [{status}]")

# Scenario 2: Fog Appears and Clears (40 frames)
print("\n=== SCENARIO 2: Fog Appears/Clears (40 frames) ===")
fog_cycle = []
for i in range(40):
    if i < 10:
        severity = 0
    elif i < 20:
        severity = (i - 10) / 10.0
    elif i < 30:
        severity = 1.0 - (i - 20) / 10.0
    else:
        severity = 0
    if severity > 0:
        frame = apply_corruption(base_img, 'fog', severity)
    else:
        frame = base_img
    h = extract_hidden(model, processor, frame, prompt)
    d = compute_distance(h, centroid)
    fog_cycle.append({'frame': i, 'severity': float(severity), 'distance': d})
    print(f"  Frame {i:2d}: sev={severity:.2f}, d={d:.6f}")

# Scenario 3: Rain = noise + blur + darkening
print("\n=== SCENARIO 3: Rain Simulation (20 frames) ===")
rain_sim = []
for i in range(20):
    severity = i / 19.0
    frame = base_img
    frame = apply_corruption(frame, 'noise', severity * 0.5)
    frame = apply_corruption(frame, 'blur', severity * 0.3)
    frame = apply_corruption(frame, 'night', severity * 0.2)
    h = extract_hidden(model, processor, frame, prompt)
    d = compute_distance(h, centroid)
    rain_sim.append({'frame': i, 'severity': float(severity), 'distance': d})
    print(f"  Frame {i:2d}: sev={severity:.2f}, d={d:.6f}")

# Scenario 4: JPEG quality sweep
print("\n=== SCENARIO 4: JPEG Quality Sweep ===")
jpeg_sweep = []
for q in [100, 95, 90, 80, 70, 60, 50, 40, 30, 20, 15, 10, 5, 3, 1]:
    buf = io.BytesIO()
    base_img.save(buf, format='JPEG', quality=q)
    buf.seek(0)
    frame = Image.open(buf).convert('RGB')
    h = extract_hidden(model, processor, frame, prompt)
    d = compute_distance(h, centroid)
    jpeg_sweep.append({'quality': q, 'distance': d})
    print(f"  q={q:3d}: d={d:.6f}")

# Scenario 5: Sensor failure pattern
print("\n=== SCENARIO 5: Sensor Failure Pattern ===")
failure_pattern = []
pattern = ['clean', 'noise', 'clean', 'blur', 'clean', 'clean', 'night',
           'night', 'clean', 'fog', 'fog', 'fog', 'clean', 'clean', 'clean',
           'noise', 'blur', 'clean', 'clean', 'clean']
for i, ctype in enumerate(pattern):
    if ctype == 'clean':
        frame = base_img
    else:
        frame = apply_corruption(base_img, ctype, 0.5)
    h = extract_hidden(model, processor, frame, prompt)
    d = compute_distance(h, centroid)
    failure_pattern.append({'frame': i, 'corruption': ctype, 'distance': d})
    detected = d > 1e-6
    correct = (ctype != 'clean') == detected
    print(f"  Frame {i:2d}: {ctype:6s} d={d:.6f} det={detected} {'OK' if correct else 'ERR'}")

tp = sum(1 for f in failure_pattern if f['corruption'] != 'clean' and f['distance'] > 1e-6)
fp = sum(1 for f in failure_pattern if f['corruption'] == 'clean' and f['distance'] > 1e-6)
fn = sum(1 for f in failure_pattern if f['corruption'] != 'clean' and f['distance'] <= 1e-6)
tn = sum(1 for f in failure_pattern if f['corruption'] == 'clean' and f['distance'] <= 1e-6)

print(f"\n  Detection: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
print(f"  Sensitivity: {tp/(tp+fn+1e-10):.3f}, Specificity: {tn/(tn+fp+1e-10):.3f}")

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
out = {
    'experiment': 'deployment_simulation',
    'experiment_number': 280,
    'timestamp': ts,
    'results': {
        'dawn_night': dawn_night,
        'fog_cycle': fog_cycle,
        'rain_simulation': rain_sim,
        'jpeg_sweep': jpeg_sweep,
        'failure_pattern': failure_pattern,
        'detection_metrics': {
            'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            'sensitivity': float(tp/(tp+fn)) if (tp+fn) > 0 else 1.0,
            'specificity': float(tn/(tn+fp)) if (tn+fp) > 0 else 1.0
        }
    }
}

path = f'/workspace/Vizuara-VLA-Research/experiments/deployment_{ts}.json'
with open(path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {path}")
