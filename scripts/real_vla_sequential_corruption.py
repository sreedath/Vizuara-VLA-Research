#!/usr/bin/env python3
"""Experiment 272: Sequential Corruption Detection
Tests the detector's ability to handle corruptions that appear and
disappear over time - simulating a driving scenario where the robot
encounters fog, exits it, then enters a night tunnel.
"""
import torch, json, numpy as np
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor
from datetime import datetime

def extract_hidden(model, processor, image, prompt, layer=3):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return fwd.hidden_states[layer][0, -1, :].float().cpu().numpy()

def cosine_distance(a, b):
    return 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

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

print("=" * 60)
print("Experiment 272: Sequential Corruption Detection")
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
img = Image.fromarray(pixels)

h_clean = extract_hidden(model, processor, img, prompt)

# Simulate a temporal sequence of corruption events
# Scenario: clean -> fog ramp up -> fog full -> fog ramp down -> clean ->
#           night ramp up -> night full -> night ramp down -> clean ->
#           blur sudden onset -> blur full -> clean
sequence = []

# Phase 1: Clean baseline (5 frames)
for i in range(5):
    sequence.append(('clean', 0.0))

# Phase 2: Fog ramp up (5 frames)
for i in range(5):
    sev = (i + 1) / 5.0
    sequence.append(('fog', sev))

# Phase 3: Full fog (5 frames)
for i in range(5):
    sequence.append(('fog', 1.0))

# Phase 4: Fog ramp down (5 frames)
for i in range(5):
    sev = 1.0 - (i + 1) / 5.0
    sequence.append(('fog', sev))

# Phase 5: Clean (3 frames)
for i in range(3):
    sequence.append(('clean', 0.0))

# Phase 6: Night ramp up (5 frames)
for i in range(5):
    sev = (i + 1) / 5.0
    sequence.append(('night', sev))

# Phase 7: Full night (5 frames)
for i in range(5):
    sequence.append(('night', 1.0))

# Phase 8: Night ramp down (5 frames)
for i in range(5):
    sev = 1.0 - (i + 1) / 5.0
    sequence.append(('night', sev))

# Phase 9: Clean (3 frames)
for i in range(3):
    sequence.append(('clean', 0.0))

# Phase 10: Sudden blur onset + sustained (5 frames)
for i in range(5):
    sequence.append(('blur', 1.0))

# Phase 11: Clean recovery (5 frames)
for i in range(5):
    sequence.append(('clean', 0.0))

print(f"Total frames: {len(sequence)}")

# Process each frame
frame_results = []
for frame_idx, (ctype, sev) in enumerate(sequence):
    if ctype == 'clean' or sev == 0:
        frame_img = img
    else:
        frame_img = apply_corruption(img, ctype, sev)

    h = extract_hidden(model, processor, frame_img, prompt)
    d = cosine_distance(h_clean, h)

    frame_results.append({
        'frame': frame_idx,
        'corruption': ctype if sev > 0 else 'clean',
        'severity': float(sev),
        'distance': float(d)
    })

    if frame_idx % 10 == 0:
        print(f"  Frame {frame_idx}: {ctype} sev={sev:.1f}, d={d:.6f}")

# Analyze detection with threshold
threshold = 1e-5  # very low threshold
detections = [1 if fr['distance'] > threshold else 0 for fr in frame_results]
true_labels = [1 if fr['severity'] > 0 else 0 for fr in frame_results]

tp = sum(1 for d, t in zip(detections, true_labels) if d == 1 and t == 1)
fp = sum(1 for d, t in zip(detections, true_labels) if d == 1 and t == 0)
fn = sum(1 for d, t in zip(detections, true_labels) if d == 0 and t == 1)
tn = sum(1 for d, t in zip(detections, true_labels) if d == 0 and t == 0)

print(f"\nThreshold={threshold:.1e}: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
print(f"Precision={tp/(tp+fp) if tp+fp > 0 else 0:.3f}, Recall={tp/(tp+fn) if tp+fn > 0 else 0:.3f}")

# Detection latency (first frame where distance exceeds threshold in each corruption episode)
phases = [
    ('fog_onset', 5, 10),
    ('night_onset', 23, 28),
    ('blur_onset', 41, 46),
]
for name, start, end in phases:
    for i in range(start, end):
        if frame_results[i]['distance'] > threshold:
            print(f"  {name}: detected at frame {i} (delay={i-start} frames)")
            break

results = {
    'sequence': frame_results,
    'threshold': float(threshold),
    'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
    'precision': float(tp/(tp+fp) if tp+fp > 0 else 0),
    'recall': float(tp/(tp+fn) if tp+fn > 0 else 0),
}

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
out = {
    'experiment': 'sequential_corruption',
    'experiment_number': 272,
    'timestamp': ts,
    'results': results
}

path = f'/workspace/Vizuara-VLA-Research/experiments/sequential_{ts}.json'
with open(path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {path}")
