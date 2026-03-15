#!/usr/bin/env python3
"""Experiment 263: Gradient-Free Sensitivity Analysis
Measures how much each input pixel region contributes to the OOD signal
by systematically masking regions and measuring cosine distance change.
Uses a grid of patches (no gradients needed - works with frozen model).
"""
import torch, json, numpy as np
from PIL import Image, ImageDraw
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
        from PIL import ImageFilter
        return img.filter(ImageFilter.GaussianBlur(radius=10 * severity))
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))

def mask_region(img, x, y, w, h, fill_value=128):
    """Mask a rectangular region with gray."""
    masked = img.copy()
    draw = ImageDraw.Draw(masked)
    draw.rectangle([x, y, x+w, y+h], fill=(fill_value, fill_value, fill_value))
    return masked

print("=" * 60)
print("Experiment 263: Gradient-Free Sensitivity Analysis")
print("=" * 60)

print("Loading OpenVLA-7B...")
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", torch_dtype=torch.bfloat16,
    device_map="auto", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
model.eval()

prompt = "In: What action should the robot take to pick up the object?\nOut:"
img = Image.new('RGB', (224, 224))
np.random.seed(42)
pixels = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
img = Image.fromarray(pixels)

# Grid: 7x7 = 49 patches of 32x32
grid_size = 7
patch_size = 32
corruptions = ['fog', 'night', 'noise', 'blur']

# Get clean embedding
h_clean = extract_hidden(model, processor, img, prompt)

results = {}

for ctype in corruptions:
    print(f"\n--- {ctype} ---")
    corrupted = apply_corruption(img, ctype)
    h_corrupt = extract_hidden(model, processor, corrupted, prompt)
    full_distance = cosine_distance(h_clean, h_corrupt)
    print(f"  Full corruption distance: {full_distance:.6f}")

    # For each patch: mask that region of the CORRUPTED image with the CLEAN patch
    # If distance drops, that patch was contributing to OOD signal
    sensitivity_map = np.zeros((grid_size, grid_size))

    for gi in range(grid_size):
        for gj in range(grid_size):
            y, x = gi * patch_size, gj * patch_size
            # Replace corrupted patch with clean patch
            restored = corrupted.copy()
            clean_crop = img.crop((x, y, x+patch_size, y+patch_size))
            restored.paste(clean_crop, (x, y))

            h_restored = extract_hidden(model, processor, restored, prompt)
            restored_distance = cosine_distance(h_clean, h_restored)

            # Sensitivity = how much distance drops when this patch is restored
            sensitivity_map[gi, gj] = full_distance - restored_distance

    # Normalize by full distance
    sensitivity_pct = sensitivity_map / (full_distance + 1e-10) * 100

    print(f"  Max sensitivity: {sensitivity_pct.max():.1f}%")
    print(f"  Min sensitivity: {sensitivity_pct.min():.1f}%")
    print(f"  Mean sensitivity: {sensitivity_pct.mean():.1f}%")
    print(f"  Std sensitivity: {sensitivity_pct.std():.1f}%")

    # Top-5 most sensitive patches
    flat_idx = np.argsort(sensitivity_pct.ravel())[::-1][:5]
    top5 = [(int(idx // grid_size), int(idx % grid_size), float(sensitivity_pct.ravel()[idx]))
            for idx in flat_idx]
    print(f"  Top-5 patches: {top5}")

    results[ctype] = {
        'full_distance': float(full_distance),
        'sensitivity_map': sensitivity_map.tolist(),
        'sensitivity_pct': sensitivity_pct.tolist(),
        'max_sensitivity': float(sensitivity_pct.max()),
        'min_sensitivity': float(sensitivity_pct.min()),
        'mean_sensitivity': float(sensitivity_pct.mean()),
        'std_sensitivity': float(sensitivity_pct.std()),
        'top5_patches': top5,
        'sum_sensitivity': float(sensitivity_pct.sum())
    }

# Also test: mask clean image patches (should have no effect)
print(f"\n--- clean masking control ---")
mask_distances = []
for gi in range(grid_size):
    for gj in range(grid_size):
        y, x = gi * patch_size, gj * patch_size
        masked = mask_region(img, x, y, patch_size, patch_size)
        h_masked = extract_hidden(model, processor, masked, prompt)
        d = cosine_distance(h_clean, h_masked)
        mask_distances.append(d)

results['clean_masking'] = {
    'mean_distance': float(np.mean(mask_distances)),
    'max_distance': float(np.max(mask_distances)),
    'std_distance': float(np.std(mask_distances))
}
print(f"  Mean mask distance: {np.mean(mask_distances):.6f}")
print(f"  Max mask distance: {np.max(mask_distances):.6f}")

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
out = {
    'experiment': 'gradient_free_sensitivity',
    'experiment_number': 263,
    'timestamp': ts,
    'grid_size': grid_size,
    'patch_size': patch_size,
    'results': results
}

path = f'/workspace/Vizuara-VLA-Research/experiments/sensitivity_{ts}.json'
with open(path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {path}")
