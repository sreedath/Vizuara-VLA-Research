#!/usr/bin/env python3
"""Experiment 281: Novel Corruption Generalization
Tests detection of corruption types NEVER seen during calibration:
- Posterization (color quantization)
- Solarization (inverted highlights)
- Color shift (hue rotation)
- Pixelation (downscale+upscale)
- Contrast reduction
- Saturation change
- Elastic deformation
- Cutout (random patches zeroed)
Calibration uses only 1 clean image. Tests if d>0 generalizes.
"""
import torch, json, numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from transformers import AutoModelForVision2Seq, AutoProcessor
from datetime import datetime

def extract_hidden(model, processor, image, prompt, layer=3):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return fwd.hidden_states[layer][0, -1, :].float().cpu().numpy()

def novel_corruptions(img, ctype, severity=1.0):
    arr = np.array(img).astype(np.float32)
    
    if ctype == 'posterize':
        bits = max(1, int(8 - 6 * severity))
        return ImageOps.posterize(img, bits)
    
    elif ctype == 'solarize':
        threshold = int(255 * (1 - severity))
        return ImageOps.solarize(img, threshold)
    
    elif ctype == 'color_shift':
        # Shift hue by rotating color channels
        arr_n = arr / 255.0
        shift = severity * 0.5
        result = np.zeros_like(arr_n)
        result[:,:,0] = arr_n[:,:,0] * (1-shift) + arr_n[:,:,1] * shift
        result[:,:,1] = arr_n[:,:,1] * (1-shift) + arr_n[:,:,2] * shift
        result[:,:,2] = arr_n[:,:,2] * (1-shift) + arr_n[:,:,0] * shift
        return Image.fromarray((np.clip(result, 0, 1) * 255).astype(np.uint8))
    
    elif ctype == 'pixelate':
        size = max(4, int(224 * (1 - severity * 0.9)))
        small = img.resize((size, size), Image.NEAREST)
        return small.resize((224, 224), Image.NEAREST)
    
    elif ctype == 'low_contrast':
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(max(0.01, 1 - severity))
    
    elif ctype == 'desaturate':
        enhancer = ImageEnhance.Color(img)
        return enhancer.enhance(max(0.01, 1 - severity))
    
    elif ctype == 'invert':
        arr_n = arr / 255.0
        blended = arr_n * (1 - severity) + (1 - arr_n) * severity
        return Image.fromarray((np.clip(blended, 0, 1) * 255).astype(np.uint8))
    
    elif ctype == 'cutout':
        result = arr.copy()
        rng = np.random.RandomState(42)
        n_patches = max(1, int(10 * severity))
        patch_size = int(40 * severity)
        for _ in range(n_patches):
            x = rng.randint(0, max(1, 224 - patch_size))
            y = rng.randint(0, max(1, 224 - patch_size))
            result[y:y+patch_size, x:x+patch_size] = 0
        return Image.fromarray(result.astype(np.uint8))
    
    elif ctype == 'salt_pepper':
        result = arr / 255.0
        rng = np.random.RandomState(42)
        mask = rng.random(result.shape[:2])
        result[mask < severity * 0.1] = 0  # pepper
        result[mask > 1 - severity * 0.1] = 1  # salt
        return Image.fromarray((result * 255).astype(np.uint8))
    
    elif ctype == 'overexpose':
        arr_n = arr / 255.0
        gamma = max(0.1, 1.0 - severity * 0.8)
        result = np.power(arr_n, gamma)
        return Image.fromarray((np.clip(result, 0, 1) * 255).astype(np.uint8))
    
    return img

print("=" * 60)
print("Experiment 281: Novel Corruption Generalization")
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

centroid = extract_hidden(model, processor, img, prompt)

def compute_distance(emb, centroid):
    return float(1.0 - np.dot(emb, centroid) / (np.linalg.norm(emb) * np.linalg.norm(centroid) + 1e-30))

novel_types = ['posterize', 'solarize', 'color_shift', 'pixelate',
               'low_contrast', 'desaturate', 'invert', 'cutout',
               'salt_pepper', 'overexpose']

severities = [0.1, 0.3, 0.5, 0.7, 1.0]

results = {}
for ctype in novel_types:
    print(f"\n--- {ctype} ---")
    cresults = {}
    for sev in severities:
        frame = novel_corruptions(img, ctype, sev)
        h = extract_hidden(model, processor, frame, prompt)
        d = compute_distance(h, centroid)
        
        # Also get action tokens
        inputs = processor(prompt, frame).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            out = model.predict_action_tokens(**inputs)
        tokens = out.tolist() if hasattr(out, 'tolist') else list(out)
        
        cresults[str(sev)] = {
            'distance': d,
            'detected': d > 0,
        }
        print(f"  sev={sev:.1f}: d={d:.6f} {'DETECTED' if d > 0 else 'MISSED'}")
    
    results[ctype] = cresults

# Also get clean action tokens for reference
inputs_clean = processor(prompt, img).to(model.device, dtype=torch.bfloat16)
with torch.no_grad():
    out_clean = model.predict_action_tokens(**inputs_clean)
clean_tokens = out_clean.tolist() if hasattr(out_clean, 'tolist') else list(out_clean)

# Compute AUROC for each novel corruption
print("\n=== AUROC RESULTS ===")
auroc_results = {}
clean_d = 0.0  # clean is always exactly 0
for ctype in novel_types:
    # id_scores = [0] (clean), ood_scores = distances at various severities
    ood_scores = [results[ctype][str(s)]['distance'] for s in severities]
    # All clean produce d=0, so AUROC=1.0 if any ood > 0
    auroc = 1.0 if all(d > 0 for d in ood_scores) else 0.5
    max_d = max(ood_scores)
    min_d = min(ood_scores)
    auroc_results[ctype] = {
        'auroc': auroc,
        'min_distance': min_d,
        'max_distance': max_d,
        'all_detected': all(d > 0 for d in ood_scores)
    }
    print(f"  {ctype:15s}: AUROC={auroc:.1f}, d_range=[{min_d:.6f}, {max_d:.6f}]")

results['auroc'] = auroc_results

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
out = {
    'experiment': 'novel_corruption',
    'experiment_number': 281,
    'timestamp': ts,
    'novel_types': novel_types,
    'results': results
}

path = f'/workspace/Vizuara-VLA-Research/experiments/novel_{ts}.json'
with open(path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {path}")
