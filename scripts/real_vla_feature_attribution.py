#!/usr/bin/env python3
"""Experiment 332: Feature Attribution Deep Dive (Real OpenVLA-7B)

Probes which image features drive the OOD embedding signal:
1. Edge-only vs texture-only corruption
2. Color channel isolation (R/G/B contribution)
3. Frequency band analysis (low/mid/high)
4. Spatial gradient analysis
5. Semantic vs statistical features
6. Feature importance via masking
"""

import json, time, os, sys
import numpy as np
import torch
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor

def extract_hidden(model, processor, image, prompt, layer=3):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return fwd.hidden_states[layer][0, -1, :].float().cpu().numpy()

def apply_corruption(image, ctype, severity=1.0):
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
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return 1.0 - dot / (na * nb)

def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return np.dot(a, b) / (na * nb)

def main():
    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    results = {}

    # Base image
    np.random.seed(42)
    pixels = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    base_img = Image.fromarray(pixels)
    clean_emb = extract_hidden(model, processor, base_img, prompt)

    # ========== 1. Color Channel Isolation ==========
    print("\n=== Color Channel Isolation ===")
    channel_results = {}

    for ch_name, ch_idx in [('red', 0), ('green', 1), ('blue', 2)]:
        arr = np.array(base_img).astype(np.float32)
        # Corrupt only one channel
        for ctype in ['fog', 'night', 'noise']:
            corrupted = arr.copy() / 255.0
            if ctype == 'fog':
                corrupted[:, :, ch_idx] = corrupted[:, :, ch_idx] * 0.4 + 0.6
            elif ctype == 'night':
                corrupted[:, :, ch_idx] = corrupted[:, :, ch_idx] * 0.05
            elif ctype == 'noise':
                rng = np.random.RandomState(42)
                noise = rng.randn(224, 224) * 0.3
                corrupted[:, :, ch_idx] = np.clip(corrupted[:, :, ch_idx] + noise, 0, 1)

            img = Image.fromarray((np.clip(corrupted, 0, 1) * 255).astype(np.uint8))
            emb = extract_hidden(model, processor, img, prompt)
            d = cosine_dist(clean_emb, emb)

            # Also get full corruption distance for ratio
            full_emb = extract_hidden(model, processor, apply_corruption(base_img, ctype, 1.0), prompt)
            full_d = cosine_dist(clean_emb, full_emb)

            key = f"{ch_name}_{ctype}"
            channel_results[key] = {
                'distance': float(d),
                'full_distance': float(full_d),
                'ratio': float(d / full_d) if full_d > 0 else 0,
            }
            print(f"  {ch_name}+{ctype}: d={d:.6f} (ratio={d/full_d:.3f} of full)")

    results['color_channels'] = channel_results

    # ========== 2. Frequency Band Analysis ==========
    print("\n=== Frequency Band Analysis ===")
    freq_results = {}

    arr = np.array(base_img).astype(np.float32)
    # Use numpy FFT on each channel
    for band_name, freq_range in [('low', (0, 20)), ('mid', (20, 60)), ('high', (60, 112))]:
        # Create frequency-band corruption
        corrupted = arr.copy()
        for ch in range(3):
            fft = np.fft.fft2(corrupted[:, :, ch])
            fft_shift = np.fft.fftshift(fft)

            # Create mask for this frequency band
            rows, cols = 224, 224
            crow, ccol = rows // 2, cols // 2
            mask = np.zeros((rows, cols))
            for r in range(rows):
                for c in range(cols):
                    dist = np.sqrt((r - crow) ** 2 + (c - ccol) ** 2)
                    if freq_range[0] <= dist <= freq_range[1]:
                        mask[r, c] = 1.0

            # Zero out this frequency band (remove those frequencies)
            fft_shift_modified = fft_shift.copy()
            fft_shift_modified[mask > 0] = 0

            # Inverse FFT
            fft_ishift = np.fft.ifftshift(fft_shift_modified)
            img_back = np.fft.ifft2(fft_ishift).real
            corrupted[:, :, ch] = np.clip(img_back, 0, 255)

        img = Image.fromarray(corrupted.astype(np.uint8))
        emb = extract_hidden(model, processor, img, prompt)
        d = cosine_dist(clean_emb, emb)
        freq_results[band_name] = {
            'distance': float(d),
            'freq_range': freq_range,
        }
        print(f"  Remove {band_name} ({freq_range}): d={d:.6f}")

    results['frequency_bands'] = freq_results

    # ========== 3. Edge vs Texture ==========
    print("\n=== Edge vs Texture ===")
    edge_texture_results = {}

    # Edge detection
    from PIL import ImageFilter
    edges = base_img.filter(ImageFilter.FIND_EDGES)
    edges_arr = np.array(edges).astype(np.float32)

    # Texture = original - edges (smooth regions)
    smooth = base_img.filter(ImageFilter.GaussianBlur(radius=5))

    # Corrupt only edges (high-freq components)
    arr_orig = np.array(base_img).astype(np.float32) / 255.0
    arr_smooth = np.array(smooth).astype(np.float32) / 255.0
    edge_component = arr_orig - arr_smooth  # high-freq

    # Add noise to edges only
    rng = np.random.RandomState(42)
    noise = rng.randn(*edge_component.shape) * 0.3

    edge_corrupted = arr_smooth + edge_component + noise * np.abs(edge_component) / (np.abs(edge_component).max() + 1e-10)
    edge_corrupted = np.clip(edge_corrupted, 0, 1)
    edge_img = Image.fromarray((edge_corrupted * 255).astype(np.uint8))
    edge_emb = extract_hidden(model, processor, edge_img, prompt)
    edge_d = cosine_dist(clean_emb, edge_emb)

    # Add noise to texture only (smooth regions)
    texture_corrupted = (arr_smooth + noise * 0.3) + edge_component
    texture_corrupted = np.clip(texture_corrupted, 0, 1)
    texture_img = Image.fromarray((texture_corrupted * 255).astype(np.uint8))
    texture_emb = extract_hidden(model, processor, texture_img, prompt)
    texture_d = cosine_dist(clean_emb, texture_emb)

    # Full noise for comparison
    full_noise_emb = extract_hidden(model, processor, apply_corruption(base_img, 'noise', 1.0), prompt)
    full_noise_d = cosine_dist(clean_emb, full_noise_emb)

    edge_texture_results = {
        'edge_noise_dist': float(edge_d),
        'texture_noise_dist': float(texture_d),
        'full_noise_dist': float(full_noise_d),
        'edge_ratio': float(edge_d / full_noise_d) if full_noise_d > 0 else 0,
        'texture_ratio': float(texture_d / full_noise_d) if full_noise_d > 0 else 0,
    }
    print(f"  Edge noise: d={edge_d:.6f} ({edge_d/full_noise_d:.3f}x)")
    print(f"  Texture noise: d={texture_d:.6f} ({texture_d/full_noise_d:.3f}x)")
    print(f"  Full noise: d={full_noise_d:.6f}")
    results['edge_vs_texture'] = edge_texture_results

    # ========== 4. Corruption Direction Alignment ==========
    print("\n=== Corruption Direction Alignment ===")
    direction_results = {}

    # Get corruption shift vectors
    ctypes = ['fog', 'night', 'noise', 'blur']
    shift_vectors = {}
    for ct in ctypes:
        ct_emb = extract_hidden(model, processor, apply_corruption(base_img, ct, 1.0), prompt)
        shift_vectors[ct] = ct_emb - clean_emb

    # Pairwise alignment of shift directions
    for i, ct1 in enumerate(ctypes):
        for ct2 in ctypes[i+1:]:
            sim = cosine_sim(shift_vectors[ct1], shift_vectors[ct2])
            angle = np.arccos(np.clip(sim, -1, 1)) * 180 / np.pi
            direction_results[f"{ct1}_vs_{ct2}"] = {
                'cosine_sim': float(sim),
                'angle_degrees': float(angle),
            }
            print(f"  {ct1} vs {ct2}: sim={sim:.4f}, angle={angle:.1f}°")

    # Check if shift directions are consistent across scenes
    print("\n  Cross-scene direction consistency:")
    scene_shift_consistency = {}
    for ct in ctypes:
        sims = []
        base_shift = shift_vectors[ct]
        for seed in [99, 123, 777]:
            rng = np.random.RandomState(seed)
            px = rng.randint(50, 200, (224, 224, 3), dtype=np.uint8)
            scene_img = Image.fromarray(px)
            scene_clean = extract_hidden(model, processor, scene_img, prompt)
            scene_corrupt = extract_hidden(model, processor, apply_corruption(scene_img, ct, 1.0), prompt)
            scene_shift = scene_corrupt - scene_clean
            sims.append(float(cosine_sim(base_shift, scene_shift)))

        scene_shift_consistency[ct] = {
            'cross_scene_sims': sims,
            'mean_sim': float(np.mean(sims)),
            'min_sim': float(min(sims)),
        }
        print(f"    {ct}: mean_sim={np.mean(sims):.4f}, min={min(sims):.4f}")

    results['direction_alignment'] = direction_results
    results['direction_consistency'] = scene_shift_consistency

    # ========== 5. Brightness vs Contrast vs Hue ==========
    print("\n=== Brightness/Contrast/Hue Decomposition ===")
    bch_results = {}

    arr = np.array(base_img).astype(np.float32) / 255.0

    # Pure brightness change (uniform additive)
    for delta in [-0.3, -0.1, 0.1, 0.3]:
        bright = np.clip(arr + delta, 0, 1)
        img = Image.fromarray((bright * 255).astype(np.uint8))
        emb = extract_hidden(model, processor, img, prompt)
        d = cosine_dist(clean_emb, emb)
        bch_results[f"brightness_{delta}"] = float(d)
        print(f"  Brightness {delta:+.1f}: d={d:.6f}")

    # Pure contrast change (multiplicative around mean)
    mean_val = arr.mean()
    for factor in [0.3, 0.5, 1.5, 2.0]:
        contrast = np.clip((arr - mean_val) * factor + mean_val, 0, 1)
        img = Image.fromarray((contrast * 255).astype(np.uint8))
        emb = extract_hidden(model, processor, img, prompt)
        d = cosine_dist(clean_emb, emb)
        bch_results[f"contrast_{factor}"] = float(d)
        print(f"  Contrast ×{factor}: d={d:.6f}")

    # Hue shift (rotate R→G→B)
    for shift in [1, 2]:
        hue_shifted = np.roll(arr, shift, axis=2)
        img = Image.fromarray((np.clip(hue_shifted, 0, 1) * 255).astype(np.uint8))
        emb = extract_hidden(model, processor, img, prompt)
        d = cosine_dist(clean_emb, emb)
        bch_results[f"hue_shift_{shift}"] = float(d)
        print(f"  Hue shift {shift}: d={d:.6f}")

    # Grayscale
    gray = np.mean(arr, axis=2, keepdims=True).repeat(3, axis=2)
    img = Image.fromarray((np.clip(gray, 0, 1) * 255).astype(np.uint8))
    emb = extract_hidden(model, processor, img, prompt)
    d = cosine_dist(clean_emb, emb)
    bch_results['grayscale'] = float(d)
    print(f"  Grayscale: d={d:.6f}")

    # Color inversion
    inverted = 1.0 - arr
    img = Image.fromarray((inverted * 255).astype(np.uint8))
    emb = extract_hidden(model, processor, img, prompt)
    d = cosine_dist(clean_emb, emb)
    bch_results['inverted'] = float(d)
    print(f"  Inverted: d={d:.6f}")

    results['brightness_contrast_hue'] = bch_results

    # ========== 6. Spatial Structure Attribution ==========
    print("\n=== Spatial Structure ===")
    spatial_results = {}

    arr = np.array(base_img).astype(np.float32)

    # Shuffle pixels (destroy all spatial structure)
    rng = np.random.RandomState(42)
    flat = arr.reshape(-1, 3).copy()
    rng.shuffle(flat)
    shuffled_img = Image.fromarray(flat.reshape(224, 224, 3).astype(np.uint8))
    shuf_emb = extract_hidden(model, processor, shuffled_img, prompt)
    shuf_d = cosine_dist(clean_emb, shuf_emb)
    spatial_results['pixel_shuffle'] = float(shuf_d)
    print(f"  Pixel shuffle: d={shuf_d:.6f}")

    # Shuffle 8x8 patches (destroy macro structure, keep micro)
    patch_arr = arr.copy()
    patches = []
    for i in range(0, 224, 8):
        for j in range(0, 224, 8):
            patches.append((i, j, patch_arr[i:i+8, j:j+8].copy()))
    rng2 = np.random.RandomState(42)
    rng2.shuffle(patches)
    idx = 0
    for i in range(0, 224, 8):
        for j in range(0, 224, 8):
            patch_arr[i:i+8, j:j+8] = patches[idx][2]
            idx += 1
    patch_img = Image.fromarray(patch_arr.astype(np.uint8))
    patch_emb = extract_hidden(model, processor, patch_img, prompt)
    patch_d = cosine_dist(clean_emb, patch_emb)
    spatial_results['patch_shuffle_8x8'] = float(patch_d)
    print(f"  8x8 patch shuffle: d={patch_d:.6f}")

    # Horizontal flip
    flip_img = base_img.transpose(Image.FLIP_LEFT_RIGHT)
    flip_emb = extract_hidden(model, processor, flip_img, prompt)
    flip_d = cosine_dist(clean_emb, flip_emb)
    spatial_results['horizontal_flip'] = float(flip_d)
    print(f"  Horizontal flip: d={flip_d:.6f}")

    # 90° rotation
    rot_img = base_img.rotate(90)
    rot_emb = extract_hidden(model, processor, rot_img, prompt)
    rot_d = cosine_dist(clean_emb, rot_emb)
    spatial_results['rotation_90'] = float(rot_d)
    print(f"  90° rotation: d={rot_d:.6f}")

    # Compare to fog distance
    fog_d = cosine_dist(clean_emb, extract_hidden(model, processor, apply_corruption(base_img, 'fog', 1.0), prompt))
    spatial_results['fog_reference'] = float(fog_d)
    print(f"  Fog reference: d={fog_d:.6f}")

    results['spatial_structure'] = spatial_results

    # ========== 7. Corruption Gradient (fine-grained severity) ==========
    print("\n=== Fine-Grained Severity Gradient ===")
    gradient_results = {}

    for ct in ['fog', 'night', 'noise', 'blur']:
        sevs = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
        dists = []
        for sev in sevs:
            img = apply_corruption(base_img, ct, sev)
            emb = extract_hidden(model, processor, img, prompt)
            d = cosine_dist(clean_emb, emb)
            dists.append(float(d))

        gradient_results[ct] = {
            'severities': sevs,
            'distances': dists,
            'min_detectable': float(sevs[next((i for i, d in enumerate(dists) if d > 0), -1)]) if any(d > 0 for d in dists) else None,
        }
        print(f"  {ct}: min_detect={gradient_results[ct]['min_detectable']}, max_d={max(dists):.6f}")

    results['severity_gradient'] = gradient_results

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/feature_attribution_{ts}.json"
    def convert(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)): return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return obj
    def recursive_convert(d):
        if isinstance(d, dict): return {k: recursive_convert(v) for k, v in d.items()}
        if isinstance(d, list): return [recursive_convert(x) for x in d]
        return convert(d)
    results = recursive_convert(results)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
