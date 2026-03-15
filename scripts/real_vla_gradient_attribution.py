#!/usr/bin/env python3
"""Experiment 306: Gradient-Based Input Attribution for OOD Detection
Analyzes which input regions/pixels most influence the embedding distance:
1. Pixel occlusion sensitivity map (systematic patch removal)
2. Per-corruption attribution comparison
3. Spatial frequency analysis (low vs high frequency corruption)
4. Input region importance (quadrants, center, periphery)
5. Corruption-specific vulnerable regions
"""

import torch
import numpy as np
import json
from datetime import datetime
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor
from scipy.spatial.distance import cosine

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

def extract_hidden(model, processor, image, prompt, layer=3):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return fwd.hidden_states[layer][0, -1, :].float().cpu().numpy()

def compute_auroc(id_scores, ood_scores):
    id_s = np.asarray(id_scores)
    ood_s = np.asarray(ood_scores)
    n_id, n_ood = len(id_s), len(ood_s)
    if n_id == 0 or n_ood == 0: return 0.5
    count = sum(float(np.sum(o > id_s) + 0.5 * np.sum(o == id_s)) for o in ood_s)
    return count / (n_id * n_ood)

def main():
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

    results = {
        "experiment": "gradient_attribution",
        "experiment_number": 306,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    corruptions = ['fog', 'night', 'blur', 'noise']
    clean_emb = extract_hidden(model, processor, base_img, prompt)

    # Part 1: Patch occlusion sensitivity map
    print("=== Part 1: Patch Occlusion Sensitivity ===")
    patch_size = 32  # 7x7 grid of 32x32 patches
    n_patches = 224 // patch_size  # 7

    occlusion_maps = {}
    for c in corruptions:
        print(f"  {c}...")
        corrupted = apply_corruption(base_img, c, 1.0)
        full_d = float(cosine(clean_emb, extract_hidden(model, processor, corrupted, prompt)))

        sensitivity = np.zeros((n_patches, n_patches))
        for pi in range(n_patches):
            for pj in range(n_patches):
                # Create corrupted image but restore this patch to clean
                mixed = np.array(corrupted).copy()
                clean_arr = np.array(base_img)
                y0, y1 = pi * patch_size, (pi + 1) * patch_size
                x0, x1 = pj * patch_size, (pj + 1) * patch_size
                mixed[y0:y1, x0:x1] = clean_arr[y0:y1, x0:x1]
                mixed_img = Image.fromarray(mixed)

                d = float(cosine(clean_emb, extract_hidden(model, processor, mixed_img, prompt)))
                # How much did restoring this patch reduce the distance?
                sensitivity[pi, pj] = full_d - d

        occlusion_maps[c] = {
            "full_distance": full_d,
            "sensitivity": sensitivity.tolist(),
            "max_patch": [int(x) for x in np.unravel_index(np.argmax(sensitivity), sensitivity.shape)],
            "max_reduction": float(sensitivity.max()),
            "max_reduction_pct": float(sensitivity.max() / full_d * 100) if full_d > 0 else 0,
            "sum_reductions": float(sensitivity.sum()),
            "superadditivity": float(sensitivity.sum() / full_d) if full_d > 0 else 0,
        }
        print(f"    full_d={full_d:.6f}, max_patch={occlusion_maps[c]['max_patch']}, "
              f"max_reduction={sensitivity.max():.6f} ({occlusion_maps[c]['max_reduction_pct']:.1f}%)")

    results["occlusion_maps"] = occlusion_maps

    # Part 2: Spatial frequency analysis
    print("\n=== Part 2: Spatial Frequency Analysis ===")
    freq_results = {}

    # Apply corruption only to low or high frequency components
    arr = np.array(base_img).astype(np.float32) / 255.0

    for c in ['fog', 'night', 'noise']:
        corrupted_full = np.array(apply_corruption(base_img, c, 1.0)).astype(np.float32) / 255.0
        diff = corrupted_full - arr

        # Separate into low and high frequency via Gaussian blur
        from scipy.ndimage import gaussian_filter
        diff_low = gaussian_filter(diff, sigma=5)
        diff_high = diff - diff_low

        # Apply only low-freq corruption
        low_only = np.clip(arr + diff_low, 0, 1)
        low_img = Image.fromarray((low_only * 255).astype(np.uint8))
        d_low = float(cosine(clean_emb, extract_hidden(model, processor, low_img, prompt)))

        # Apply only high-freq corruption
        high_only = np.clip(arr + diff_high, 0, 1)
        high_img = Image.fromarray((high_only * 255).astype(np.uint8))
        d_high = float(cosine(clean_emb, extract_hidden(model, processor, high_img, prompt)))

        # Full corruption distance
        d_full = float(cosine(clean_emb, extract_hidden(model, processor,
                       Image.fromarray((corrupted_full * 255).astype(np.uint8)), prompt)))

        freq_results[c] = {
            "d_full": d_full,
            "d_low_freq": d_low,
            "d_high_freq": d_high,
            "low_pct": float(d_low / d_full * 100) if d_full > 0 else 0,
            "high_pct": float(d_high / d_full * 100) if d_full > 0 else 0,
            "energy_low": float(np.sum(diff_low**2)),
            "energy_high": float(np.sum(diff_high**2)),
        }
        print(f"  {c}: low={d_low:.6f} ({freq_results[c]['low_pct']:.1f}%), "
              f"high={d_high:.6f} ({freq_results[c]['high_pct']:.1f}%), full={d_full:.6f}")

    results["spatial_frequency"] = freq_results

    # Part 3: Region importance (quadrants, center, periphery)
    print("\n=== Part 3: Region Importance ===")
    region_results = {}

    regions = {
        "top_left": (0, 112, 0, 112),
        "top_right": (0, 112, 112, 224),
        "bottom_left": (112, 224, 0, 112),
        "bottom_right": (112, 224, 112, 224),
        "center": (56, 168, 56, 168),
        "periphery_top": (0, 56, 0, 224),
        "periphery_bottom": (168, 224, 0, 224),
    }

    for c in corruptions:
        print(f"  {c}...")
        corrupted_arr = np.array(apply_corruption(base_img, c, 1.0))
        clean_arr = np.array(base_img)
        region_results[c] = {}

        for rname, (y0, y1, x0, x1) in regions.items():
            # Corrupt only this region
            partial = clean_arr.copy()
            partial[y0:y1, x0:x1] = corrupted_arr[y0:y1, x0:x1]
            partial_img = Image.fromarray(partial)

            d = float(cosine(clean_emb, extract_hidden(model, processor, partial_img, prompt)))
            region_results[c][rname] = {
                "distance": d,
                "detected": d > 0,
                "n_pixels": (y1 - y0) * (x1 - x0),
            }

        # Full distance for reference
        full_d = float(cosine(clean_emb, extract_hidden(model, processor,
                       apply_corruption(base_img, c, 1.0), prompt)))
        region_results[c]["full"] = {"distance": full_d}

        best_region = max(regions.keys(), key=lambda r: region_results[c][r]["distance"])
        print(f"    Most sensitive: {best_region} (d={region_results[c][best_region]['distance']:.6f}), "
              f"full={full_d:.6f}")

    results["region_importance"] = region_results

    # Part 4: Progressive corruption spread
    print("\n=== Part 4: Progressive Corruption Spread ===")
    spread_results = {}

    for c in corruptions:
        print(f"  {c}...")
        corrupted_arr = np.array(apply_corruption(base_img, c, 1.0))
        clean_arr = np.array(base_img)
        spread_results[c] = []

        # Corrupt increasing number of rows from top
        for n_rows in [0, 28, 56, 84, 112, 140, 168, 196, 224]:
            partial = clean_arr.copy()
            if n_rows > 0:
                partial[:n_rows, :, :] = corrupted_arr[:n_rows, :, :]
            partial_img = Image.fromarray(partial)
            d = float(cosine(clean_emb, extract_hidden(model, processor, partial_img, prompt)))
            spread_results[c].append({
                "n_rows": n_rows,
                "pct_corrupted": n_rows / 224 * 100,
                "distance": d,
                "detected": d > 0,
            })

        # Find minimum rows for detection
        first_detect = next((s for s in spread_results[c] if s["detected"]), None)
        if first_detect:
            print(f"    First detection at {first_detect['n_rows']} rows ({first_detect['pct_corrupted']:.0f}%)")
        else:
            print(f"    Not detected at any partial corruption level")

    results["progressive_spread"] = spread_results

    # Part 5: Channel-specific corruption
    print("\n=== Part 5: Channel-Specific Corruption ===")
    channel_results = {}

    for c in ['fog', 'night', 'noise']:
        print(f"  {c}...")
        corrupted_arr = np.array(apply_corruption(base_img, c, 1.0)).astype(np.float32)
        clean_arr = np.array(base_img).astype(np.float32)
        diff = corrupted_arr - clean_arr
        channel_results[c] = {}

        for ch_name, ch_idx in [("red", 0), ("green", 1), ("blue", 2)]:
            # Apply corruption to only one channel
            single_ch = clean_arr.copy()
            single_ch[:, :, ch_idx] = corrupted_arr[:, :, ch_idx]
            single_img = Image.fromarray(np.clip(single_ch, 0, 255).astype(np.uint8))
            d = float(cosine(clean_emb, extract_hidden(model, processor, single_img, prompt)))
            channel_results[c][ch_name] = {
                "distance": d,
                "detected": d > 0,
                "channel_energy": float(np.sum(diff[:, :, ch_idx]**2)),
            }

        # Full
        full_d = float(cosine(clean_emb, extract_hidden(model, processor,
                       apply_corruption(base_img, c, 1.0), prompt)))
        channel_results[c]["all_channels"] = {"distance": full_d}

        ch_strs = [f"{ch}={channel_results[c][ch]['distance']:.6f}" for ch in ["red", "green", "blue"]]
        print(f"    {', '.join(ch_strs)}, all={full_d:.6f}")

    results["channel_attribution"] = channel_results

    # Save
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(v) for v in obj]
        return obj

    ts = results["timestamp"]
    out_path = f"experiments/attribution_{ts}.json"
    with open(out_path, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
