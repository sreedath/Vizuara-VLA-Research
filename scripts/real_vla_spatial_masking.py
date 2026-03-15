"""
Experiment 213: Spatial Region Importance for OOD Detection
Which spatial regions of the input image contribute most to OOD detection?
Tests by masking different quadrants and measuring detection impact.
"""
import torch, json, numpy as np, os
from datetime import datetime
from PIL import Image, ImageFilter, ImageDraw

def make_driving_image(w=256, h=256):
    img = Image.new('RGB', (w, h))
    pixels = img.load()
    for y in range(h):
        for x in range(w):
            if y < h // 2:
                b = int(180 + 75 * (1 - y / (h / 2)))
                pixels[x, y] = (100, 150, b)
            else:
                g = int(80 + 40 * ((y - h/2) / (h/2)))
                pixels[x, y] = (g, g + 10, g - 10)
    return img

def apply_corruption(img, name, rng):
    arr = np.array(img, dtype=np.float32)
    if name == 'fog':
        fog = np.full_like(arr, 200)
        arr = arr * 0.4 + fog * 0.6
    elif name == 'night':
        arr = arr * 0.15
    elif name == 'blur':
        return img.filter(ImageFilter.GaussianBlur(radius=5))
    elif name == 'noise':
        arr = arr + rng.normal(0, 40, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def mask_region(img, region):
    """Mask a region of the image with gray (128,128,128)."""
    arr = np.array(img).copy()
    h, w = arr.shape[:2]
    if region == 'top_left':
        arr[:h//2, :w//2] = 128
    elif region == 'top_right':
        arr[:h//2, w//2:] = 128
    elif region == 'bottom_left':
        arr[h//2:, :w//2] = 128
    elif region == 'bottom_right':
        arr[h//2:, w//2:] = 128
    elif region == 'top_half':
        arr[:h//2] = 128
    elif region == 'bottom_half':
        arr[h//2:] = 128
    elif region == 'center':
        arr[h//4:3*h//4, w//4:3*w//4] = 128
    elif region == 'periphery':
        mask = np.ones_like(arr) * 128
        mask[h//4:3*h//4, w//4:3*w//4] = arr[h//4:3*h//4, w//4:3*w//4]
        arr = mask
    return Image.fromarray(arr.astype(np.uint8))

def extract_hidden(model, processor, image, prompt, layers):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return {l: fwd.hidden_states[l][0, -1, :].float().cpu().numpy() for l in layers}

def cosine_dist(a, b):
    return 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def compute_auroc(id_scores, ood_scores):
    id_scores = np.asarray(id_scores)
    ood_scores = np.asarray(ood_scores)
    n_id, n_ood = len(id_scores), len(ood_scores)
    if n_id == 0 or n_ood == 0:
        return 0.5
    count = sum(float(np.sum(o > id_scores) + 0.5 * np.sum(o == id_scores)) for o in ood_scores)
    return count / (n_id * n_ood)

def main():
    print("=" * 60)
    print("Experiment 213: Spatial Region Importance")
    print("=" * 60)

    from transformers import AutoModelForVision2Seq, AutoProcessor
    print("Loading OpenVLA-7B...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    layers = [1, 3]
    prompt = "In: What action should the robot take to drive forward?\nOut:"
    n_cal, n_test = 10, 8
    rng = np.random.default_rng(42)
    base_imgs = [make_driving_image() for _ in range(20)]
    corruption_types = ['fog', 'night', 'blur', 'noise']
    regions = ['top_left', 'top_right', 'bottom_left', 'bottom_right',
               'top_half', 'bottom_half', 'center', 'periphery']

    # Test: apply corruption ONLY to specific regions
    # This tells us which regions matter for detection
    print("\n--- Calibrating with clean images ---")
    id_embeds = {l: [] for l in layers}
    for i in range(n_cal):
        h = extract_hidden(model, processor, base_imgs[i], prompt, layers)
        for l in layers:
            id_embeds[l].append(h[l])
    centroids = {l: np.mean(id_embeds[l], axis=0) for l in layers}

    # Get ID test scores
    id_scores = {l: [] for l in layers}
    for i in range(n_cal, n_cal + n_test):
        h = extract_hidden(model, processor, base_imgs[i], prompt, layers)
        for l in layers:
            id_scores[l].append(cosine_dist(h[l], centroids[l]))

    # For each corruption type, apply corruption to specific regions only
    print("\n--- Testing regional corruption ---")
    results = {}
    for ctype in corruption_types:
        ctype_results = {}

        # Full corruption (baseline)
        full_ood = {l: [] for l in layers}
        for i in range(n_test):
            img = apply_corruption(base_imgs[i], ctype, rng)
            h = extract_hidden(model, processor, img, prompt, layers)
            for l in layers:
                full_ood[l].append(cosine_dist(h[l], centroids[l]))
        ctype_results['full'] = {f"L{l}": round(compute_auroc(id_scores[l], full_ood[l]), 4) for l in layers}
        print(f"\n  {ctype} full: {ctype_results['full']}")

        # Regional corruption
        for region in regions:
            reg_ood = {l: [] for l in layers}
            for i in range(n_test):
                # Apply corruption to full image, then restore clean pixels outside region
                corrupted = apply_corruption(base_imgs[i], ctype, rng)
                corr_arr = np.array(corrupted, dtype=np.float32)
                clean_arr = np.array(base_imgs[i], dtype=np.float32)
                h_img, w_img = corr_arr.shape[:2]

                # Start with clean, paste corrupted region
                result_arr = clean_arr.copy()
                if region == 'top_left':
                    result_arr[:h_img//2, :w_img//2] = corr_arr[:h_img//2, :w_img//2]
                elif region == 'top_right':
                    result_arr[:h_img//2, w_img//2:] = corr_arr[:h_img//2, w_img//2:]
                elif region == 'bottom_left':
                    result_arr[h_img//2:, :w_img//2] = corr_arr[h_img//2:, :w_img//2]
                elif region == 'bottom_right':
                    result_arr[h_img//2:, w_img//2:] = corr_arr[h_img//2:, w_img//2:]
                elif region == 'top_half':
                    result_arr[:h_img//2] = corr_arr[:h_img//2]
                elif region == 'bottom_half':
                    result_arr[h_img//2:] = corr_arr[h_img//2:]
                elif region == 'center':
                    result_arr[h_img//4:3*h_img//4, w_img//4:3*w_img//4] = corr_arr[h_img//4:3*h_img//4, w_img//4:3*w_img//4]
                elif region == 'periphery':
                    result_arr[:h_img//4] = corr_arr[:h_img//4]
                    result_arr[3*h_img//4:] = corr_arr[3*h_img//4:]
                    result_arr[:, :w_img//4] = corr_arr[:, :w_img//4]
                    result_arr[:, 3*w_img//4:] = corr_arr[:, 3*w_img//4:]

                img_partial = Image.fromarray(np.clip(result_arr, 0, 255).astype(np.uint8))
                h = extract_hidden(model, processor, img_partial, prompt, layers)
                for l in layers:
                    reg_ood[l].append(cosine_dist(h[l], centroids[l]))

            ctype_results[region] = {f"L{l}": round(compute_auroc(id_scores[l], reg_ood[l]), 4) for l in layers}
            print(f"  {ctype} {region:16s}: {ctype_results[region]}")

        results[ctype] = ctype_results

    output = {
        "experiment": "spatial_region_importance",
        "experiment_number": 213,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "n_cal": n_cal,
        "n_test": n_test,
        "regions": regions,
        "layers": [1, 3],
        "results": results,
    }

    out_path = f"/workspace/Vizuara-VLA-Research/experiments/spatial_masking_{output['timestamp']}.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
