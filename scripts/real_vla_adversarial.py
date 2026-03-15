"""
Experiment 231: Adversarial Patch Detection
Can the cosine distance detector detect adversarial-style perturbations?
Tests random patches, constant-color patches, and targeted pixel perturbations.
"""
import torch, json, numpy as np, os
from datetime import datetime
from PIL import Image, ImageFilter

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

def apply_patch(img, patch_type, rng, size=32):
    arr = np.array(img).copy()
    h, w = arr.shape[:2]
    y_start = (h - size) // 2
    x_start = (w - size) // 2

    if patch_type == 'random_small':
        arr[y_start:y_start+size, x_start:x_start+size] = rng.integers(0, 256, (size, size, 3))
    elif patch_type == 'random_large':
        s = 64
        y_s = (h - s) // 2
        x_s = (w - s) // 2
        arr[y_s:y_s+s, x_s:x_s+s] = rng.integers(0, 256, (s, s, 3))
    elif patch_type == 'white_patch':
        arr[y_start:y_start+size, x_start:x_start+size] = 255
    elif patch_type == 'black_patch':
        arr[y_start:y_start+size, x_start:x_start+size] = 0
    elif patch_type == 'red_patch':
        arr[y_start:y_start+size, x_start:x_start+size] = [255, 0, 0]
    elif patch_type == 'pixel_noise_1pct':
        n_pixels = int(h * w * 0.01)
        for _ in range(n_pixels):
            py, px = rng.integers(0, h), rng.integers(0, w)
            arr[py, px] = rng.integers(0, 256, 3)
    elif patch_type == 'pixel_noise_5pct':
        n_pixels = int(h * w * 0.05)
        for _ in range(n_pixels):
            py, px = rng.integers(0, h), rng.integers(0, w)
            arr[py, px] = rng.integers(0, 256, 3)
    elif patch_type == 'stripe_horizontal':
        arr[h//2-5:h//2+5, :] = [255, 0, 0]
    elif patch_type == 'occlusion_center':
        s = 80
        y_s = (h - s) // 2
        x_s = (w - s) // 2
        arr[y_s:y_s+s, x_s:x_s+s] = 0

    return Image.fromarray(arr)

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
    print("Experiment 231: Adversarial Patch Detection")
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

    patch_types = ['random_small', 'random_large', 'white_patch', 'black_patch',
                   'red_patch', 'pixel_noise_1pct', 'pixel_noise_5pct',
                   'stripe_horizontal', 'occlusion_center']

    # Calibrate
    print("\n--- Calibration ---")
    cal_embeds = {l: [] for l in layers}
    for i in range(n_cal):
        h = extract_hidden(model, processor, base_imgs[i], prompt, layers)
        for l in layers:
            cal_embeds[l].append(h[l])
    centroids = {l: np.mean(cal_embeds[l], axis=0) for l in layers}

    # Test ID
    id_scores = {l: [] for l in layers}
    for i in range(n_cal, n_cal + n_test):
        h = extract_hidden(model, processor, base_imgs[i], prompt, layers)
        for l in layers:
            id_scores[l].append(cosine_dist(h[l], centroids[l]))

    # Test each patch type
    print("\n--- Patch tests ---")
    results = {}
    for ptype in patch_types:
        scores = {l: [] for l in layers}
        for i in range(n_test):
            rng_local = np.random.default_rng(42 + i)
            img = apply_patch(base_imgs[i], ptype, rng_local)
            h = extract_hidden(model, processor, img, prompt, layers)
            for l in layers:
                scores[l].append(cosine_dist(h[l], centroids[l]))

        results[ptype] = {}
        for l in layers:
            auroc = round(compute_auroc(id_scores[l], scores[l]), 4)
            results[ptype][f"L{l}"] = {
                "auroc": auroc,
                "mean_dist": round(float(np.mean(scores[l])), 6),
            }
        print(f"  {ptype}: L1={results[ptype]['L1']['auroc']} L3={results[ptype]['L3']['auroc']} | L3_dist={results[ptype]['L3']['mean_dist']:.6f}")

    output = {
        "experiment": "adversarial_patch",
        "experiment_number": 231,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "n_cal": n_cal,
        "n_test": n_test,
        "patch_types": patch_types,
        "results": results,
    }

    out_path = f"/workspace/Vizuara-VLA-Research/experiments/adversarial_patch_{output['timestamp']}.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
