"""
Experiment 223: Calibration Set Size Sensitivity
How few calibration images are needed for reliable OOD detection?
Tests n_cal from 1 to 20 images.
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
    print("Experiment 223: Calibration Set Size Sensitivity")
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
    n_total = 30  # need enough for all tests
    rng = np.random.default_rng(42)
    base_imgs = [make_driving_image() for _ in range(n_total)]
    corruption_types = ['fog', 'night', 'blur', 'noise']

    # Extract ALL embeddings upfront
    print("\n--- Extracting all embeddings ---")
    all_embeds = {l: [] for l in layers}
    for i in range(n_total):
        h = extract_hidden(model, processor, base_imgs[i], prompt, layers)
        for l in layers:
            all_embeds[l].append(h[l])
        if (i+1) % 10 == 0:
            print(f"  Clean: {i+1}/{n_total}")

    n_test = 8
    # Always use the LAST 8 images for test (indices 22-29)
    test_start = n_total - n_test

    ood_embeds = {ctype: {l: [] for l in layers} for ctype in corruption_types}
    for ctype in corruption_types:
        for i in range(test_start, n_total):
            img = apply_corruption(base_imgs[i], ctype, rng)
            h = extract_hidden(model, processor, img, prompt, layers)
            for l in layers:
                ood_embeds[ctype][l].append(h[l])
        print(f"  {ctype}: done")

    # Test different calibration sizes
    cal_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]
    results = {}

    for n_cal in cal_sizes:
        print(f"\n--- n_cal={n_cal} ---")
        results[f"n{n_cal}"] = {}

        for l in layers:
            # Use first n_cal images for calibration
            cal = all_embeds[l][:n_cal]
            centroid = np.mean(cal, axis=0)

            # Test ID: use images after calibration but before test set
            # Use indices n_cal to n_cal+n_test (or test_start to n_total)
            test_id = all_embeds[l][test_start:n_total]
            id_scores = [cosine_dist(e, centroid) for e in test_id]

            per_corr = {}
            ood_all = []
            for ctype in corruption_types:
                ood_scores = [cosine_dist(e, ood_embeds[ctype][l][j]) for j, e in enumerate(ood_embeds[ctype][l])]
                # Fix: compute distance to centroid for each OOD embed
                ood_scores = [cosine_dist(e, centroid) for e in ood_embeds[ctype][l]]
                ood_all.extend(ood_scores)
                per_corr[ctype] = round(compute_auroc(id_scores, ood_scores), 4)

            overall = round(compute_auroc(id_scores, ood_all), 4)
            results[f"n{n_cal}"][f"L{l}"] = {
                "auroc": overall,
                "per_corruption": per_corr,
                "id_mean": round(float(np.mean(id_scores)), 6),
                "ood_mean": round(float(np.mean(ood_all)), 6),
            }
            print(f"  L{l}: AUROC={overall}")

    output = {
        "experiment": "calibration_size_sensitivity",
        "experiment_number": 223,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "n_test": n_test,
        "cal_sizes": cal_sizes,
        "layers": layers,
        "results": results,
    }

    out_path = f"/workspace/Vizuara-VLA-Research/experiments/cal_size_{output['timestamp']}.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
