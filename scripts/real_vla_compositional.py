"""
Experiment 217: Compositional Corruption Detection
Can we detect combinations of multiple simultaneous corruptions?
Tests all pairwise combinations and a triple combination.
"""
import torch, json, numpy as np, os
from datetime import datetime
from PIL import Image, ImageFilter
from itertools import combinations

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

def apply_single_corruption(img, name, rng):
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

def apply_multi_corruption(img, names, rng):
    """Apply multiple corruptions sequentially."""
    result = img
    for name in names:
        result = apply_single_corruption(result, name, rng)
    return result

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
    print("Experiment 217: Compositional Corruption Detection")
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
    single_types = ['fog', 'night', 'blur', 'noise']

    # Calibrate
    print("\n--- Calibrating ---")
    id_embeds = {l: [] for l in layers}
    for i in range(n_cal):
        h = extract_hidden(model, processor, base_imgs[i], prompt, layers)
        for l in layers:
            id_embeds[l].append(h[l])
    centroids = {l: np.mean(id_embeds[l], axis=0) for l in layers}

    id_scores = {l: [] for l in layers}
    for i in range(n_cal, n_cal + n_test):
        h = extract_hidden(model, processor, base_imgs[i], prompt, layers)
        for l in layers:
            id_scores[l].append(cosine_dist(h[l], centroids[l]))

    results = {}

    # Single corruptions (baseline)
    print("\n--- Single corruptions ---")
    for ctype in single_types:
        ood_scores = {l: [] for l in layers}
        for i in range(n_test):
            img = apply_single_corruption(base_imgs[i], ctype, rng)
            h = extract_hidden(model, processor, img, prompt, layers)
            for l in layers:
                ood_scores[l].append(cosine_dist(h[l], centroids[l]))
        aurocs = {f"L{l}": round(compute_auroc(id_scores[l], ood_scores[l]), 4) for l in layers}
        ood_means = {f"L{l}": round(float(np.mean(ood_scores[l])), 6) for l in layers}
        results[ctype] = {"auroc": aurocs, "ood_mean": ood_means}
        print(f"  {ctype}: {aurocs}")

    # Pairwise combinations
    print("\n--- Pairwise combinations ---")
    for c1, c2 in combinations(single_types, 2):
        combo = f"{c1}+{c2}"
        ood_scores = {l: [] for l in layers}
        for i in range(n_test):
            img = apply_multi_corruption(base_imgs[i], [c1, c2], rng)
            h = extract_hidden(model, processor, img, prompt, layers)
            for l in layers:
                ood_scores[l].append(cosine_dist(h[l], centroids[l]))
        aurocs = {f"L{l}": round(compute_auroc(id_scores[l], ood_scores[l]), 4) for l in layers}
        ood_means = {f"L{l}": round(float(np.mean(ood_scores[l])), 6) for l in layers}
        results[combo] = {"auroc": aurocs, "ood_mean": ood_means}
        print(f"  {combo}: {aurocs}")

    # Triple combinations
    print("\n--- Triple combinations ---")
    for c1, c2, c3 in combinations(single_types, 3):
        combo = f"{c1}+{c2}+{c3}"
        ood_scores = {l: [] for l in layers}
        for i in range(n_test):
            img = apply_multi_corruption(base_imgs[i], [c1, c2, c3], rng)
            h = extract_hidden(model, processor, img, prompt, layers)
            for l in layers:
                ood_scores[l].append(cosine_dist(h[l], centroids[l]))
        aurocs = {f"L{l}": round(compute_auroc(id_scores[l], ood_scores[l]), 4) for l in layers}
        ood_means = {f"L{l}": round(float(np.mean(ood_scores[l])), 6) for l in layers}
        results[combo] = {"auroc": aurocs, "ood_mean": ood_means}
        print(f"  {combo}: {aurocs}")

    # Quadruple combination
    print("\n--- All four corruptions ---")
    ood_scores = {l: [] for l in layers}
    for i in range(n_test):
        img = apply_multi_corruption(base_imgs[i], single_types, rng)
        h = extract_hidden(model, processor, img, prompt, layers)
        for l in layers:
            ood_scores[l].append(cosine_dist(h[l], centroids[l]))
    aurocs = {f"L{l}": round(compute_auroc(id_scores[l], ood_scores[l]), 4) for l in layers}
    ood_means = {f"L{l}": round(float(np.mean(ood_scores[l])), 6) for l in layers}
    results["all_four"] = {"auroc": aurocs, "ood_mean": ood_means}
    print(f"  all_four: {aurocs}")

    output = {
        "experiment": "compositional_corruption",
        "experiment_number": 217,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "n_cal": n_cal,
        "n_test": n_test,
        "layers": [1, 3],
        "results": results,
    }

    out_path = f"/workspace/Vizuara-VLA-Research/experiments/compositional_{output['timestamp']}.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
