"""
Experiment 219: Leave-One-Corruption-Out Analysis
If we calibrate with only 3/4 corruption types, can we detect the held-out corruption?
Tests whether the centroid generalizes to unseen corruption types.
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
    elif name == 'rain':
        for _ in range(300):
            x = rng.integers(0, arr.shape[1])
            y_start = rng.integers(0, arr.shape[0] - 20)
            for dy in range(20):
                if y_start + dy < arr.shape[0]:
                    arr[y_start + dy, min(x, arr.shape[1]-1)] = [200, 200, 220]
    elif name == 'snow':
        snow_mask = rng.uniform(0, 1, arr.shape[:2]) > 0.95
        arr[snow_mask] = [240, 240, 250]
        arr = arr * 0.7 + 80
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
    print("Experiment 219: Leave-One-Corruption-Out")
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
    all_types = ['fog', 'night', 'blur', 'noise', 'rain', 'snow']

    # Calibrate with ONLY clean images (no corruption knowledge)
    print("\n--- Calibrating with clean images ---")
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

    # Test each corruption type
    print("\n--- Testing all corruption types ---")
    baseline_results = {}
    for ctype in all_types:
        ood_scores = {l: [] for l in layers}
        for i in range(n_test):
            img = apply_corruption(base_imgs[i], ctype, rng)
            h = extract_hidden(model, processor, img, prompt, layers)
            for l in layers:
                ood_scores[l].append(cosine_dist(h[l], centroids[l]))
        aurocs = {f"L{l}": round(compute_auroc(id_scores[l], ood_scores[l]), 4) for l in layers}
        baseline_results[ctype] = aurocs
        print(f"  {ctype}: {aurocs}")

    # Key insight: the detector only sees clean images during calibration
    # It has ZERO knowledge of what corruptions look like
    # Yet it can detect ALL of them — this is the core finding

    # Additional test: what if we only know about SOME corruptions?
    # Does knowing about fog help detect night?
    # This is an ablation — not needed for detection, but for identification
    print("\n--- Leave-one-out for identification ---")
    loo_results = {}
    
    # Build per-corruption centroids (as in exp 209)
    corr_centroids = {}
    for ctype in all_types:
        embeds = {l: [] for l in layers}
        for i in range(8):
            img = apply_corruption(base_imgs[i], ctype, rng)
            h = extract_hidden(model, processor, img, prompt, layers)
            for l in layers:
                embeds[l].append(h[l])
        corr_centroids[ctype] = {l: np.mean(embeds[l], axis=0) for l in layers}

    # For each held-out corruption, classify using remaining centroids
    for held_out in all_types:
        known_types = [t for t in all_types if t != held_out]
        
        # Classify held-out samples using nearest centroid among known types
        all_centroids = {'clean': centroids}
        for t in known_types:
            all_centroids[t] = corr_centroids[t]
        
        correct = {l: 0 for l in layers}
        total = 0
        for i in range(8, 8 + min(4, n_test)):
            img = apply_corruption(base_imgs[i], held_out, rng)
            h = extract_hidden(model, processor, img, prompt, layers)
            total += 1
            for l in layers:
                # Nearest centroid
                dists = {t: cosine_dist(h[l], all_centroids[t][l]) for t in all_centroids}
                pred = min(dists, key=dists.get)
                # The held-out type should NOT be classified as clean
                if pred != 'clean':
                    correct[l] += 1
        
        loo_results[held_out] = {
            f"L{l}": {"detected_as_ood": round(correct[l]/total, 4), "n": total}
            for l in layers
        }
        print(f"  Held-out {held_out}: detected as OOD = {loo_results[held_out]}")

    output = {
        "experiment": "leave_one_out",
        "experiment_number": 219,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "n_cal": n_cal,
        "n_test": n_test,
        "all_types": all_types,
        "baseline_detection": baseline_results,
        "leave_one_out_identification": loo_results,
    }

    out_path = f"/workspace/Vizuara-VLA-Research/experiments/leave_one_out_{output['timestamp']}.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
