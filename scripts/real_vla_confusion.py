"""
Experiment 209: Corruption Type Identification via Hidden-State Signatures
Can we identify WHICH type of corruption is present, not just detect OOD?
Uses per-corruption centroids and nearest-centroid classification.
"""
import torch, json, numpy as np, os, sys
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance

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
        alpha = 0.6
        arr = arr * (1 - alpha) + fog * alpha
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
        snow = rng.uniform(0, 1, arr.shape[:2])
        mask = snow > 0.95
        arr[mask] = [240, 240, 250]
        arr = arr * 0.7 + 80
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def extract_hidden(model, processor, image, prompt, layers):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return {l: fwd.hidden_states[l][0, -1, :].float().cpu().numpy() for l in layers}

def cosine_dist(a, b):
    return 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def main():
    print("=" * 60)
    print("Experiment 209: Corruption Type Identification")
    print("=" * 60)

    from transformers import AutoModelForVision2Seq, AutoProcessor
    print("Loading OpenVLA-7B...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    layers = [1, 3, 32]
    prompt = "In: What action should the robot take to drive forward?\nOut:"
    corruption_types = ['fog', 'night', 'blur', 'noise', 'rain', 'snow']
    n_cal = 8   # per corruption type for building corruption centroids
    n_test = 6  # per corruption type for testing
    n_id_cal = 10  # clean calibration

    rng = np.random.default_rng(42)
    base_imgs = [make_driving_image() for _ in range(30)]

    print("\n--- Extracting ID calibration embeddings ---")
    id_embeds = {l: [] for l in layers}
    for i in range(n_id_cal):
        h = extract_hidden(model, processor, base_imgs[i], prompt, layers)
        for l in layers:
            id_embeds[l].append(h[l])
        if (i + 1) % 5 == 0:
            print(f"  ID cal: {i+1}/{n_id_cal}")
    id_centroids = {l: np.mean(id_embeds[l], axis=0) for l in layers}

    print("\n--- Extracting corruption calibration embeddings ---")
    corr_centroids = {}
    for ctype in corruption_types:
        corr_embeds = {l: [] for l in layers}
        for i in range(n_cal):
            img = apply_corruption(base_imgs[i], ctype, rng)
            h = extract_hidden(model, processor, img, prompt, layers)
            for l in layers:
                corr_embeds[l].append(h[l])
        corr_centroids[ctype] = {l: np.mean(corr_embeds[l], axis=0) for l in layers}
        print(f"  {ctype}: done")

    print("\n--- Nearest-centroid classification on test set ---")
    # For each test corruption image, classify by finding nearest centroid
    # Centroids: ID + each corruption type
    all_types = ['clean'] + corruption_types
    centroids_map = {}
    for l in layers:
        centroids_map[l] = {'clean': id_centroids[l]}
        for ctype in corruption_types:
            centroids_map[l][ctype] = corr_centroids[ctype][l]

    # Build confusion matrix
    results = {}
    for l in layers:
        confusion = {true: {pred: 0 for pred in all_types} for true in all_types}

        # Test clean images
        for i in range(n_id_cal, n_id_cal + n_test):
            h = extract_hidden(model, processor, base_imgs[i], prompt, [l])
            dists = {t: cosine_dist(h[l], centroids_map[l][t]) for t in all_types}
            pred = min(dists, key=dists.get)
            confusion['clean'][pred] += 1

        # Test corruption images
        for ctype in corruption_types:
            for i in range(n_cal, n_cal + n_test):
                img = apply_corruption(base_imgs[i], ctype, rng)
                h = extract_hidden(model, processor, img, prompt, [l])
                dists = {t: cosine_dist(h[l], centroids_map[l][t]) for t in all_types}
                pred = min(dists, key=dists.get)
                confusion[ctype][pred] += 1

        # Compute accuracy
        total = 0
        correct = 0
        for t in all_types:
            for p in all_types:
                total += confusion[t][p]
                if t == p:
                    correct += confusion[t][p]

        acc = correct / total if total > 0 else 0
        per_class_acc = {}
        for t in all_types:
            row_total = sum(confusion[t].values())
            per_class_acc[t] = confusion[t][t] / row_total if row_total > 0 else 0

        results[f"L{l}"] = {
            "confusion": confusion,
            "overall_accuracy": round(acc, 4),
            "per_class_accuracy": {k: round(v, 4) for k, v in per_class_acc.items()},
        }
        print(f"  L{l} accuracy: {acc:.4f}")
        for t in all_types:
            row = [confusion[t][p] for p in all_types]
            print(f"    {t:8s}: {row}")

    # Also test: binary OOD detection (clean vs any corruption) using just ID centroid
    print("\n--- Binary detection comparison ---")
    binary = {}
    for l in layers:
        id_scores = []
        ood_scores = []
        # clean test
        for i in range(n_id_cal, n_id_cal + n_test):
            h = extract_hidden(model, processor, base_imgs[i], prompt, [l])
            id_scores.append(cosine_dist(h[l], id_centroids[l]))
        # ood test
        for ctype in corruption_types:
            for i in range(n_cal, n_cal + n_test):
                img = apply_corruption(base_imgs[i], ctype, rng)
                h = extract_hidden(model, processor, img, prompt, [l])
                ood_scores.append(cosine_dist(h[l], id_centroids[l]))

        # AUROC
        id_arr = np.array(id_scores)
        ood_arr = np.array(ood_scores)
        count = sum(float(np.sum(o > id_arr) + 0.5 * np.sum(o == id_arr)) for o in ood_arr)
        auroc = count / (len(id_arr) * len(ood_arr))
        binary[f"L{l}"] = {"auroc": round(auroc, 4), "id_mean": round(float(np.mean(id_arr)), 6), "ood_mean": round(float(np.mean(ood_arr)), 6)}
        print(f"  L{l} binary AUROC: {auroc:.4f}")

    # Inter-corruption distances
    print("\n--- Inter-corruption centroid distances ---")
    inter_dists = {}
    for l in layers:
        dists = {}
        for i, t1 in enumerate(all_types):
            for t2 in all_types[i+1:]:
                d = cosine_dist(centroids_map[l][t1], centroids_map[l][t2])
                dists[f"{t1}_vs_{t2}"] = round(d, 6)
        inter_dists[f"L{l}"] = dists
        # Find most confusable pair
        min_pair = min(dists, key=dists.get)
        print(f"  L{l}: most similar pair = {min_pair} (d={dists[min_pair]:.6f})")

    output = {
        "experiment": "corruption_identification",
        "experiment_number": 209,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "n_id_cal": n_id_cal,
        "n_corr_cal": n_cal,
        "n_test": n_test,
        "corruption_types": corruption_types,
        "classification": results,
        "binary_detection": binary,
        "inter_corruption_distances": inter_dists,
    }

    out_path = f"/workspace/Vizuara-VLA-Research/experiments/confusion_matrix_{output['timestamp']}.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
