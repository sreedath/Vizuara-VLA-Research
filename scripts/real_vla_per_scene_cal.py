"""
Experiment 229: Per-Scene Calibration Strategy
Test per-scene centroids as a solution to the diverse scene degradation.
For each scene type, use scene-specific centroid for detection.
Compare: global centroid, per-scene centroid, nearest-scene centroid.
"""
import torch, json, numpy as np, os
from datetime import datetime
from PIL import Image, ImageFilter

def make_highway_image(idx, w=256, h=256):
    img = Image.new('RGB', (w, h))
    pixels = img.load()
    for y in range(h):
        for x in range(w):
            if y < h // 2:
                pixels[x, y] = (100 + idx, 150 + idx*2, 235 - idx*3)
            else:
                g = 80 + idx * 5
                pixels[x, y] = (g, g, g)
                if abs(x - w//2) < 3:
                    pixels[x, y] = (255, 255, 255)
    return img

def make_urban_image(idx, w=256, h=256):
    img = Image.new('RGB', (w, h))
    pixels = img.load()
    for y in range(h):
        for x in range(w):
            if y < h // 3:
                pixels[x, y] = (100 + idx*3, 140 + idx*2, 200 - idx)
            elif y < h // 2:
                pixels[x, y] = (139 + idx, 119 + idx*2, 101 + idx)
            else:
                pixels[x, y] = (60 + idx*3, 60 + idx*3, 60 + idx*3)
    return img

def make_rural_image(idx, w=256, h=256):
    img = Image.new('RGB', (w, h))
    pixels = img.load()
    for y in range(h):
        for x in range(w):
            if y < h // 3:
                pixels[x, y] = (100, 180 + idx*3, 255 - idx*2)
            elif y < h * 2 // 3:
                pixels[x, y] = (34 + idx*5, 139 + idx*2, 34 + idx*3)
            else:
                pixels[x, y] = (90 + idx*4, 90 + idx*4, 80 + idx*4)
    return img

def make_parking_image(idx, w=256, h=256):
    img = Image.new('RGB', (w, h))
    pixels = img.load()
    for y in range(h):
        for x in range(w):
            if y < h // 4:
                pixels[x, y] = (180 + idx*2, 200 + idx, 220 - idx)
            else:
                g = 120 + idx * 3
                pixels[x, y] = (g, g, g-10)
                if x % 50 < 3 and y > h // 3:
                    pixels[x, y] = (255, 255, 200)
    return img

def make_intersection_image(idx, w=256, h=256):
    img = Image.new('RGB', (w, h))
    pixels = img.load()
    for y in range(h):
        for x in range(w):
            if y < h // 2:
                pixels[x, y] = (120 + idx*2, 160 + idx, 200 - idx*2)
            else:
                g = 70 + idx * 4
                pixels[x, y] = (g, g, g)
            if y < 30 and 100 < x < 130:
                pixels[x, y] = (200 + idx*3, 50, 50)
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
    print("Experiment 229: Per-Scene Calibration Strategy")
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
    rng = np.random.default_rng(42)
    corruption_types = ['fog', 'night', 'blur', 'noise']

    scene_makers = {
        'highway': make_highway_image,
        'urban': make_urban_image,
        'rural': make_rural_image,
        'parking': make_parking_image,
        'intersection': make_intersection_image,
    }

    # Generate 6 images per scene: 3 cal, 3 test
    n_per_scene = 6
    scene_data = {}
    for scene_name, maker in scene_makers.items():
        imgs = [maker(i) for i in range(n_per_scene)]
        scene_data[scene_name] = {
            'imgs': imgs,
            'embeds': {l: [] for l in layers},
        }

    # Extract embeddings
    print("\n--- Extracting embeddings ---")
    for scene_name in scene_data:
        for i, img in enumerate(scene_data[scene_name]['imgs']):
            h = extract_hidden(model, processor, img, prompt, layers)
            for l in layers:
                scene_data[scene_name]['embeds'][l].append(h[l])
        print(f"  {scene_name}: done")

    # Strategy 1: Global centroid (all scenes mixed)
    print("\n--- Strategy 1: Global centroid ---")
    results = {"global": {}, "per_scene": {}, "nearest_scene": {}}
    for l in layers:
        all_cal = []
        all_test = []
        all_test_labels = []
        for scene_name in scene_data:
            all_cal.extend(scene_data[scene_name]['embeds'][l][:3])
            all_test.extend(scene_data[scene_name]['embeds'][l][3:])
            all_test_labels.extend([scene_name] * 3)

        global_centroid = np.mean(all_cal, axis=0)
        id_scores = [cosine_dist(e, global_centroid) for e in all_test]

        ood_all = []
        for ctype in corruption_types:
            for scene_name in scene_data:
                for i in range(3, 6):
                    img = apply_corruption(scene_data[scene_name]['imgs'][i], ctype, rng)
                    h = extract_hidden(model, processor, img, prompt, [l])
                    ood_all.append(cosine_dist(h[l], global_centroid))

        auroc = round(compute_auroc(id_scores, ood_all), 4)
        results["global"][f"L{l}"] = {
            "auroc": auroc,
            "id_mean": round(float(np.mean(id_scores)), 6),
            "ood_mean": round(float(np.mean(ood_all)), 6),
        }
        print(f"  L{l}: AUROC={auroc}")

    # Strategy 2: Per-scene centroid
    print("\n--- Strategy 2: Per-scene centroid ---")
    for l in layers:
        scene_centroids = {}
        for scene_name in scene_data:
            scene_centroids[scene_name] = np.mean(scene_data[scene_name]['embeds'][l][:3], axis=0)

        id_scores = []
        for scene_name in scene_data:
            for i in range(3, 6):
                d = cosine_dist(scene_data[scene_name]['embeds'][l][i], scene_centroids[scene_name])
                id_scores.append(d)

        ood_all = []
        for ctype in corruption_types:
            for scene_name in scene_data:
                for i in range(3, 6):
                    img = apply_corruption(scene_data[scene_name]['imgs'][i], ctype, rng)
                    h = extract_hidden(model, processor, img, prompt, [l])
                    d = cosine_dist(h[l], scene_centroids[scene_name])
                    ood_all.append(d)

        auroc = round(compute_auroc(id_scores, ood_all), 4)
        results["per_scene"][f"L{l}"] = {
            "auroc": auroc,
            "id_mean": round(float(np.mean(id_scores)), 6),
            "ood_mean": round(float(np.mean(ood_all)), 6),
        }
        print(f"  L{l}: AUROC={auroc}")

    # Strategy 3: Nearest-scene centroid (auto-select closest centroid)
    print("\n--- Strategy 3: Nearest-scene centroid ---")
    for l in layers:
        scene_centroids = {}
        for scene_name in scene_data:
            scene_centroids[scene_name] = np.mean(scene_data[scene_name]['embeds'][l][:3], axis=0)

        def nearest_centroid_dist(embed):
            dists = {s: cosine_dist(embed, c) for s, c in scene_centroids.items()}
            return min(dists.values())

        id_scores = []
        for scene_name in scene_data:
            for i in range(3, 6):
                id_scores.append(nearest_centroid_dist(scene_data[scene_name]['embeds'][l][i]))

        ood_all = []
        for ctype in corruption_types:
            for scene_name in scene_data:
                for i in range(3, 6):
                    img = apply_corruption(scene_data[scene_name]['imgs'][i], ctype, rng)
                    h = extract_hidden(model, processor, img, prompt, [l])
                    ood_all.append(nearest_centroid_dist(h[l]))

        auroc = round(compute_auroc(id_scores, ood_all), 4)
        results["nearest_scene"][f"L{l}"] = {
            "auroc": auroc,
            "id_mean": round(float(np.mean(id_scores)), 6),
            "ood_mean": round(float(np.mean(ood_all)), 6),
        }
        print(f"  L{l}: AUROC={auroc}")

    output = {
        "experiment": "per_scene_calibration",
        "experiment_number": 229,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "n_per_scene": n_per_scene,
        "scene_types": list(scene_makers.keys()),
        "results": results,
    }

    out_path = f"/workspace/Vizuara-VLA-Research/experiments/per_scene_cal_{output['timestamp']}.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
