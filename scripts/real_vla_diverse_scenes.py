"""
Experiment 228: Diverse Scene Robustness
Test OOD detection with visually diverse clean images: highway, urban, rural,
parking lot, intersection. Does the detector still work when clean images
are NOT identical?
"""
import torch, json, numpy as np, os
from datetime import datetime
from PIL import Image, ImageFilter

def make_highway_image(idx, w=256, h=256):
    """Highway: sky top, gray road bottom, white lane marker."""
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
    """Urban: buildings top, road bottom, crosswalk."""
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
    """Rural: sky, trees/fields, narrow road."""
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
    """Parking lot: flat gray surface, parking lines."""
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
    """Intersection: traffic light colors, multiple lanes."""
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
                pixels[x, y] = (200 + idx*3, 50, 50)  # red light
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
    print("Experiment 228: Diverse Scene Robustness")
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

    # Generate diverse images: 4 per scene type = 20 total
    n_per_scene = 4
    all_scenes = []
    all_labels = []
    for scene_name, maker in scene_makers.items():
        for i in range(n_per_scene):
            all_scenes.append(maker(i))
            all_labels.append(scene_name)

    # Extract all clean embeddings
    print("\n--- Extracting clean embeddings (diverse scenes) ---")
    all_embeds = {l: [] for l in layers}
    for i, img in enumerate(all_scenes):
        h = extract_hidden(model, processor, img, prompt, layers)
        for l in layers:
            all_embeds[l].append(h[l])
        if (i+1) % 5 == 0:
            print(f"  Clean: {i+1}/{len(all_scenes)}")

    # Compute cross-scene variance
    print("\n--- Cross-scene analysis ---")
    scene_results = {}
    for l in layers:
        all_matrix = np.array(all_embeds[l])
        global_centroid = all_matrix.mean(axis=0)

        # Per-scene centroids
        per_scene = {}
        for scene_name in scene_makers:
            idxs = [i for i, label in enumerate(all_labels) if label == scene_name]
            scene_embeds = [all_embeds[l][i] for i in idxs]
            scene_centroid = np.mean(scene_embeds, axis=0)
            dist_to_global = cosine_dist(scene_centroid, global_centroid)
            per_scene[scene_name] = round(dist_to_global, 6)

        # Cross-scene distances
        cross_dists = []
        for i in range(len(all_scenes)):
            d = cosine_dist(all_embeds[l][i], global_centroid)
            cross_dists.append(d)

        scene_results[f"L{l}"] = {
            "per_scene_to_global": per_scene,
            "cross_scene_mean_dist": round(float(np.mean(cross_dists)), 6),
            "cross_scene_max_dist": round(float(np.max(cross_dists)), 6),
            "cross_scene_std_dist": round(float(np.std(cross_dists)), 6),
        }
        print(f"  L{l}: cross-scene mean={np.mean(cross_dists):.6f} max={np.max(cross_dists):.6f}")
        for s, d in per_scene.items():
            print(f"    {s}: {d:.6f}")

    # OOD detection with mixed calibration (use 2 per scene = 10 cal, rest test)
    print("\n--- OOD detection with diverse calibration ---")
    n_cal_per_scene = 2
    cal_indices = []
    test_indices = []
    for scene_name in scene_makers:
        idxs = [i for i, label in enumerate(all_labels) if label == scene_name]
        cal_indices.extend(idxs[:n_cal_per_scene])
        test_indices.extend(idxs[n_cal_per_scene:])

    detection_results = {}
    for l in layers:
        centroid = np.mean([all_embeds[l][i] for i in cal_indices], axis=0)
        id_scores = [cosine_dist(all_embeds[l][i], centroid) for i in test_indices]

        per_corr = {}
        ood_all = []
        for ctype in corruption_types:
            ood_scores = []
            for i in test_indices:
                img = apply_corruption(all_scenes[i], ctype, rng)
                h = extract_hidden(model, processor, img, prompt, [l])
                ood_scores.append(cosine_dist(h[l], centroid))
                ood_all.append(ood_scores[-1])
            per_corr[ctype] = round(compute_auroc(id_scores, ood_scores), 4)

        overall = round(compute_auroc(id_scores, ood_all), 4)
        detection_results[f"L{l}"] = {
            "auroc": overall,
            "per_corruption": per_corr,
            "id_mean": round(float(np.mean(id_scores)), 6),
            "ood_mean": round(float(np.mean(ood_all)), 6),
        }
        print(f"  L{l}: AUROC={overall} | ID_mean={np.mean(id_scores):.6f} | OOD_mean={np.mean(ood_all):.6f}")

    output = {
        "experiment": "diverse_scene_robustness",
        "experiment_number": 228,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "n_per_scene": n_per_scene,
        "scene_types": list(scene_makers.keys()),
        "scene_results": scene_results,
        "detection_results": detection_results,
    }

    out_path = f"/workspace/Vizuara-VLA-Research/experiments/diverse_scenes_{output['timestamp']}.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
