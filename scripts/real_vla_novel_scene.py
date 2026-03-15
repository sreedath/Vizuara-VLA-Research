"""
Experiment 243: Novel Scene Detection
Can the detector distinguish corruptions from legitimate scene changes?
Tests with 5 different scene types to see if novel (but clean) scenes
trigger false alarms.
"""
import torch, json, numpy as np, os
from datetime import datetime
from PIL import Image, ImageFilter

def make_scene(scene_type, w=256, h=256):
    """Generate different scene types."""
    img = Image.new('RGB', (w, h))
    pixels = img.load()

    if scene_type == 'driving':
        for y in range(h):
            for x in range(w):
                if y < h // 2:
                    b = int(180 + 75 * (1 - y / (h / 2)))
                    pixels[x, y] = (100, 150, b)
                else:
                    g = int(80 + 40 * ((y - h/2) / (h/2)))
                    pixels[x, y] = (g, g + 10, g - 10)

    elif scene_type == 'indoor':
        for y in range(h):
            for x in range(w):
                if y < h * 0.3:
                    pixels[x, y] = (220, 210, 200)
                elif y < h * 0.7:
                    pixels[x, y] = (180, 170, 160)
                else:
                    pixels[x, y] = (120, 110, 100)

    elif scene_type == 'warehouse':
        for y in range(h):
            for x in range(w):
                if y < h * 0.4:
                    pixels[x, y] = (100, 100, 110)
                else:
                    g = int(150 + 20 * ((x % 50) / 50))
                    pixels[x, y] = (g, g-5, g-10)

    elif scene_type == 'outdoor_park':
        for y in range(h):
            for x in range(w):
                if y < h * 0.4:
                    b = int(200 + 55 * (1 - y / (h * 0.4)))
                    pixels[x, y] = (80, 130, b)
                elif y < h * 0.6:
                    pixels[x, y] = (50, 130, 50)
                else:
                    pixels[x, y] = (60, 100, 45)

    elif scene_type == 'kitchen':
        for y in range(h):
            for x in range(w):
                if y < h * 0.2:
                    pixels[x, y] = (240, 235, 225)
                elif y < h * 0.8:
                    pixels[x, y] = (200, 195, 185)
                else:
                    pixels[x, y] = (160, 155, 145)

    return img

def apply_corruption(img, name, rng):
    arr = np.array(img, dtype=np.float32)
    if name == 'fog':
        fog = np.full_like(arr, 200)
        arr = arr * 0.4 + fog * 0.6
    elif name == 'night':
        arr = arr * 0.15
    elif name == 'noise':
        arr = arr + rng.normal(0, 30, arr.shape)
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
    print("Experiment 243: Novel Scene Detection")
    print("=" * 60)

    from transformers import AutoModelForVision2Seq, AutoProcessor
    print("Loading OpenVLA-7B...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    layers = [3]
    prompt = "In: What action should the robot take to drive forward?\nOut:"

    scene_types = ['driving', 'indoor', 'warehouse', 'outdoor_park', 'kitchen']
    corruption_types = ['fog', 'night', 'noise']

    # Get centroid from driving scene
    driving_img = make_scene('driving')
    centroid = extract_hidden(model, processor, driving_img, prompt, layers)[3]

    # Distance of each clean scene to driving centroid
    print("\n--- Clean scene distances to driving centroid ---")
    scene_dists = {}
    for stype in scene_types:
        img = make_scene(stype)
        h = extract_hidden(model, processor, img, prompt, layers)
        d = cosine_dist(h[3], centroid)
        scene_dists[stype] = round(d, 6)
        print(f"  {stype}: {d:.6f}")

    # Distance of corrupted driving to driving centroid
    print("\n--- Corrupted driving distances ---")
    corr_dists = {}
    for ctype in corruption_types:
        rng = np.random.default_rng(42)
        img = apply_corruption(driving_img, ctype, rng)
        h = extract_hidden(model, processor, img, prompt, layers)
        d = cosine_dist(h[3], centroid)
        corr_dists[ctype] = round(d, 6)
        print(f"  {ctype}: {d:.6f}")

    # Key question: can we distinguish "novel scene" from "corruption"?
    # Novel scenes should be far from driving centroid but in a different direction
    print("\n--- Novel scene vs corruption discrimination ---")
    # AUROC: novel scenes as ID, corruptions as OOD
    novel_dists = [scene_dists[s] for s in scene_types if s != 'driving']
    corruption_dists_list = [corr_dists[c] for c in corruption_types]

    # Can novel scenes be separated from corruptions?
    auroc_novel_vs_corr = compute_auroc(novel_dists, corruption_dists_list)
    print(f"  AUROC (novel scenes vs corruptions): {auroc_novel_vs_corr:.4f}")

    # Per-scene centroid: detect corruption relative to each scene's own centroid
    print("\n--- Per-scene corruption detection ---")
    per_scene_results = {}
    for stype in scene_types:
        img = make_scene(stype)
        scene_centroid = extract_hidden(model, processor, img, prompt, layers)[3]

        id_scores = [0.0]  # Clean has dist=0
        ood_scores = []
        for ctype in corruption_types:
            rng = np.random.default_rng(42)
            corr = apply_corruption(img, ctype, rng)
            h = extract_hidden(model, processor, corr, prompt, layers)
            d = cosine_dist(h[3], scene_centroid)
            ood_scores.append(d)

        auroc = compute_auroc(id_scores, ood_scores)
        per_scene_results[stype] = {
            "corruption_dists": {c: round(d, 6) for c, d in zip(corruption_types, ood_scores)},
            "auroc": round(auroc, 4),
        }
        print(f"  {stype}: auroc={auroc:.4f} dists={[round(d, 6) for d in ood_scores]}")

    output = {
        "experiment": "novel_scene",
        "experiment_number": 243,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "layer": 3,
        "scene_types": scene_types,
        "scene_distances_to_driving": scene_dists,
        "corruption_distances": corr_dists,
        "auroc_novel_vs_corruption": round(auroc_novel_vs_corr, 4),
        "per_scene_results": per_scene_results,
    }

    out_path = f"/workspace/Vizuara-VLA-Research/experiments/novel_scene_{output['timestamp']}.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
