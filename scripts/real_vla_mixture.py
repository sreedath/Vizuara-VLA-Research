"""
Experiment 240: Corruption Mixture Decomposition
When two corruptions are applied simultaneously, can the detector
identify the component types? Tests all pairwise mixtures.
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

def apply_corruption(img, name, rng, severity=1.0):
    arr = np.array(img, dtype=np.float32)
    if name == 'fog':
        fog = np.full_like(arr, 200)
        arr = arr * (1 - 0.6 * severity) + fog * (0.6 * severity)
    elif name == 'night':
        arr = arr * (1 - 0.85 * severity)
    elif name == 'noise':
        arr = arr + rng.normal(0, 30 * severity, arr.shape)
    elif name == 'blur':
        if severity > 0.1:
            return img.filter(ImageFilter.GaussianBlur(radius=5 * severity))
        return img
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def apply_double_corruption(img, c1, c2, rng, sev1=1.0, sev2=1.0):
    """Apply two corruptions sequentially."""
    img = apply_corruption(img, c1, rng, sev1)
    rng2 = np.random.default_rng(rng.integers(0, 2**31))
    img = apply_corruption(img, c2, rng2, sev2)
    return img

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
    print("Experiment 240: Corruption Mixture Decomposition")
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
    base_img = make_driving_image()
    centroid = extract_hidden(model, processor, base_img, prompt, layers)[3]

    corruption_types = ['fog', 'night', 'noise', 'blur']

    # Single-corruption centroids
    type_centroids = {}
    for ctype in corruption_types:
        rng = np.random.default_rng(42)
        img = apply_corruption(base_img, ctype, rng)
        type_centroids[ctype] = extract_hidden(model, processor, img, prompt, layers)[3]
        d = cosine_dist(type_centroids[ctype], centroid)
        print(f"  {ctype}: dist_to_clean={d:.6f}")

    # Pairwise mixtures
    print("\n--- Pairwise mixtures ---")
    import itertools
    results = {}
    for c1, c2 in itertools.combinations(corruption_types, 2):
        rng = np.random.default_rng(42)
        img = apply_double_corruption(base_img, c1, c2, rng)
        h = extract_hidden(model, processor, img, prompt, layers)
        emb = h[3]

        d_clean = cosine_dist(emb, centroid)
        # Distance to each single-corruption centroid
        d_types = {c: cosine_dist(emb, type_centroids[c]) for c in corruption_types}

        # Find closest single type
        closest = min(d_types, key=d_types.get)
        # Top-2 closest
        sorted_types = sorted(d_types.items(), key=lambda x: x[1])

        results[f"{c1}+{c2}"] = {
            "dist_to_clean": round(d_clean, 6),
            "dist_to_types": {c: round(d, 6) for c, d in d_types.items()},
            "closest_type": closest,
            "top2_types": [sorted_types[0][0], sorted_types[1][0]],
            "components_in_top2": bool(c1 in [sorted_types[0][0], sorted_types[1][0]] and
                                       c2 in [sorted_types[0][0], sorted_types[1][0]]),
        }
        print(f"  {c1}+{c2}: dist_clean={d_clean:.6f} closest={closest} top2={[sorted_types[0][0], sorted_types[1][0]]} components_match={results[f'{c1}+{c2}']['components_in_top2']}")

    # Summary: how often do top-2 nearest centroids match the actual components?
    total = len(results)
    matches = sum(1 for r in results.values() if r['components_in_top2'])
    print(f"\n  Top-2 component identification: {matches}/{total}")

    output = {
        "experiment": "mixture_decomposition",
        "experiment_number": 240,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "layer": 3,
        "corruption_types": corruption_types,
        "results": results,
        "top2_accuracy": round(matches / total, 4),
    }

    out_path = f"/workspace/Vizuara-VLA-Research/experiments/mixture_{output['timestamp']}.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
