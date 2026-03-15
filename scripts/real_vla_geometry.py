"""
Experiment 244: Embedding Space Geometry
What are the angular relationships between corruption shift directions?
Are corruption shift vectors orthogonal, parallel, or somewhere in between?
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
    elif name == 'noise':
        arr = arr + rng.normal(0, 30, arr.shape)
    elif name == 'blur':
        return img.filter(ImageFilter.GaussianBlur(radius=5))
    elif name == 'snow':
        snow = rng.random(arr.shape) > 0.97
        arr[snow] = 255
        arr = arr * 0.7 + np.full_like(arr, 200) * 0.3
    elif name == 'rain':
        for _ in range(200):
            x = rng.integers(0, arr.shape[1])
            y_start = rng.integers(0, arr.shape[0] - 20)
            arr[y_start:y_start+20, x, :] = [180, 180, 220]
        arr = arr * 0.85
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def extract_hidden(model, processor, image, prompt, layers):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return {l: fwd.hidden_states[l][0, -1, :].float().cpu().numpy() for l in layers}

def main():
    print("=" * 60)
    print("Experiment 244: Embedding Space Geometry")
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

    corruption_types = ['fog', 'night', 'noise', 'blur', 'snow', 'rain']

    # Get corruption shift vectors: delta = corrupted - clean
    shifts = {}
    for ctype in corruption_types:
        rng = np.random.default_rng(42)
        img = apply_corruption(base_img, ctype, rng)
        emb = extract_hidden(model, processor, img, prompt, layers)[3]
        delta = emb - centroid
        shifts[ctype] = delta
        print(f"  {ctype}: ||delta|| = {np.linalg.norm(delta):.4f}")

    # Compute pairwise angles between shift vectors
    print("\n--- Pairwise angles (degrees) ---")
    angles = {}
    for i, c1 in enumerate(corruption_types):
        for j, c2 in enumerate(corruption_types):
            if i >= j:
                continue
            d1 = shifts[c1]
            d2 = shifts[c2]
            cos_angle = np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-10)
            angle_deg = float(np.degrees(np.arccos(np.clip(cos_angle, -1, 1))))
            angles[f"{c1}_vs_{c2}"] = {
                "cosine_similarity": round(float(cos_angle), 6),
                "angle_degrees": round(angle_deg, 2),
            }
            print(f"  {c1} vs {c2}: {angle_deg:.1f}° (cos={cos_angle:.4f})")

    # Summary statistics
    all_angles = [v["angle_degrees"] for v in angles.values()]
    print(f"\n  Mean angle: {np.mean(all_angles):.1f}°")
    print(f"  Min angle: {np.min(all_angles):.1f}°")
    print(f"  Max angle: {np.max(all_angles):.1f}°")

    # Effective dimensionality: how many dimensions does each shift vector use?
    print("\n--- Effective dimensionality ---")
    eff_dims = {}
    for ctype in corruption_types:
        delta = shifts[ctype]
        # Participation ratio: (sum |d_i|)^2 / sum |d_i|^2
        abs_d = np.abs(delta)
        pr = (abs_d.sum() ** 2) / (abs_d ** 2).sum()
        eff_dims[ctype] = round(float(pr), 1)
        print(f"  {ctype}: effective dim = {pr:.1f} / 4096")

    output = {
        "experiment": "geometry",
        "experiment_number": 244,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "layer": 3,
        "pairwise_angles": angles,
        "effective_dimensions": eff_dims,
        "summary": {
            "mean_angle": round(float(np.mean(all_angles)), 2),
            "min_angle": round(float(np.min(all_angles)), 2),
            "max_angle": round(float(np.max(all_angles)), 2),
        }
    }

    out_path = f"/workspace/Vizuara-VLA-Research/experiments/geometry_{output['timestamp']}.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
