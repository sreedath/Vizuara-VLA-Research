"""
Experiment 241: Occlusion Sensitivity Map
Which image regions contribute most to the OOD distance?
Systematically occlude patches and measure distance change.
Creates a spatial importance map for OOD detection.
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
    print("Experiment 241: Occlusion Sensitivity Map")
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
    rng = np.random.default_rng(42)
    centroid = extract_hidden(model, processor, base_img, prompt, layers)[3]

    # Get corrupted embeddings for reference
    fog_img = apply_corruption(base_img, 'fog', rng)
    night_img = apply_corruption(base_img, 'night', rng)
    fog_emb = extract_hidden(model, processor, fog_img, prompt, layers)[3]
    night_emb = extract_hidden(model, processor, night_img, prompt, layers)[3]
    fog_dist = cosine_dist(fog_emb, centroid)
    night_dist = cosine_dist(night_emb, centroid)
    print(f"Full fog dist: {fog_dist:.6f}")
    print(f"Full night dist: {night_dist:.6f}")

    # Occlusion sensitivity: slide a gray patch across the corrupted image
    patch_size = 32
    stride = 32
    h, w = 256, 256

    results = {}
    for ctype, corr_img, full_dist in [('fog', fog_img, fog_dist), ('night', night_img, night_dist)]:
        print(f"\n--- {ctype} occlusion map ---")
        grid_h = (h - patch_size) // stride + 1
        grid_w = (w - patch_size) // stride + 1
        heatmap = np.zeros((grid_h, grid_w))

        for gy in range(grid_h):
            for gx in range(grid_w):
                y0 = gy * stride
                x0 = gx * stride
                # Replace patch with gray (128, 128, 128)
                arr = np.array(corr_img).copy()
                arr[y0:y0+patch_size, x0:x0+patch_size] = 128
                occluded = Image.fromarray(arr)
                h_occ = extract_hidden(model, processor, occluded, prompt, layers)
                d_occ = cosine_dist(h_occ[3], centroid)
                # Importance = how much distance DECREASES when this patch is gray
                # (positive = this patch contributed to OOD distance)
                importance = full_dist - d_occ
                heatmap[gy, gx] = importance

            y_pos = gy * stride
            print(f"  Row {gy} (y={y_pos}): max_importance={heatmap[gy].max():.8f}")

        results[ctype] = {
            "heatmap": [[round(float(v), 8) for v in row] for row in heatmap],
            "full_dist": round(full_dist, 6),
            "max_importance": round(float(heatmap.max()), 8),
            "min_importance": round(float(heatmap.min()), 8),
            "max_location": [int(np.unravel_index(heatmap.argmax(), heatmap.shape)[0]),
                            int(np.unravel_index(heatmap.argmax(), heatmap.shape)[1])],
        }

    output = {
        "experiment": "occlusion_map",
        "experiment_number": 241,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "layer": 3,
        "patch_size": patch_size,
        "stride": stride,
        "results": results,
    }

    out_path = f"/workspace/Vizuara-VLA-Research/experiments/occlusion_map_{output['timestamp']}.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
