"""
Experiment 245: Adversarial Evasion Analysis
Can a corrupted image be modified to evade the cosine distance detector?
Tests simple evasion strategies: brightness compensation, color adjustment,
and whether the detector can be fooled.
"""
import torch, json, numpy as np, os
from datetime import datetime
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

def main():
    print("=" * 60)
    print("Experiment 245: Adversarial Evasion Analysis")
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

    rng = np.random.default_rng(42)
    results = {}

    # Test 1: Night + brightness compensation
    print("\n--- Night + brightness compensation ---")
    night_img = apply_corruption(base_img, 'night', rng)
    night_dist = cosine_dist(extract_hidden(model, processor, night_img, prompt, layers)[3], centroid)
    print(f"  Night (raw): dist={night_dist:.6f}")

    # Try brightness multipliers to compensate
    for mult in [2.0, 4.0, 6.67, 8.0, 10.0]:
        compensated = ImageEnhance.Brightness(night_img).enhance(mult)
        comp_emb = extract_hidden(model, processor, compensated, prompt, layers)[3]
        d = cosine_dist(comp_emb, centroid)
        print(f"  Night + bright×{mult}: dist={d:.6f}")
        results[f"night_bright_{mult}"] = round(d, 6)

    # Test 2: Fog + contrast enhancement
    print("\n--- Fog + contrast compensation ---")
    fog_img = apply_corruption(base_img, 'fog', rng)
    fog_dist = cosine_dist(extract_hidden(model, processor, fog_img, prompt, layers)[3], centroid)
    print(f"  Fog (raw): dist={fog_dist:.6f}")

    for contrast in [1.5, 2.0, 3.0, 5.0]:
        compensated = ImageEnhance.Contrast(fog_img).enhance(contrast)
        comp_emb = extract_hidden(model, processor, compensated, prompt, layers)[3]
        d = cosine_dist(comp_emb, centroid)
        print(f"  Fog + contrast×{contrast}: dist={d:.6f}")
        results[f"fog_contrast_{contrast}"] = round(d, 6)

    # Test 3: Noise + blur to smooth out noise
    print("\n--- Noise + smoothing ---")
    noise_img = apply_corruption(base_img, 'noise', rng)
    noise_dist = cosine_dist(extract_hidden(model, processor, noise_img, prompt, layers)[3], centroid)
    print(f"  Noise (raw): dist={noise_dist:.6f}")

    for radius in [1, 2, 3, 5]:
        smoothed = noise_img.filter(ImageFilter.GaussianBlur(radius=radius))
        sm_emb = extract_hidden(model, processor, smoothed, prompt, layers)[3]
        d = cosine_dist(sm_emb, centroid)
        print(f"  Noise + blur(r={radius}): dist={d:.6f}")
        results[f"noise_blur_{radius}"] = round(d, 6)

    # Test 4: Pixel-level inversion (subtract corruption)
    print("\n--- Night pixel-level inversion ---")
    # Night = img * 0.15, so inversion = img / 0.15 (capped at 255)
    night_arr = np.array(night_img, dtype=np.float32)
    inverted_arr = night_arr / 0.15
    inverted = Image.fromarray(np.clip(inverted_arr, 0, 255).astype(np.uint8))
    inv_emb = extract_hidden(model, processor, inverted, prompt, layers)[3]
    d = cosine_dist(inv_emb, centroid)
    print(f"  Night inverted: dist={d:.6f}")
    results["night_inverted"] = round(d, 6)

    # Test 5: Fog pixel-level inversion
    print("\n--- Fog pixel-level inversion ---")
    fog_arr = np.array(fog_img, dtype=np.float32)
    base_arr = np.array(base_img, dtype=np.float32)
    # Fog = base * 0.4 + 200 * 0.6, so base = (fog - 120) / 0.4
    inverted_fog = (fog_arr - 120.0) / 0.4
    inverted_fog_img = Image.fromarray(np.clip(inverted_fog, 0, 255).astype(np.uint8))
    inv_fog_emb = extract_hidden(model, processor, inverted_fog_img, prompt, layers)[3]
    d = cosine_dist(inv_fog_emb, centroid)
    print(f"  Fog inverted: dist={d:.6f}")
    results["fog_inverted"] = round(d, 6)

    # Summary: minimum distance achieved by any evasion strategy
    results["night_raw"] = round(night_dist, 6)
    results["fog_raw"] = round(fog_dist, 6)
    results["noise_raw"] = round(noise_dist, 6)

    min_dist = min(results.values())
    min_key = min(results, key=results.get)
    print(f"\n  Minimum distance across all strategies: {min_dist:.6f} ({min_key})")
    print(f"  Clean distance: 0.000000")
    print(f"  Still detectable: {min_dist > 0}")

    output = {
        "experiment": "evasion",
        "experiment_number": 245,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "layer": 3,
        "results": results,
        "min_evasion_dist": round(min_dist, 6),
        "min_evasion_strategy": min_key,
    }

    out_path = f"/workspace/Vizuara-VLA-Research/experiments/evasion_{output['timestamp']}.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
