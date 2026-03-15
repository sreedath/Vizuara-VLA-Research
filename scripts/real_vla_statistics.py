"""
Experiment 247: Hidden State Statistics
Do the mean, variance, skewness, and kurtosis of hidden state vectors
change under corruption? Tests whether simple statistical moments
provide detection signals.
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
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def extract_hidden(model, processor, image, prompt, layers):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return {l: fwd.hidden_states[l][0, -1, :].float().cpu().numpy() for l in layers}

def compute_stats(vec):
    """Compute statistical moments of a vector."""
    return {
        "mean": round(float(np.mean(vec)), 6),
        "std": round(float(np.std(vec)), 6),
        "median": round(float(np.median(vec)), 6),
        "min": round(float(np.min(vec)), 6),
        "max": round(float(np.max(vec)), 6),
        "skewness": round(float(((vec - vec.mean()) ** 3).mean() / (vec.std() ** 3 + 1e-10)), 6),
        "kurtosis": round(float(((vec - vec.mean()) ** 4).mean() / (vec.std() ** 4 + 1e-10) - 3), 6),
        "pct_positive": round(float((vec > 0).mean()), 4),
        "pct_large": round(float((np.abs(vec) > vec.std() * 2).mean()), 4),
    }

def main():
    print("=" * 60)
    print("Experiment 247: Hidden State Statistics")
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

    # Clean statistics
    print("\n--- Clean ---")
    clean_h = extract_hidden(model, processor, base_img, prompt, layers)[3]
    clean_stats = compute_stats(clean_h)
    print(f"  {clean_stats}")

    # Corrupted statistics
    corruption_types = ['fog', 'night', 'noise', 'blur']
    results = {"clean": clean_stats}

    for ctype in corruption_types:
        print(f"\n--- {ctype} ---")
        rng = np.random.default_rng(42)
        img = apply_corruption(base_img, ctype, rng)
        h = extract_hidden(model, processor, img, prompt, layers)[3]
        stats = compute_stats(h)
        results[ctype] = stats
        print(f"  {stats}")

        # Changes from clean
        for key in ['mean', 'std', 'skewness', 'kurtosis']:
            change = stats[key] - clean_stats[key]
            pct = change / (abs(clean_stats[key]) + 1e-10) * 100
            print(f"  {key}: {change:+.6f} ({pct:+.2f}%)")

    output = {
        "experiment": "statistics",
        "experiment_number": 247,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "layer": 3,
        "results": results,
    }

    out_path = f"/workspace/Vizuara-VLA-Research/experiments/statistics_{output['timestamp']}.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
