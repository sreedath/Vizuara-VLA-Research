"""
Experiment 233: Severity Estimation from Cosine Distance
Can we predict corruption severity from a single cosine distance measurement?
Fit linear regression: severity ~ distance, evaluate R² and MAE.
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

def apply_corruption_severity(img, name, severity, rng):
    arr = np.array(img, dtype=np.float32)
    if name == 'fog':
        fog = np.full_like(arr, 200)
        arr = arr * (1 - severity) + fog * severity
    elif name == 'night':
        arr = arr * (1 - severity * 0.85)
    elif name == 'noise':
        arr = arr + rng.normal(0, severity * 40, arr.shape)
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
    print("Experiment 233: Severity Estimation")
    print("=" * 60)

    from transformers import AutoModelForVision2Seq, AutoProcessor
    print("Loading OpenVLA-7B...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    layers = [3]  # Focus on L3
    prompt = "In: What action should the robot take to drive forward?\nOut:"
    base_img = make_driving_image()
    rng = np.random.default_rng(42)

    # Get centroid
    centroid = extract_hidden(model, processor, base_img, prompt, layers)[3]

    # Sample severity levels densely
    severities = np.linspace(0.05, 1.0, 20)
    corruption_types = ['fog', 'night', 'noise']

    results = {}
    for ctype in corruption_types:
        print(f"\n--- {ctype} ---")
        distances = []
        for sev in severities:
            rng_local = np.random.default_rng(42)
            img = apply_corruption_severity(base_img, ctype, sev, rng_local)
            h = extract_hidden(model, processor, img, prompt, layers)
            d = cosine_dist(h[3], centroid)
            distances.append(d)
            print(f"  sev={sev:.2f}: dist={d:.8f}")

        distances = np.array(distances)

        # Linear fit: distance = a * severity + b
        A = np.vstack([severities, np.ones(len(severities))]).T
        slope, intercept = np.linalg.lstsq(A, distances, rcond=None)[0]
        predicted = slope * severities + intercept
        residuals = distances - predicted
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((distances - distances.mean())**2)
        r_squared = 1 - ss_res / (ss_tot + 1e-10)
        mae = float(np.mean(np.abs(residuals)))

        # Inverse: predict severity from distance
        predicted_sev = (distances - intercept) / (slope + 1e-10)
        sev_mae = float(np.mean(np.abs(severities - predicted_sev)))

        results[ctype] = {
            "distances": [round(float(d), 8) for d in distances],
            "severities": [round(float(s), 4) for s in severities],
            "linear_slope": round(float(slope), 8),
            "linear_intercept": round(float(intercept), 8),
            "r_squared": round(float(r_squared), 6),
            "distance_mae": round(mae, 8),
            "severity_mae": round(sev_mae, 4),
        }
        print(f"  Linear fit: d = {slope:.8f} * sev + {intercept:.8f}")
        print(f"  R² = {r_squared:.6f}, MAE(sev) = {sev_mae:.4f}")

    output = {
        "experiment": "severity_estimation",
        "experiment_number": 233,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "layer": 3,
        "results": results,
    }

    out_path = f"/workspace/Vizuara-VLA-Research/experiments/severity_est_{output['timestamp']}.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
