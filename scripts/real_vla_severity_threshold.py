"""
Experiment 230: Severity-Dependent Detection Threshold
At what corruption severity does AUROC drop below 1.0?
Tests fog/night/noise at severity levels from 0.01 to 1.0.
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
        arr = arr * (1 - severity * 0.85)  # at severity=1, multiplier=0.15
    elif name == 'noise':
        arr = arr + rng.normal(0, severity * 40, arr.shape)
    elif name == 'blur':
        radius = max(0.1, severity * 5)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))
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
    print("Experiment 230: Severity-Dependent Detection")
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
    n_cal, n_test = 10, 8
    rng = np.random.default_rng(42)
    base_imgs = [make_driving_image() for _ in range(20)]
    corruption_types = ['fog', 'night', 'blur', 'noise']
    severities = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]

    # Calibrate
    print("\n--- Calibration ---")
    cal_embeds = {l: [] for l in layers}
    for i in range(n_cal):
        h = extract_hidden(model, processor, base_imgs[i], prompt, layers)
        for l in layers:
            cal_embeds[l].append(h[l])
    centroids = {l: np.mean(cal_embeds[l], axis=0) for l in layers}

    # Test ID
    id_scores = {l: [] for l in layers}
    for i in range(n_cal, n_cal + n_test):
        h = extract_hidden(model, processor, base_imgs[i], prompt, layers)
        for l in layers:
            id_scores[l].append(cosine_dist(h[l], centroids[l]))

    # Test each corruption at each severity
    results = {}
    for ctype in corruption_types:
        results[ctype] = {}
        print(f"\n--- {ctype} ---")
        for sev in severities:
            sev_results = {}
            for l in layers:
                ood_scores = []
                for i in range(n_test):
                    rng_local = np.random.default_rng(42 + i)
                    img = apply_corruption_severity(base_imgs[i], ctype, sev, rng_local)
                    h = extract_hidden(model, processor, img, prompt, [l])
                    ood_scores.append(cosine_dist(h[l], centroids[l]))

                auroc = round(compute_auroc(id_scores[l], ood_scores), 4)
                sev_results[f"L{l}"] = {
                    "auroc": auroc,
                    "mean_dist": round(float(np.mean(ood_scores)), 8),
                }
            results[ctype][f"sev_{sev}"] = sev_results
            print(f"  sev={sev:.2f}: L1={sev_results['L1']['auroc']} L3={sev_results['L3']['auroc']} | L3_dist={sev_results['L3']['mean_dist']:.8f}")

    output = {
        "experiment": "severity_detection",
        "experiment_number": 230,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "n_cal": n_cal,
        "n_test": n_test,
        "severities": severities,
        "corruption_types": corruption_types,
        "results": results,
    }

    out_path = f"/workspace/Vizuara-VLA-Research/experiments/severity_detection_{output['timestamp']}.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
