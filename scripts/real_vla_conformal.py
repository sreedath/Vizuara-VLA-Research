"""
Experiment 226: Conformal Prediction Threshold Calibration
Apply split conformal prediction to set OOD detection thresholds with
guaranteed coverage. Tests alpha in {0.01, 0.05, 0.10, 0.20}.
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

def conformal_threshold(cal_scores, alpha):
    """Compute conformal prediction threshold at level alpha.
    Returns threshold q such that P(score <= q) >= 1 - alpha."""
    n = len(cal_scores)
    sorted_scores = np.sort(cal_scores)
    # Quantile level: ceil((n+1)*(1-alpha))/n
    level = np.ceil((n + 1) * (1 - alpha)) / n
    level = min(level, 1.0)
    idx = int(np.ceil(level * n)) - 1
    idx = min(idx, n - 1)
    return float(sorted_scores[idx])

def main():
    print("=" * 60)
    print("Experiment 226: Conformal Prediction Threshold Calibration")
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
    n_total = 30
    rng = np.random.default_rng(42)
    base_imgs = [make_driving_image() for _ in range(n_total)]
    corruption_types = ['fog', 'night', 'blur', 'noise']
    alphas = [0.01, 0.05, 0.10, 0.20]

    # Extract all embeddings
    print("\n--- Extracting embeddings ---")
    all_embeds = {l: [] for l in layers}
    for i in range(n_total):
        h = extract_hidden(model, processor, base_imgs[i], prompt, layers)
        for l in layers:
            all_embeds[l].append(h[l])
        if (i+1) % 10 == 0:
            print(f"  Clean: {i+1}/{n_total}")

    ood_embeds = {ctype: {l: [] for l in layers} for ctype in corruption_types}
    for ctype in corruption_types:
        for i in range(10):  # 10 OOD images per type
            img = apply_corruption(base_imgs[i], ctype, rng)
            h = extract_hidden(model, processor, img, prompt, layers)
            for l in layers:
                ood_embeds[ctype][l].append(h[l])
        print(f"  {ctype}: done")

    # Split: first 10 for centroid, next 10 for conformal calibration, last 10 for test
    n_centroid = 10
    n_conformal_cal = 10
    n_test = 10

    results = {}
    for l in layers:
        print(f"\n--- L{l} ---")
        centroid = np.mean(all_embeds[l][:n_centroid], axis=0)

        # Conformal calibration scores
        cal_scores = []
        for i in range(n_centroid, n_centroid + n_conformal_cal):
            cal_scores.append(cosine_dist(all_embeds[l][i], centroid))

        # Test ID scores
        test_id_scores = []
        for i in range(n_centroid + n_conformal_cal, n_total):
            test_id_scores.append(cosine_dist(all_embeds[l][i], centroid))

        layer_results = {
            "cal_scores": [round(s, 8) for s in cal_scores],
            "cal_mean": round(float(np.mean(cal_scores)), 8),
            "cal_max": round(float(np.max(cal_scores)), 8),
            "test_id_scores": [round(s, 8) for s in test_id_scores],
        }

        # Compute thresholds at each alpha
        for alpha in alphas:
            threshold = conformal_threshold(cal_scores, alpha)

            # Test ID: what fraction are below threshold?
            id_below = sum(1 for s in test_id_scores if s <= threshold) / len(test_id_scores)

            # Test OOD: what fraction are above threshold?
            ood_results = {}
            for ctype in corruption_types:
                ood_scores = [cosine_dist(e, centroid) for e in ood_embeds[ctype][l]]
                ood_above = sum(1 for s in ood_scores if s > threshold) / len(ood_scores)
                ood_results[ctype] = {
                    "detection_rate": round(ood_above, 4),
                    "mean_score": round(float(np.mean(ood_scores)), 8),
                }

            layer_results[f"alpha_{alpha}"] = {
                "threshold": round(threshold, 8),
                "id_acceptance_rate": round(id_below, 4),
                "ood_detection": ood_results,
            }
            print(f"  alpha={alpha}: threshold={threshold:.8f} | ID_accept={id_below:.2f} | OOD_detect: " +
                  " ".join(f"{c}={ood_results[c]['detection_rate']:.2f}" for c in corruption_types))

        results[f"L{l}"] = layer_results

    output = {
        "experiment": "conformal_prediction",
        "experiment_number": 226,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "n_centroid": n_centroid,
        "n_conformal_cal": n_conformal_cal,
        "n_test": n_test,
        "alphas": alphas,
        "layers": layers,
        "results": results,
    }

    out_path = f"/workspace/Vizuara-VLA-Research/experiments/conformal_{output['timestamp']}.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
