"""
Experiment 222: Distance Metric Comparison
Compare cosine distance, Euclidean distance, and Mahalanobis distance for OOD detection.
Uses pseudo-inverse covariance since n_cal < embed_dim.
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

def euclidean_dist(a, b):
    return float(np.linalg.norm(a - b))

def mahalanobis_dist(x, mean, cov_inv):
    diff = x - mean
    return float(np.sqrt(np.abs(diff @ cov_inv @ diff)))

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
    print("Experiment 222: Distance Metric Comparison")
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

    # Extract all embeddings
    print("\n--- Extracting embeddings ---")
    cal_embeds = {l: [] for l in layers}
    for i in range(n_cal):
        h = extract_hidden(model, processor, base_imgs[i], prompt, layers)
        for l in layers:
            cal_embeds[l].append(h[l])
        if (i+1) % 5 == 0:
            print(f"  Cal: {i+1}/{n_cal}")

    test_embeds = {l: [] for l in layers}
    for i in range(n_cal, n_cal + n_test):
        h = extract_hidden(model, processor, base_imgs[i], prompt, layers)
        for l in layers:
            test_embeds[l].append(h[l])

    ood_embeds = {ctype: {l: [] for l in layers} for ctype in corruption_types}
    for ctype in corruption_types:
        for i in range(n_test):
            img = apply_corruption(base_imgs[i], ctype, rng)
            h = extract_hidden(model, processor, img, prompt, layers)
            for l in layers:
                ood_embeds[ctype][l].append(h[l])
        print(f"  {ctype}: done")

    results = {}
    for l in layers:
        print(f"\n--- L{l} ---")
        centroid = np.mean(cal_embeds[l], axis=0)

        # Compute covariance pseudo-inverse for Mahalanobis
        cal_matrix = np.array(cal_embeds[l])
        centered = cal_matrix - centroid
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        k = min(len(S), n_cal - 1)
        S_inv = np.zeros_like(S)
        S_inv[:k] = 1.0 / (S[:k]**2 / (n_cal - 1) + 1e-8)
        cov_inv = Vt.T @ np.diag(S_inv) @ Vt

        layer_results = {}
        for metric_name, dist_fn in [("cosine", lambda x, c=centroid: cosine_dist(x, c)),
                                      ("euclidean", lambda x, c=centroid: euclidean_dist(x, c)),
                                      ("mahalanobis", lambda x, c=centroid, ci=cov_inv: mahalanobis_dist(x, c, ci))]:
            id_scores = [dist_fn(e) for e in test_embeds[l]]
            per_corr = {}
            ood_all = []
            for ctype in corruption_types:
                ood_scores = [dist_fn(e) for e in ood_embeds[ctype][l]]
                ood_all.extend(ood_scores)
                per_corr[ctype] = round(compute_auroc(id_scores, ood_scores), 4)

            overall = round(compute_auroc(id_scores, ood_all), 4)
            layer_results[metric_name] = {
                "auroc": overall,
                "per_corruption": per_corr,
                "id_mean": round(float(np.mean(id_scores)), 6),
                "ood_mean": round(float(np.mean(ood_all)), 6),
                "id_std": round(float(np.std(id_scores)), 6),
                "ood_std": round(float(np.std(ood_all)), 6),
            }
            print(f"  {metric_name}: AUROC={overall} | ID_mean={np.mean(id_scores):.6f} | OOD_mean={np.mean(ood_all):.6f}")

        results[f"L{l}"] = layer_results

    output = {
        "experiment": "distance_metric_comparison",
        "experiment_number": 222,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "n_cal": n_cal,
        "n_test": n_test,
        "layers": layers,
        "metrics": ["cosine", "euclidean", "mahalanobis"],
        "results": results,
    }

    out_path = f"/workspace/Vizuara-VLA-Research/experiments/distance_metrics_{output['timestamp']}.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
