"""
Experiment 221: Random Projection Robustness
Does OOD detection survive aggressive dimensionality reduction via random projection?
Tests Johnson-Lindenstrauss-style random projections from 4096D to {32, 64, 128, 256, 512, 1024, 2048}D.
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

def main():
    print("=" * 60)
    print("Experiment 221: Random Projection Robustness")
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

    # Random projection experiments
    proj_dims = [32, 64, 128, 256, 512, 1024, 2048]
    n_trials = 5  # average over 5 random matrices
    embed_dim = len(cal_embeds[layers[0]][0])  # 4096
    print(f"\nEmbedding dim: {embed_dim}")

    results = {}
    for l in layers:
        results[f"L{l}"] = {}
        centroids_full = np.mean(cal_embeds[l], axis=0)

        # Full dimension baseline
        id_scores_full = [cosine_dist(e, centroids_full) for e in test_embeds[l]]
        ood_all_full = []
        for ctype in corruption_types:
            for e in ood_embeds[ctype][l]:
                ood_all_full.append(cosine_dist(e, centroids_full))
        auroc_full = round(compute_auroc(id_scores_full, ood_all_full), 4)
        results[f"L{l}"]["full_4096"] = {"auroc": auroc_full, "id_mean": round(float(np.mean(id_scores_full)), 6), "ood_mean": round(float(np.mean(ood_all_full)), 6)}
        print(f"\n  L{l} full ({embed_dim}D): AUROC={auroc_full}")

        for k in proj_dims:
            trial_aurocs = []
            trial_id_means = []
            trial_ood_means = []
            for trial in range(n_trials):
                proj_rng = np.random.default_rng(42 + trial * 100 + k)
                # Gaussian random projection (JL-transform)
                R = proj_rng.standard_normal((embed_dim, k)) / np.sqrt(k)

                # Project calibration data
                cal_proj = [e @ R for e in cal_embeds[l]]
                centroid_proj = np.mean(cal_proj, axis=0)

                # Project test data
                test_proj = [e @ R for e in test_embeds[l]]
                id_scores = [cosine_dist(e, centroid_proj) for e in test_proj]

                ood_all = []
                for ctype in corruption_types:
                    for e in ood_embeds[ctype][l]:
                        ood_all.append(cosine_dist(e @ R, centroid_proj))

                auroc = compute_auroc(id_scores, ood_all)
                trial_aurocs.append(auroc)
                trial_id_means.append(float(np.mean(id_scores)))
                trial_ood_means.append(float(np.mean(ood_all)))

            mean_auroc = round(float(np.mean(trial_aurocs)), 4)
            std_auroc = round(float(np.std(trial_aurocs)), 4)
            results[f"L{l}"][f"k{k}"] = {
                "auroc_mean": mean_auroc,
                "auroc_std": std_auroc,
                "auroc_min": round(float(np.min(trial_aurocs)), 4),
                "auroc_max": round(float(np.max(trial_aurocs)), 4),
                "id_mean": round(float(np.mean(trial_id_means)), 6),
                "ood_mean": round(float(np.mean(trial_ood_means)), 6),
            }
            print(f"  L{l} k={k}: AUROC={mean_auroc}±{std_auroc}")

    output = {
        "experiment": "random_projection",
        "experiment_number": 221,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "n_cal": n_cal,
        "n_test": n_test,
        "n_trials": n_trials,
        "embed_dim": embed_dim,
        "proj_dims": proj_dims,
        "layers": layers,
        "results": results,
    }

    out_path = f"/workspace/Vizuara-VLA-Research/experiments/random_proj_{output['timestamp']}.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
