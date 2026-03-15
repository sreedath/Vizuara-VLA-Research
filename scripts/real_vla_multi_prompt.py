"""
Experiment 211: Multi-Prompt Ensemble for OOD Detection
Does averaging detection scores across different text prompts improve robustness?
Tests 5 different prompts and their ensemble.
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
    print("Experiment 211: Multi-Prompt Ensemble")
    print("=" * 60)

    from transformers import AutoModelForVision2Seq, AutoProcessor
    print("Loading OpenVLA-7B...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompts = [
        "In: What action should the robot take to drive forward?\nOut:",
        "In: What action should the robot take to navigate safely?\nOut:",
        "In: What action should the robot take to stay in lane?\nOut:",
        "In: What action should the robot take to avoid obstacles?\nOut:",
        "In: What action should the robot take to reach the destination?\nOut:",
    ]
    prompt_names = ['drive_forward', 'navigate_safely', 'stay_lane', 'avoid_obstacles', 'reach_dest']

    layers = [1, 3]
    corruption_types = ['fog', 'night', 'blur', 'noise']
    n_cal, n_test = 10, 8
    rng = np.random.default_rng(42)
    base_imgs = [make_driving_image() for _ in range(20)]

    # Build per-prompt centroids
    print("\n--- Calibrating per-prompt centroids ---")
    centroids = {}  # centroids[prompt_idx][layer]
    for pi, prompt in enumerate(prompts):
        centroids[pi] = {}
        embeds = {l: [] for l in layers}
        for i in range(n_cal):
            h = extract_hidden(model, processor, base_imgs[i], prompt, layers)
            for l in layers:
                embeds[l].append(h[l])
        for l in layers:
            centroids[pi][l] = np.mean(embeds[l], axis=0)
        print(f"  Prompt {pi} ({prompt_names[pi]}): done")

    # Test detection per prompt and ensemble
    print("\n--- Testing per-prompt and ensemble detection ---")
    results = {}

    for l in layers:
        per_prompt_results = {}
        # Collect scores for ensemble
        all_id_scores = {pi: [] for pi in range(len(prompts))}
        all_ood_scores = {pi: [] for pi in range(len(prompts))}

        for pi, prompt in enumerate(prompts):
            id_scores = []
            for i in range(n_cal, n_cal + n_test):
                h = extract_hidden(model, processor, base_imgs[i], prompt, [l])
                id_scores.append(cosine_dist(h[l], centroids[pi][l]))
            all_id_scores[pi] = id_scores

            per_corr = {}
            ood_all = []
            for ctype in corruption_types:
                ood_scores = []
                for i in range(n_test):
                    img = apply_corruption(base_imgs[i], ctype, rng)
                    h = extract_hidden(model, processor, img, prompt, [l])
                    d = cosine_dist(h[l], centroids[pi][l])
                    ood_scores.append(d)
                    ood_all.append(d)
                per_corr[ctype] = round(compute_auroc(id_scores, ood_scores), 4)

            overall = round(compute_auroc(id_scores, ood_all), 4)
            all_ood_scores[pi] = ood_all
            per_prompt_results[prompt_names[pi]] = {
                "overall_auroc": overall,
                "per_corruption": per_corr,
                "id_mean": round(float(np.mean(id_scores)), 6),
                "ood_mean": round(float(np.mean(ood_all)), 6),
            }
            print(f"  L{l} {prompt_names[pi]}: AUROC={overall}")

        # Ensemble: average scores across prompts
        n_id = len(all_id_scores[0])
        n_ood = len(all_ood_scores[0])
        ensemble_id = [np.mean([all_id_scores[pi][j] for pi in range(len(prompts))]) for j in range(n_id)]
        ensemble_ood = [np.mean([all_ood_scores[pi][j] for pi in range(len(prompts))]) for j in range(n_ood)]
        ensemble_auroc = round(compute_auroc(ensemble_id, ensemble_ood), 4)

        # Cross-prompt centroid distance
        cross_dists = []
        for pi in range(len(prompts)):
            for pj in range(pi + 1, len(prompts)):
                cross_dists.append(cosine_dist(centroids[pi][l], centroids[pj][l]))

        per_prompt_results["ensemble_avg"] = {
            "overall_auroc": ensemble_auroc,
            "id_mean": round(float(np.mean(ensemble_id)), 6),
            "ood_mean": round(float(np.mean(ensemble_ood)), 6),
        }
        per_prompt_results["cross_prompt_centroid_dist"] = {
            "mean": round(float(np.mean(cross_dists)), 6),
            "max": round(float(np.max(cross_dists)), 6),
            "min": round(float(np.min(cross_dists)), 6),
        }
        results[f"L{l}"] = per_prompt_results
        print(f"  L{l} ENSEMBLE: AUROC={ensemble_auroc}")
        print(f"  L{l} cross-prompt centroid dist: mean={np.mean(cross_dists):.6f}")

    output = {
        "experiment": "multi_prompt_ensemble",
        "experiment_number": 211,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "n_cal": n_cal,
        "n_test": n_test,
        "prompts": prompt_names,
        "layers": [1, 3],
        "results": results,
    }

    out_path = f"/workspace/Vizuara-VLA-Research/experiments/multi_prompt_{output['timestamp']}.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
