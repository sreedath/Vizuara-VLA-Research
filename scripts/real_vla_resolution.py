"""
Experiment 242: Image Resolution Sensitivity
Does the detector work when images are resized to different resolutions?
Tests whether detection is robust to resolution changes (the processor resizes anyway).
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
    print("Experiment 242: Image Resolution Sensitivity")
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
    resolutions = [64, 128, 256, 512, 1024]
    corruption_types = ['fog', 'night', 'noise']

    results = {}
    for res in resolutions:
        print(f"\n=== Resolution: {res}x{res} ===")
        base_img = make_driving_image(w=res, h=res)
        centroid = extract_hidden(model, processor, base_img, prompt, layers)[3]

        # Cross-resolution: also test if centroid from 256 works for other resolutions
        res_results = {}
        for ctype in corruption_types:
            rng = np.random.default_rng(42)
            corr_img = apply_corruption(base_img, ctype, rng)
            h = extract_hidden(model, processor, corr_img, prompt, layers)
            d = cosine_dist(h[3], centroid)
            res_results[ctype] = round(d, 6)
            print(f"  {ctype}: dist={d:.6f}")

        results[f"{res}x{res}"] = {
            "distances": res_results,
            "all_detected": all(d > 0 for d in res_results.values()),
        }

    # Cross-resolution detection: calibrate at 256, test at other resolutions
    print("\n=== Cross-resolution transfer ===")
    base_256 = make_driving_image(w=256, h=256)
    centroid_256 = extract_hidden(model, processor, base_256, prompt, layers)[3]

    cross_results = {}
    for res in resolutions:
        base_res = make_driving_image(w=res, h=res)
        clean_d = cosine_dist(
            extract_hidden(model, processor, base_res, prompt, layers)[3],
            centroid_256
        )

        ood_dists = []
        for ctype in corruption_types:
            rng = np.random.default_rng(42)
            corr = apply_corruption(base_res, ctype, rng)
            d = cosine_dist(
                extract_hidden(model, processor, corr, prompt, layers)[3],
                centroid_256
            )
            ood_dists.append(d)

        auroc = compute_auroc([clean_d], ood_dists)
        cross_results[f"{res}x{res}"] = {
            "clean_dist_to_256_centroid": round(clean_d, 6),
            "mean_ood_dist": round(float(np.mean(ood_dists)), 6),
            "auroc_vs_256_centroid": round(auroc, 4),
        }
        print(f"  {res}x{res}: clean_d={clean_d:.6f} ood_d={np.mean(ood_dists):.6f} auroc={auroc:.4f}")

    output = {
        "experiment": "resolution_sensitivity",
        "experiment_number": 242,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "layer": 3,
        "resolutions": resolutions,
        "same_resolution_results": results,
        "cross_resolution_results": cross_results,
    }

    out_path = f"/workspace/Vizuara-VLA-Research/experiments/resolution_{output['timestamp']}.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
