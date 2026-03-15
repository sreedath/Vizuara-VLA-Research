"""
Experiment 236: Embedding Norm Analysis
Do OOD inputs produce embeddings with different L2 norms?
If norm changes under corruption, it provides a complementary detection signal
(direction-independent, unlike cosine distance).
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
    elif name == 'snow':
        snow = rng.random(arr.shape) > 0.97
        arr[snow] = 255
        arr = arr * 0.7 + np.full_like(arr, 200) * 0.3
    elif name == 'rain':
        for _ in range(200):
            x = rng.integers(0, arr.shape[1])
            y_start = rng.integers(0, arr.shape[0] - 20)
            arr[y_start:y_start+20, x, :] = [180, 180, 220]
        arr = arr * 0.85
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def extract_hidden(model, processor, image, prompt, layers):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return {l: fwd.hidden_states[l][0, -1, :].float().cpu().numpy() for l in layers}

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
    print("Experiment 236: Embedding Norm Analysis")
    print("=" * 60)

    from transformers import AutoModelForVision2Seq, AutoProcessor
    print("Loading OpenVLA-7B...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    layers = [1, 3, 7, 15, 31]
    prompt = "In: What action should the robot take to drive forward?\nOut:"
    base_img = make_driving_image()

    # Clean embedding norms
    print("\n--- Clean embedding norms ---")
    clean_h = extract_hidden(model, processor, base_img, prompt, layers)
    clean_norms = {l: float(np.linalg.norm(clean_h[l])) for l in layers}
    for l in layers:
        print(f"  L{l}: norm = {clean_norms[l]:.4f}")

    # Corrupted norms
    corruption_types = ['fog', 'night', 'noise', 'blur', 'snow', 'rain']
    n_samples = 5

    results = {}
    for ctype in corruption_types:
        print(f"\n--- {ctype} ---")
        layer_norms = {l: [] for l in layers}
        for i in range(n_samples):
            rng = np.random.default_rng(42 + i)
            img = apply_corruption(base_img, ctype, rng)
            h = extract_hidden(model, processor, img, prompt, layers)
            for l in layers:
                norm = float(np.linalg.norm(h[l]))
                layer_norms[l].append(norm)

        results[ctype] = {}
        for l in layers:
            norms = np.array(layer_norms[l])
            norm_diff = float(np.mean(norms) - clean_norms[l])
            norm_ratio = float(np.mean(norms) / clean_norms[l])
            # AUROC using |norm - clean_norm| as score
            id_scores = [0.0]  # Clean has norm diff = 0
            ood_scores = [abs(n - clean_norms[l]) for n in norms]
            auroc = compute_auroc(id_scores, ood_scores)
            results[ctype][f"L{l}"] = {
                "mean_norm": round(float(np.mean(norms)), 4),
                "norm_diff": round(norm_diff, 4),
                "norm_ratio": round(norm_ratio, 6),
                "norm_auroc": round(auroc, 4),
            }
            print(f"  L{l}: mean_norm={np.mean(norms):.4f} diff={norm_diff:+.4f} ratio={norm_ratio:.6f}")

    # Summary: compare norm-based vs cosine-based detection
    print("\n--- Summary: Norm-based AUROC ---")
    for ctype in corruption_types:
        for l in layers:
            auroc = results[ctype][f"L{l}"]["norm_auroc"]
            print(f"  {ctype} L{l}: AUROC={auroc}")

    output = {
        "experiment": "norm_analysis",
        "experiment_number": 236,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "clean_norms": {f"L{l}": round(clean_norms[l], 4) for l in layers},
        "corruption_types": corruption_types,
        "results": results,
    }

    out_path = f"/workspace/Vizuara-VLA-Research/experiments/norm_analysis_{output['timestamp']}.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
