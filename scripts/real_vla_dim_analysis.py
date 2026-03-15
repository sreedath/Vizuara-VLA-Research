"""
Experiment 227: Per-Dimension Embedding Analysis
Which embedding dimensions carry the most OOD signal?
Compute per-dimension deviation between clean and corrupted embeddings.
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

def main():
    print("=" * 60)
    print("Experiment 227: Per-Dimension Embedding Analysis")
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
    n_imgs = 10
    rng = np.random.default_rng(42)
    base_imgs = [make_driving_image() for _ in range(n_imgs)]
    corruption_types = ['fog', 'night', 'blur', 'noise']

    # Extract clean embeddings
    print("\n--- Clean embeddings ---")
    clean_embeds = {l: [] for l in layers}
    for i in range(n_imgs):
        h = extract_hidden(model, processor, base_imgs[i], prompt, layers)
        for l in layers:
            clean_embeds[l].append(h[l])

    # Extract corrupted embeddings
    print("\n--- Corrupted embeddings ---")
    corr_embeds = {ctype: {l: [] for l in layers} for ctype in corruption_types}
    for ctype in corruption_types:
        for i in range(n_imgs):
            img = apply_corruption(base_imgs[i], ctype, rng)
            h = extract_hidden(model, processor, img, prompt, layers)
            for l in layers:
                corr_embeds[ctype][l].append(h[l])
        print(f"  {ctype}: done")

    results = {}
    for l in layers:
        print(f"\n--- L{l} ---")
        clean_mean = np.mean(clean_embeds[l], axis=0)
        clean_std = np.std(clean_embeds[l], axis=0)
        embed_dim = len(clean_mean)
        print(f"  Embed dim: {embed_dim}")
        print(f"  Clean mean range: [{clean_mean.min():.4f}, {clean_mean.max():.4f}]")
        print(f"  Clean std range: [{clean_std.min():.6f}, {clean_std.max():.6f}]")
        print(f"  Clean std mean: {clean_std.mean():.6f}")

        # Per-dimension deviation for each corruption
        layer_results = {
            "embed_dim": embed_dim,
            "clean_std_mean": round(float(clean_std.mean()), 8),
            "clean_std_max": round(float(clean_std.max()), 8),
            "n_zero_std_dims": int(np.sum(clean_std < 1e-8)),
        }

        for ctype in corruption_types:
            corr_mean = np.mean(corr_embeds[ctype][l], axis=0)
            # Per-dimension absolute deviation
            deviation = np.abs(corr_mean - clean_mean)
            # Relative deviation (where clean_std > 0)
            nonzero_mask = clean_std > 1e-8

            # Top-k most deviating dimensions
            top_k = 20
            top_indices = np.argsort(deviation)[-top_k:][::-1]

            # What fraction of total deviation comes from top-k dims?
            total_dev = deviation.sum()
            top_dev = deviation[top_indices].sum()

            # Statistics on dimension-wise deviation
            n_active = int(np.sum(deviation > 1e-6))
            concentration = float(top_dev / (total_dev + 1e-10))

            layer_results[ctype] = {
                "mean_deviation": round(float(deviation.mean()), 8),
                "max_deviation": round(float(deviation.max()), 6),
                "n_active_dims": n_active,
                "active_fraction": round(n_active / embed_dim, 4),
                "top20_concentration": round(concentration, 4),
                "top5_dims": top_indices[:5].tolist(),
                "top5_deviations": [round(float(deviation[i]), 6) for i in top_indices[:5]],
            }
            print(f"  {ctype}: active_dims={n_active}/{embed_dim} ({n_active/embed_dim:.1%}) | top20_conc={concentration:.3f}")

        # Cross-corruption dimension overlap: how many of the top-20 dims are shared?
        top20_sets = {}
        for ctype in corruption_types:
            corr_mean = np.mean(corr_embeds[ctype][l], axis=0)
            deviation = np.abs(corr_mean - clean_mean)
            top20_sets[ctype] = set(np.argsort(deviation)[-20:][::-1].tolist())

        overlap_results = {}
        for c1 in corruption_types:
            for c2 in corruption_types:
                if c1 < c2:
                    overlap = len(top20_sets[c1] & top20_sets[c2])
                    overlap_results[f"{c1}_vs_{c2}"] = overlap
        layer_results["top20_overlap"] = overlap_results
        print(f"  Cross-corruption top-20 overlap: {overlap_results}")

        results[f"L{l}"] = layer_results

    output = {
        "experiment": "per_dimension_analysis",
        "experiment_number": 227,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "n_images": n_imgs,
        "layers": layers,
        "results": results,
    }

    out_path = f"/workspace/Vizuara-VLA-Research/experiments/dim_analysis_{output['timestamp']}.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
