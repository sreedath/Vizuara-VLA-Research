"""
Experiment 232: Temporal Stability of Embeddings
Are embeddings stable across multiple forward passes with the same image?
Tests whether model stochasticity (if any) affects detection reliability.
Also tests with slightly different images (simulating temporal video frames).
"""
import torch, json, numpy as np, os
from datetime import datetime
from PIL import Image, ImageFilter

def make_driving_image(w=256, h=256, variation=0):
    img = Image.new('RGB', (w, h))
    pixels = img.load()
    for y in range(h):
        for x in range(w):
            if y < h // 2:
                b = int(180 + 75 * (1 - y / (h / 2)))
                pixels[x, y] = (min(255, 100 + variation), min(255, 150 + variation), b)
            else:
                g = int(80 + 40 * ((y - h/2) / (h/2)))
                pixels[x, y] = (min(255, g + variation), min(255, g + 10 + variation), max(0, g - 10 + variation))
    return img

def apply_corruption(img, name, rng):
    arr = np.array(img, dtype=np.float32)
    if name == 'fog':
        fog = np.full_like(arr, 200)
        arr = arr * 0.4 + fog * 0.6
    elif name == 'night':
        arr = arr * 0.15
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
    print("Experiment 232: Temporal Stability")
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
    base_img = make_driving_image()

    # Test 1: Same image, 20 forward passes
    print("\n--- Test 1: 20 repeated passes (same image) ---")
    n_repeats = 20
    repeated_embeds = {l: [] for l in layers}
    for i in range(n_repeats):
        h = extract_hidden(model, processor, base_img, prompt, layers)
        for l in layers:
            repeated_embeds[l].append(h[l])
        if (i+1) % 5 == 0:
            print(f"  Pass {i+1}/{n_repeats}")

    repeat_results = {}
    for l in layers:
        matrix = np.array(repeated_embeds[l])
        diffs = [cosine_dist(repeated_embeds[l][0], repeated_embeds[l][i]) for i in range(1, n_repeats)]
        repeat_results[f"L{l}"] = {
            "mean_pairwise_dist": round(float(np.mean(diffs)), 10),
            "max_pairwise_dist": round(float(np.max(diffs)), 10),
            "std_per_dim": round(float(matrix.std(axis=0).mean()), 10),
            "all_identical": bool(np.all(np.array(diffs) == 0)),
        }
        print(f"  L{l}: mean_dist={np.mean(diffs):.10f} | identical={np.all(np.array(diffs) == 0)}")

    # Test 2: Temporal video frames (1-pixel variation per frame)
    print("\n--- Test 2: Temporal frames (1-pixel variation) ---")
    n_frames = 20
    frame_embeds = {l: [] for l in layers}
    for i in range(n_frames):
        frame = make_driving_image(variation=i)
        h = extract_hidden(model, processor, frame, prompt, layers)
        for l in layers:
            frame_embeds[l].append(h[l])
        if (i+1) % 5 == 0:
            print(f"  Frame {i+1}/{n_frames}")

    centroid = {l: np.mean(frame_embeds[l], axis=0) for l in layers}
    frame_results = {}
    for l in layers:
        dists = [cosine_dist(e, centroid[l]) for e in frame_embeds[l]]
        frame_results[f"L{l}"] = {
            "mean_dist": round(float(np.mean(dists)), 8),
            "max_dist": round(float(np.max(dists)), 8),
        }
        print(f"  L{l}: mean_dist={np.mean(dists):.8f} max={np.max(dists):.8f}")

    # Test 3: Corrupted temporal frames
    print("\n--- Test 3: Corrupted frames ---")
    rng = np.random.default_rng(42)
    corr_results = {}
    for ctype in ['fog', 'night']:
        corr_dists = {l: [] for l in layers}
        for i in range(n_frames):
            frame = make_driving_image(variation=i)
            corr = apply_corruption(frame, ctype, rng)
            h = extract_hidden(model, processor, corr, prompt, layers)
            for l in layers:
                corr_dists[l].append(cosine_dist(h[l], centroid[l]))
        corr_results[ctype] = {}
        for l in layers:
            corr_results[ctype][f"L{l}"] = {
                "mean_dist": round(float(np.mean(corr_dists[l])), 6),
                "min_dist": round(float(np.min(corr_dists[l])), 6),
            }
            print(f"  {ctype} L{l}: mean={np.mean(corr_dists[l]):.6f} min={np.min(corr_dists[l]):.6f}")

    output = {
        "experiment": "temporal_stability",
        "experiment_number": 232,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "repeat_results": repeat_results,
        "frame_results": frame_results,
        "corrupted_frame_results": corr_results,
    }

    out_path = f"/workspace/Vizuara-VLA-Research/experiments/temporal_{output['timestamp']}.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
