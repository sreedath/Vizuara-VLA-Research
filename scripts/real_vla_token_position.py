"""
Experiment 212: Token Position Analysis for OOD Detection
Where in the token sequence does the OOD signal appear?
Tests hidden state at different token positions (first, middle, last, mean).
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
    print("Experiment 212: Token Position Analysis")
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

    # First, understand the token structure
    print("\n--- Analyzing token structure ---")
    inputs = processor(prompt, base_imgs[0]).to(model.device, dtype=torch.bfloat16)
    seq_len = inputs['input_ids'].shape[1]
    print(f"  Sequence length: {seq_len}")

    positions = {
        'first': 0,
        'quarter': seq_len // 4,
        'middle': seq_len // 2,
        'three_quarter': 3 * seq_len // 4,
        'last': -1,
    }
    print(f"  Positions: {positions}")

    def extract_multi_pos(model, processor, image, prompt, layers, positions):
        inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            fwd = model(**inputs, output_hidden_states=True)
        result = {}
        for l in layers:
            hs = fwd.hidden_states[l][0]
            result[l] = {}
            for name, idx in positions.items():
                result[l][name] = hs[idx].float().cpu().numpy()
            result[l]['mean_all'] = hs.float().mean(dim=0).cpu().numpy()
        return result

    print("\n--- Calibrating per-position centroids ---")
    centroids = {}
    for l in layers:
        centroids[l] = {}
        embeds = {pos: [] for pos in list(positions.keys()) + ['mean_all']}
        for i in range(n_cal):
            h = extract_multi_pos(model, processor, base_imgs[i], prompt, [l], positions)
            for pos in embeds:
                embeds[pos].append(h[l][pos])
            if (i+1) % 5 == 0:
                print(f"  L{l} cal {i+1}/{n_cal}")
        for pos in embeds:
            centroids[l][pos] = np.mean(embeds[pos], axis=0)

    print("\n--- Testing detection by token position ---")
    results = {}
    for l in layers:
        pos_results = {}
        for pos_name in list(positions.keys()) + ['mean_all']:
            id_scores = []
            for i in range(n_cal, n_cal + n_test):
                h = extract_multi_pos(model, processor, base_imgs[i], prompt, [l], positions)
                id_scores.append(cosine_dist(h[l][pos_name], centroids[l][pos_name]))

            per_corr = {}
            ood_all = []
            for ctype in corruption_types:
                ood_scores = []
                for i in range(n_test):
                    img = apply_corruption(base_imgs[i], ctype, rng)
                    h = extract_multi_pos(model, processor, img, prompt, [l], positions)
                    d = cosine_dist(h[l][pos_name], centroids[l][pos_name])
                    ood_scores.append(d)
                    ood_all.append(d)
                per_corr[ctype] = round(compute_auroc(id_scores, ood_scores), 4)

            overall = round(compute_auroc(id_scores, ood_all), 4)
            id_mean = float(np.mean(id_scores))
            ood_mean = float(np.mean(ood_all))
            separation = round(ood_mean / (id_mean + 1e-10), 4) if id_mean > 1e-8 else 0
            pos_results[pos_name] = {
                "auroc": overall,
                "per_corruption": per_corr,
                "id_mean": round(id_mean, 8),
                "ood_mean": round(ood_mean, 6),
                "separation": separation,
            }
            print(f"  L{l} {pos_name:16s}: AUROC={overall:.4f}, sep={separation:.2f}")

        results[f"L{l}"] = pos_results

    output = {
        "experiment": "token_position_analysis",
        "experiment_number": 212,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "seq_length": seq_len,
        "positions": {k: int(v) if v != -1 else seq_len - 1 for k, v in positions.items()},
        "n_cal": n_cal,
        "n_test": n_test,
        "layers": [1, 3],
        "results": results,
    }

    out_path = f"/workspace/Vizuara-VLA-Research/experiments/token_position_{output['timestamp']}.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
