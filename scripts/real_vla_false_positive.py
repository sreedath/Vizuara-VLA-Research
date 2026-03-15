"""
Experiment 225: False Positive Analysis with Benign Augmentations
Do benign transformations (small brightness change, slight resize, JPEG compression)
trigger false positives? Critical for understanding operational false alarm rate.
"""
import torch, json, numpy as np, os, io
from datetime import datetime
from PIL import Image, ImageFilter, ImageEnhance

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

def apply_benign(img, name, rng):
    """Apply benign augmentations that should NOT trigger OOD detection."""
    if name == 'brightness_up':
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(1.15)  # 15% brighter
    elif name == 'brightness_down':
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(0.85)  # 15% dimmer
    elif name == 'contrast':
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(1.2)  # 20% more contrast
    elif name == 'jpeg_q50':
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=50)
        buf.seek(0)
        return Image.open(buf).convert('RGB')
    elif name == 'jpeg_q10':
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=10)
        buf.seek(0)
        return Image.open(buf).convert('RGB')
    elif name == 'slight_blur':
        return img.filter(ImageFilter.GaussianBlur(radius=1))
    elif name == 'sharpen':
        return img.filter(ImageFilter.SHARPEN)
    elif name == 'flip_h':
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

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
    print("Experiment 225: False Positive Analysis")
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

    benign_types = ['brightness_up', 'brightness_down', 'contrast', 'jpeg_q50', 'jpeg_q10', 'slight_blur', 'sharpen', 'flip_h']
    corruption_types = ['fog', 'night', 'blur', 'noise']

    # Calibrate
    print("\n--- Calibration ---")
    cal_embeds = {l: [] for l in layers}
    for i in range(n_cal):
        h = extract_hidden(model, processor, base_imgs[i], prompt, layers)
        for l in layers:
            cal_embeds[l].append(h[l])
    centroids = {l: np.mean(cal_embeds[l], axis=0) for l in layers}

    # Test ID (clean)
    print("\n--- Clean test ---")
    id_scores = {l: [] for l in layers}
    for i in range(n_cal, n_cal + n_test):
        h = extract_hidden(model, processor, base_imgs[i], prompt, layers)
        for l in layers:
            id_scores[l].append(cosine_dist(h[l], centroids[l]))

    # Test benign augmentations
    print("\n--- Benign augmentations ---")
    benign_results = {}
    for btype in benign_types:
        scores = {l: [] for l in layers}
        for i in range(n_test):
            img = apply_benign(base_imgs[i], btype, rng)
            h = extract_hidden(model, processor, img, prompt, layers)
            for l in layers:
                scores[l].append(cosine_dist(h[l], centroids[l]))

        benign_results[btype] = {}
        for l in layers:
            # Would a threshold derived from ID scores flag these as OOD?
            id_max = max(id_scores[l]) if id_scores[l] else 0
            benign_max = max(scores[l])
            benign_mean = float(np.mean(scores[l]))

            # AUROC: are benign distinguishable from clean?
            auroc_vs_clean = round(compute_auroc(id_scores[l], scores[l]), 4)

            benign_results[btype][f"L{l}"] = {
                "auroc_vs_clean": auroc_vs_clean,
                "mean_dist": round(benign_mean, 6),
                "max_dist": round(float(benign_max), 6),
                "id_max": round(float(id_max), 6),
                "would_false_alarm": benign_max > id_max * 2 if id_max > 0 else benign_max > 0.0001,
            }
            print(f"  {btype} L{l}: AUROC_vs_clean={auroc_vs_clean} mean_dist={benign_mean:.6f}")

    # Test true corruptions for comparison
    print("\n--- True corruptions (for comparison) ---")
    corruption_results = {}
    for ctype in corruption_types:
        scores = {l: [] for l in layers}
        for i in range(n_test):
            img = apply_corruption(base_imgs[i], ctype, rng)
            h = extract_hidden(model, processor, img, prompt, layers)
            for l in layers:
                scores[l].append(cosine_dist(h[l], centroids[l]))

        corruption_results[ctype] = {}
        for l in layers:
            auroc = round(compute_auroc(id_scores[l], scores[l]), 4)
            corruption_results[ctype][f"L{l}"] = {
                "auroc": auroc,
                "mean_dist": round(float(np.mean(scores[l])), 6),
            }
            print(f"  {ctype} L{l}: AUROC={auroc} mean_dist={np.mean(scores[l]):.6f}")

    output = {
        "experiment": "false_positive_analysis",
        "experiment_number": 225,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "n_cal": n_cal,
        "n_test": n_test,
        "benign_types": benign_types,
        "corruption_types": corruption_types,
        "benign_results": benign_results,
        "corruption_results": corruption_results,
    }

    out_path = f"/workspace/Vizuara-VLA-Research/experiments/false_positive_{output['timestamp']}.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
