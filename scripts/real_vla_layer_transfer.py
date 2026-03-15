"""
Experiment 220: Cross-Layer Transfer Analysis
Does a centroid calibrated at one layer transfer to other layers?
Tests whether L1 centroid can be used for detection at L3, and vice versa.
Also tests PCA-reduced detection (50, 100, 500 dims).
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
    print("Experiment 220: Cross-Layer and PCA-Reduced Detection")
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
    n_cal, n_test = 10, 8
    rng = np.random.default_rng(42)
    base_imgs = [make_driving_image() for _ in range(20)]
    corruption_types = ['fog', 'night', 'blur', 'noise']

    # Extract all embeddings
    print("\n--- Extracting all embeddings ---")
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

    # Standard detection at each layer
    print("\n--- Standard per-layer detection ---")
    centroids = {l: np.mean(cal_embeds[l], axis=0) for l in layers}
    standard_results = {}
    for l in layers:
        id_scores = [cosine_dist(e, centroids[l]) for e in test_embeds[l]]
        ood_all = []
        for ctype in corruption_types:
            for e in ood_embeds[ctype][l]:
                ood_all.append(cosine_dist(e, centroids[l]))
        auroc = round(compute_auroc(id_scores, ood_all), 4)
        standard_results[f"L{l}"] = auroc
        print(f"  L{l}: AUROC={auroc}")

    # PCA-reduced detection
    print("\n--- PCA-reduced detection ---")
    pca_dims = [10, 50, 100, 500, 1000, 2048]
    pca_results = {}
    for l in [1, 3]:
        # Fit PCA on calibration data
        cal_matrix = np.array(cal_embeds[l])
        mean = cal_matrix.mean(axis=0)
        centered = cal_matrix - mean
        # SVD for PCA
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        
        pca_results[f"L{l}"] = {}
        for k in pca_dims:
            if k > Vt.shape[0]:
                continue
            proj = Vt[:k]  # top k components
            
            # Project all data
            cal_proj = (centered @ proj.T)
            centroid_proj = cal_proj.mean(axis=0)
            
            test_centered = np.array(test_embeds[l]) - mean
            test_proj = test_centered @ proj.T
            
            id_scores = [cosine_dist(e, centroid_proj) for e in test_proj]
            ood_all = []
            for ctype in corruption_types:
                ood_centered = np.array(ood_embeds[ctype][l]) - mean
                ood_proj = ood_centered @ proj.T
                for e in ood_proj:
                    ood_all.append(cosine_dist(e, centroid_proj))
            
            auroc = round(compute_auroc(id_scores, ood_all), 4)
            pca_results[f"L{l}"][f"k{k}"] = auroc
            print(f"  L{l} k={k}: AUROC={auroc}")

    # Concatenated layers
    print("\n--- Concatenated layer detection ---")
    concat_results = {}
    layer_combos = [[1,3], [1,31], [3,31], [1,3,7]]
    for layer_set in layer_combos:
        
        cal_concat = [np.concatenate([cal_embeds[l][i] for l in layer_set]) for i in range(n_cal)]
        centroid_concat = np.mean(cal_concat, axis=0)
        
        test_concat = [np.concatenate([test_embeds[l][i] for l in layer_set]) for i in range(n_test)]
        id_scores = [cosine_dist(e, centroid_concat) for e in test_concat]
        
        ood_all = []
        for ctype in corruption_types:
            for i in range(n_test):
                ood_concat = np.concatenate([ood_embeds[ctype][l][i] for l in layer_set])
                ood_all.append(cosine_dist(ood_concat, centroid_concat))
        
        key = "+".join([f"L{l}" for l in layer_set])
        auroc = round(compute_auroc(id_scores, ood_all), 4)
        concat_results[key] = auroc
        print(f"  {key}: AUROC={auroc}")

    output = {
        "experiment": "cross_layer_pca",
        "experiment_number": 220,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "n_cal": n_cal,
        "n_test": n_test,
        "layers": [1, 3, 7, 15, 31],
        "standard_detection": standard_results,
        "pca_reduced": pca_results,
        "concatenated_layers": concat_results,
    }

    out_path = f"/workspace/Vizuara-VLA-Research/experiments/cross_layer_pca_{output['timestamp']}.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
