"""
Experiment 215: Calibration Set Diversity Analysis
What happens when calibration images differ significantly from test images?
Tests with different scene compositions for calibration vs test.
"""
import torch, json, numpy as np, os
from datetime import datetime
from PIL import Image, ImageFilter

def make_scene(scene_type, w=256, h=256, rng=None):
    """Generate different driving scene types."""
    img = Image.new('RGB', (w, h))
    pixels = img.load()
    if rng is None:
        rng = np.random.default_rng()
    
    if scene_type == 'highway':
        for y in range(h):
            for x in range(w):
                if y < h // 2:
                    b = int(180 + 75 * (1 - y / (h / 2)))
                    pixels[x, y] = (100, 150, b)
                else:
                    g = int(80 + 40 * ((y - h/2) / (h/2)))
                    pixels[x, y] = (g, g + 10, g - 10)
    elif scene_type == 'urban':
        for y in range(h):
            for x in range(w):
                if y < h // 3:
                    pixels[x, y] = (160, 180, 210)
                elif y < 2*h // 3:
                    r = int(120 + 30 * rng.random())
                    pixels[x, y] = (r, r-10, r-20)
                else:
                    pixels[x, y] = (90, 90, 95)
    elif scene_type == 'rural':
        for y in range(h):
            for x in range(w):
                if y < h // 2:
                    pixels[x, y] = (135, 190, 240)
                else:
                    g = int(60 + 80 * rng.random())
                    pixels[x, y] = (40, g, 30)
    elif scene_type == 'parking':
        for y in range(h):
            for x in range(w):
                if y < h // 4:
                    pixels[x, y] = (200, 200, 210)
                else:
                    pixels[x, y] = (100, 100, 105)
    elif scene_type == 'tunnel':
        for y in range(h):
            for x in range(w):
                b = int(40 + 20 * (y / h))
                pixels[x, y] = (b, b, b+5)
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
    print("Experiment 215: Calibration Set Diversity")
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
    rng = np.random.default_rng(42)
    scene_types = ['highway', 'urban', 'rural', 'parking', 'tunnel']
    corruption_types = ['fog', 'night', 'blur', 'noise']
    n_per_scene = 6

    # Generate scenes
    print("\n--- Generating scenes ---")
    scenes = {}
    for st in scene_types:
        scenes[st] = [make_scene(st, rng=rng) for _ in range(n_per_scene)]
        print(f"  {st}: {n_per_scene} images")

    # Test 1: Same-scene calibration (baseline)
    print("\n--- Same-scene calibration (highway cal → highway test) ---")
    results = {}
    
    for cal_scene in scene_types:
        scene_results = {}
        # Calibrate with cal_scene
        cal_embeds = {l: [] for l in layers}
        for i in range(4):  # 4 for cal
            h = extract_hidden(model, processor, scenes[cal_scene][i], prompt, layers)
            for l in layers:
                cal_embeds[l].append(h[l])
        centroids = {l: np.mean(cal_embeds[l], axis=0) for l in layers}
        
        # Test on all scenes
        for test_scene in scene_types:
            # ID test (clean)
            id_scores = {l: [] for l in layers}
            for i in range(4, 6):
                h = extract_hidden(model, processor, scenes[test_scene][i], prompt, layers)
                for l in layers:
                    id_scores[l].append(cosine_dist(h[l], centroids[l]))
            
            # OOD test
            ood_results = {}
            for ctype in corruption_types:
                ood_scores = {l: [] for l in layers}
                for i in range(4):
                    img = apply_corruption(scenes[test_scene][i], ctype, rng)
                    h = extract_hidden(model, processor, img, prompt, layers)
                    for l in layers:
                        ood_scores[l].append(cosine_dist(h[l], centroids[l]))
                ood_results[ctype] = {f"L{l}": round(compute_auroc(id_scores[l], ood_scores[l]), 4) for l in layers}
            
            # Overall
            all_ood = {l: [] for l in layers}
            for ctype in corruption_types:
                for i in range(4):
                    img = apply_corruption(scenes[test_scene][i], ctype, rng)
                    h = extract_hidden(model, processor, img, prompt, layers)
                    for l in layers:
                        all_ood[l].append(cosine_dist(h[l], centroids[l]))
            overall = {f"L{l}": round(compute_auroc(id_scores[l], all_ood[l]), 4) for l in layers}
            
            scene_results[test_scene] = {"overall": overall, "per_corruption": ood_results}
            print(f"  Cal={cal_scene} → Test={test_scene}: {overall}")
        
        results[cal_scene] = scene_results

    # Test 2: Mixed-scene calibration
    print("\n--- Mixed-scene calibration (1 from each) ---")
    mixed_embeds = {l: [] for l in layers}
    for st in scene_types:
        h = extract_hidden(model, processor, scenes[st][0], prompt, layers)
        for l in layers:
            mixed_embeds[l].append(h[l])
    mixed_centroids = {l: np.mean(mixed_embeds[l], axis=0) for l in layers}
    
    mixed_results = {}
    for test_scene in scene_types:
        id_scores = {l: [] for l in layers}
        for i in range(4, 6):
            h = extract_hidden(model, processor, scenes[test_scene][i], prompt, layers)
            for l in layers:
                id_scores[l].append(cosine_dist(h[l], mixed_centroids[l]))
        
        all_ood = {l: [] for l in layers}
        for ctype in corruption_types:
            for i in range(4):
                img = apply_corruption(scenes[test_scene][i], ctype, rng)
                h = extract_hidden(model, processor, img, prompt, layers)
                for l in layers:
                    all_ood[l].append(cosine_dist(h[l], mixed_centroids[l]))
        
        overall = {f"L{l}": round(compute_auroc(id_scores[l], all_ood[l]), 4) for l in layers}
        mixed_results[test_scene] = overall
        print(f"  Mixed cal → Test={test_scene}: {overall}")

    output = {
        "experiment": "calibration_diversity",
        "experiment_number": 215,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "scene_types": scene_types,
        "n_per_scene": n_per_scene,
        "layers": [1, 3],
        "per_scene_calibration": results,
        "mixed_calibration": mixed_results,
    }

    out_path = f"/workspace/Vizuara-VLA-Research/experiments/cal_diversity_{output['timestamp']}.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
