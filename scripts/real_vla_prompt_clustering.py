"""
Experiment 235: Cross-Prompt Corruption Identification
Is corruption type identification robust to prompt changes?
Tests whether corruption clusters maintain separation across 5 different prompts.
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
    print("Experiment 235: Cross-Prompt Corruption Identification")
    print("=" * 60)

    from transformers import AutoModelForVision2Seq, AutoProcessor
    print("Loading OpenVLA-7B...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    layers = [3]
    prompts = [
        "In: What action should the robot take to drive forward?\nOut:",
        "In: What action should the robot take to turn left?\nOut:",
        "In: What action should the robot take to stop?\nOut:",
        "In: What action should the robot take to pick up the object?\nOut:",
        "In: What action should the robot take to navigate to the goal?\nOut:",
    ]
    prompt_names = ['forward', 'left', 'stop', 'pickup', 'navigate']

    base_img = make_driving_image()
    corruption_types = ['fog', 'night', 'noise', 'blur']

    results = {}
    for pi, (prompt, pname) in enumerate(zip(prompts, prompt_names)):
        print(f"\n=== Prompt: {pname} ===")

        # Clean centroid for this prompt
        centroid = extract_hidden(model, processor, base_img, prompt, layers)[3]

        # Get corruption embeddings
        type_centroids = {}
        type_dists = {}
        for ctype in corruption_types:
            rng = np.random.default_rng(42)
            img = apply_corruption(base_img, ctype, rng)
            h = extract_hidden(model, processor, img, prompt, layers)
            type_centroids[ctype] = h[3]
            type_dists[ctype] = cosine_dist(h[3], centroid)
            print(f"  {ctype}: dist={type_dists[ctype]:.6f}")

        # Cross-identify: for each corruption, which type centroid is closest?
        correct = 0
        for ctype in corruption_types:
            dists_to_types = {c: cosine_dist(type_centroids[ctype], type_centroids[c])
                             for c in corruption_types if c != ctype}
            # The query is: does this corruption's embedding sit closer to its own type?
            # Actually test: given corruption centroids from THIS prompt, classify
            # new samples generated with different seeds
            rng2 = np.random.default_rng(99)
            img2 = apply_corruption(base_img, ctype, rng2)
            h2 = extract_hidden(model, processor, img2, prompt, layers)
            dists_all = {c: cosine_dist(h2[3], type_centroids[c]) for c in corruption_types}
            predicted = min(dists_all, key=dists_all.get)
            if predicted == ctype:
                correct += 1
            print(f"  {ctype} -> predicted: {predicted} (correct: {predicted == ctype})")

        nc_acc = correct / len(corruption_types)

        results[pname] = {
            "distances": {c: round(type_dists[c], 6) for c in corruption_types},
            "nc_accuracy": round(nc_acc, 4),
        }

    # Cross-prompt transfer: train centroids on prompt 0, test on prompt 1-4
    print("\n=== Cross-prompt transfer ===")
    base_prompt = prompts[0]
    centroid_p0 = extract_hidden(model, processor, base_img, base_prompt, layers)[3]
    type_centroids_p0 = {}
    for ctype in corruption_types:
        rng = np.random.default_rng(42)
        img = apply_corruption(base_img, ctype, rng)
        type_centroids_p0[ctype] = extract_hidden(model, processor, img, base_prompt, layers)[3]

    transfer_results = {}
    for pi, (prompt, pname) in enumerate(zip(prompts[1:], prompt_names[1:])):
        correct = 0
        for ctype in corruption_types:
            rng = np.random.default_rng(42)
            img = apply_corruption(base_img, ctype, rng)
            h = extract_hidden(model, processor, img, prompt, layers)
            # Classify using prompt-0 centroids (relative to prompt-0 clean)
            # Need to use relative distances
            centroid_pi = extract_hidden(model, processor, base_img, prompt, layers)[3]
            rel_emb = h[3]  # The corrupted embedding under this prompt
            # Classify by finding which corruption type centroid (under p0) has most similar
            # direction-of-shift from clean
            shift_pi = rel_emb - centroid_pi
            shifts_p0 = {c: type_centroids_p0[c] - centroid_p0 for c in corruption_types}
            # Cosine similarity of shift vectors
            dists_shift = {}
            for c in corruption_types:
                s0 = shifts_p0[c]
                si = shift_pi
                cos_sim = np.dot(s0, si) / (np.linalg.norm(s0) * np.linalg.norm(si) + 1e-10)
                dists_shift[c] = -cos_sim  # negative so min = most similar
            predicted = min(dists_shift, key=dists_shift.get)
            if predicted == ctype:
                correct += 1
            print(f"  {pname}/{ctype} -> predicted: {predicted} (correct: {predicted == ctype})")

        transfer_results[pname] = {
            "accuracy": round(correct / len(corruption_types), 4),
        }

    output = {
        "experiment": "prompt_clustering",
        "experiment_number": 235,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "layer": 3,
        "prompts": prompt_names,
        "corruption_types": corruption_types,
        "per_prompt_results": results,
        "cross_prompt_transfer": transfer_results,
    }

    out_path = f"/workspace/Vizuara-VLA-Research/experiments/prompt_clustering_{output['timestamp']}.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
