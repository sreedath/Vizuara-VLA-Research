"""
Experiment 246: Action Recovery Under Corruption Removal
When corruption is removed, do the predicted action tokens return to their clean values?
Tests the full pipeline: detect corruption -> suppress action -> remove corruption -> resume.
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

def apply_corruption(img, name, rng, severity=1.0):
    arr = np.array(img, dtype=np.float32)
    if name == 'fog':
        fog = np.full_like(arr, 200)
        arr = arr * (1 - 0.6 * severity) + fog * (0.6 * severity)
    elif name == 'night':
        arr = arr * (1 - 0.85 * severity)
    elif name == 'noise':
        arr = arr + rng.normal(0, 30 * severity, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def extract_hidden_and_action(model, processor, image, prompt, layers):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    emb = {l: outputs.hidden_states[l][0, -1, :].float().cpu().numpy() for l in layers}
    action_token = int(outputs.logits[0, -1, :].argmax().cpu())
    return emb, action_token

def cosine_dist(a, b):
    return 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def main():
    print("=" * 60)
    print("Experiment 246: Action Recovery Under Corruption Removal")
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
    base_img = make_driving_image()
    centroid_data = extract_hidden_and_action(model, processor, base_img, prompt, layers)
    centroid = centroid_data[0][3]
    clean_token = centroid_data[1]
    print(f"Clean action token: {clean_token}")

    corruption_types = ['fog', 'night', 'noise']
    results = {}

    for ctype in corruption_types:
        print(f"\n=== {ctype} ===")

        # Phase: clean -> corrupt -> clean
        frames_data = []
        for frame_idx in range(15):
            img = make_driving_image(variation=frame_idx)
            if 5 <= frame_idx < 10:
                rng = np.random.default_rng(42 + frame_idx)
                img = apply_corruption(img, ctype, rng)

            emb, action = extract_hidden_and_action(model, processor, img, prompt, layers)
            d = cosine_dist(emb[3], centroid)
            phase = "clean" if frame_idx < 5 else "corrupt" if frame_idx < 10 else "recovered"
            frames_data.append({
                "frame": frame_idx,
                "phase": phase,
                "distance": round(d, 8),
                "action_token": action,
                "action_matches_clean": action == clean_token,
            })
            print(f"  Frame {frame_idx:2d} [{phase:9s}]: dist={d:.8f} action={action} {'==' if action == clean_token else '!='} clean({clean_token})")

        # Count action recovery
        clean_actions = sum(1 for f in frames_data if f['phase'] == 'clean' and f['action_matches_clean'])
        corrupt_actions = sum(1 for f in frames_data if f['phase'] == 'corrupt' and f['action_matches_clean'])
        recovered_actions = sum(1 for f in frames_data if f['phase'] == 'recovered' and f['action_matches_clean'])

        results[ctype] = {
            "frames": frames_data,
            "clean_action_match": f"{clean_actions}/5",
            "corrupt_action_match": f"{corrupt_actions}/5",
            "recovered_action_match": f"{recovered_actions}/5",
        }
        print(f"  Clean match: {clean_actions}/5, Corrupt match: {corrupt_actions}/5, Recovered: {recovered_actions}/5")

    output = {
        "experiment": "action_recovery",
        "experiment_number": 246,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "layer": 3,
        "clean_action_token": clean_token,
        "results": results,
    }

    out_path = f"/workspace/Vizuara-VLA-Research/experiments/action_recovery_{output['timestamp']}.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
