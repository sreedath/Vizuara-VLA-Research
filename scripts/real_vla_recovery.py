"""
Experiment 239: Corruption Recovery Detection
When corruption disappears mid-stream, how quickly does the detector recover?
Tests whether the detector correctly identifies return to normal after corruption.
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

def extract_hidden(model, processor, image, prompt, layers):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return {l: fwd.hidden_states[l][0, -1, :].float().cpu().numpy() for l in layers}

def cosine_dist(a, b):
    return 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def main():
    print("=" * 60)
    print("Experiment 239: Corruption Recovery Detection")
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
    centroid = extract_hidden(model, processor, base_img, prompt, layers)[3]

    corruption_types = ['fog', 'night', 'noise']
    results = {}

    for ctype in corruption_types:
        print(f"\n=== {ctype}: clean → corrupt → clean ===")

        # 5 clean, 10 corrupt (full), 5 recovery, 10 clean
        phases = []
        for i in range(5):
            phases.append(('clean', i, 0))
        for i in range(10):
            phases.append(('corrupt', 5+i, 1.0))
        for i in range(5):
            # Gradual recovery: severity goes from 1.0 to 0.0
            sev = 1.0 - (i+1) / 5
            phases.append(('recovery', 15+i, sev))
        for i in range(10):
            phases.append(('recovered', 20+i, 0))

        distances = []
        for phase_name, frame_idx, severity in phases:
            img = make_driving_image(variation=frame_idx)
            if severity > 0:
                rng = np.random.default_rng(42 + frame_idx)
                img = apply_corruption(img, ctype, rng, severity=severity)
            h = extract_hidden(model, processor, img, prompt, layers)
            d = cosine_dist(h[3], centroid)
            distances.append(d)
            print(f"  Frame {frame_idx:2d} [{phase_name:10s} sev={severity:.1f}]: dist={d:.8f}")

        # Check if recovered frames return to clean-level distances
        clean_max = max(distances[:5])
        recovered_max = max(distances[20:])
        recovered_mean = np.mean(distances[20:])

        results[ctype] = {
            "distances": [round(d, 8) for d in distances],
            "clean_max": round(clean_max, 8),
            "corrupt_mean": round(float(np.mean(distances[5:15])), 8),
            "recovered_max": round(recovered_max, 8),
            "recovered_mean": round(recovered_mean, 8),
            "recovery_complete": bool(recovered_max < clean_max * 3),
        }
        print(f"  Clean max: {clean_max:.8f}")
        print(f"  Recovered max: {recovered_max:.8f}")
        print(f"  Recovery complete: {recovered_max < clean_max * 3}")

    output = {
        "experiment": "recovery",
        "experiment_number": 239,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "layer": 3,
        "stream_structure": "5 clean + 10 corrupt + 5 recovery + 10 recovered",
        "results": results,
    }

    out_path = f"/workspace/Vizuara-VLA-Research/experiments/recovery_{output['timestamp']}.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
