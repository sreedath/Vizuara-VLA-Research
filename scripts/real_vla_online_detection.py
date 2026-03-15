"""
Experiment 237: Online Video Stream Detection
Simulate a video stream where corruption appears mid-stream.
Tests detection with sliding window statistics and EMA-smoothed scores.
Measures detection latency (frames to detect corruption onset).
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
    print("Experiment 237: Online Video Stream Detection")
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

    # Calibration: single clean image centroid
    base_img = make_driving_image()
    centroid = extract_hidden(model, processor, base_img, prompt, layers)[3]

    corruption_types = ['fog', 'night', 'noise']
    results = {}

    for ctype in corruption_types:
        print(f"\n=== Stream: {ctype} ===")

        # Simulate: 10 clean frames, then corruption gradually increases over 10 frames,
        # then 10 frames at full severity
        n_clean = 10
        n_transition = 10
        n_full = 10
        total = n_clean + n_transition + n_full

        distances = []
        ema_scores = []
        ema_alpha = 0.3  # EMA smoothing factor
        ema = 0.0
        threshold = None

        for frame_idx in range(total):
            variation = frame_idx  # slight temporal variation
            img = make_driving_image(variation=variation)

            if frame_idx < n_clean:
                # Clean phase
                pass
            elif frame_idx < n_clean + n_transition:
                # Gradual corruption
                sev = (frame_idx - n_clean + 1) / n_transition
                rng = np.random.default_rng(42 + frame_idx)
                img = apply_corruption(img, ctype, rng, severity=sev)
            else:
                # Full corruption
                rng = np.random.default_rng(42 + frame_idx)
                img = apply_corruption(img, ctype, rng, severity=1.0)

            h = extract_hidden(model, processor, img, prompt, layers)
            d = cosine_dist(h[3], centroid)
            distances.append(d)
            ema = ema_alpha * d + (1 - ema_alpha) * ema
            ema_scores.append(ema)

            phase = "clean" if frame_idx < n_clean else "transition" if frame_idx < n_clean + n_transition else "full"
            print(f"  Frame {frame_idx:2d} [{phase:10s}]: dist={d:.8f} ema={ema:.8f}")

        # Determine threshold from clean frames
        clean_dists = distances[:n_clean]
        max_clean = max(clean_dists)
        threshold = max_clean * 2  # 2x safety margin

        # Detection latency: first frame where EMA exceeds threshold
        detection_frame = None
        for i, score in enumerate(ema_scores):
            if score > threshold and i >= n_clean:
                detection_frame = i
                break

        # Also raw detection latency
        raw_detection_frame = None
        for i, d in enumerate(distances):
            if d > threshold and i >= n_clean:
                raw_detection_frame = i
                break

        results[ctype] = {
            "distances": [round(d, 8) for d in distances],
            "ema_scores": [round(e, 8) for e in ema_scores],
            "max_clean_dist": round(max_clean, 8),
            "threshold": round(threshold, 8),
            "raw_detection_frame": raw_detection_frame,
            "ema_detection_frame": detection_frame,
            "detection_latency_raw": raw_detection_frame - n_clean if raw_detection_frame else None,
            "detection_latency_ema": detection_frame - n_clean if detection_frame else None,
        }
        print(f"  Threshold: {threshold:.8f}")
        print(f"  Raw detection at frame {raw_detection_frame} (latency: {raw_detection_frame - n_clean if raw_detection_frame else 'N/A'} frames)")
        print(f"  EMA detection at frame {detection_frame} (latency: {detection_frame - n_clean if detection_frame else 'N/A'} frames)")

    output = {
        "experiment": "online_detection",
        "experiment_number": 237,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "layer": 3,
        "ema_alpha": 0.3,
        "stream_structure": {"clean": 10, "transition": 10, "full": 10},
        "results": results,
    }

    out_path = f"/workspace/Vizuara-VLA-Research/experiments/online_detection_{output['timestamp']}.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
