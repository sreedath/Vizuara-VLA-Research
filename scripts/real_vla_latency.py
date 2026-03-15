"""
Experiment 214: End-to-End Latency Benchmark
Measure the wall-clock overhead of OOD detection on top of normal VLA inference.
"""
import torch, json, numpy as np, os, time
from datetime import datetime
from PIL import Image

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

def cosine_dist(a, b):
    return 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def main():
    print("=" * 60)
    print("Experiment 214: Latency Benchmark")
    print("=" * 60)

    from transformers import AutoModelForVision2Seq, AutoProcessor
    print("Loading OpenVLA-7B...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to drive forward?\nOut:"
    img = make_driving_image()
    layers = [1, 3]
    n_warmup = 3
    n_bench = 20

    # Warmup
    print("\n--- Warmup ---")
    for i in range(n_warmup):
        inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            _ = model(**inputs, output_hidden_states=True)
        torch.cuda.synchronize()
    print(f"  {n_warmup} warmup iterations done")

    # Benchmark 1: Normal inference (without hidden states)
    print("\n--- Benchmark: Normal inference ---")
    times_normal = []
    for i in range(n_bench):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            fwd = model(**inputs)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_normal.append(t1 - t0)
    print(f"  Mean: {np.mean(times_normal)*1000:.2f} ms, Std: {np.std(times_normal)*1000:.2f} ms")

    # Benchmark 2: Inference with hidden states
    print("\n--- Benchmark: Inference + hidden state extraction ---")
    times_hidden = []
    for i in range(n_bench):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            fwd = model(**inputs, output_hidden_states=True)
        # Extract hidden states
        hidden = {l: fwd.hidden_states[l][0, -1, :].float().cpu().numpy() for l in layers}
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_hidden.append(t1 - t0)
    print(f"  Mean: {np.mean(times_hidden)*1000:.2f} ms, Std: {np.std(times_hidden)*1000:.2f} ms")

    # Benchmark 3: Cosine distance computation only
    print("\n--- Benchmark: Cosine distance computation ---")
    centroid = np.random.randn(4096).astype(np.float32)
    embedding = np.random.randn(4096).astype(np.float32)
    times_cosine = []
    for i in range(10000):
        t0 = time.perf_counter()
        d = cosine_dist(embedding, centroid)
        t1 = time.perf_counter()
        times_cosine.append(t1 - t0)
    print(f"  Mean: {np.mean(times_cosine)*1e6:.2f} us, Std: {np.std(times_cosine)*1e6:.2f} us")

    # Benchmark 4: Full pipeline (inference + hidden state + cosine distance)
    print("\n--- Benchmark: Full OOD detection pipeline ---")
    cal_embeds = [np.random.randn(4096).astype(np.float32) for _ in range(10)]
    centroid = np.mean(cal_embeds, axis=0)
    times_full = []
    for i in range(n_bench):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            fwd = model(**inputs, output_hidden_states=True)
        hidden = fwd.hidden_states[1][0, -1, :].float().cpu().numpy()
        score = cosine_dist(hidden, centroid)
        is_ood = score > 0.001  # threshold
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_full.append(t1 - t0)
    print(f"  Mean: {np.mean(times_full)*1000:.2f} ms, Std: {np.std(times_full)*1000:.2f} ms")

    # Compute overhead
    overhead_ms = (np.mean(times_hidden) - np.mean(times_normal)) * 1000
    overhead_pct = ((np.mean(times_hidden) / np.mean(times_normal)) - 1) * 100
    cosine_us = np.mean(times_cosine) * 1e6

    print(f"\n--- Summary ---")
    print(f"  Normal inference: {np.mean(times_normal)*1000:.2f} ms")
    print(f"  With hidden states: {np.mean(times_hidden)*1000:.2f} ms")
    print(f"  Overhead: {overhead_ms:.2f} ms ({overhead_pct:.2f}%)")
    print(f"  Cosine distance: {cosine_us:.2f} us")
    print(f"  Full pipeline: {np.mean(times_full)*1000:.2f} ms")

    output = {
        "experiment": "latency_benchmark",
        "experiment_number": 214,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "n_warmup": n_warmup,
        "n_bench": n_bench,
        "results": {
            "normal_inference_ms": {
                "mean": round(np.mean(times_normal) * 1000, 2),
                "std": round(np.std(times_normal) * 1000, 2),
                "min": round(np.min(times_normal) * 1000, 2),
                "max": round(np.max(times_normal) * 1000, 2),
            },
            "with_hidden_states_ms": {
                "mean": round(np.mean(times_hidden) * 1000, 2),
                "std": round(np.std(times_hidden) * 1000, 2),
                "min": round(np.min(times_hidden) * 1000, 2),
                "max": round(np.max(times_hidden) * 1000, 2),
            },
            "cosine_distance_us": {
                "mean": round(np.mean(times_cosine) * 1e6, 2),
                "std": round(np.std(times_cosine) * 1e6, 2),
            },
            "full_pipeline_ms": {
                "mean": round(np.mean(times_full) * 1000, 2),
                "std": round(np.std(times_full) * 1000, 2),
            },
            "overhead_ms": round(overhead_ms, 2),
            "overhead_pct": round(overhead_pct, 2),
        },
    }

    out_path = f"/workspace/Vizuara-VLA-Research/experiments/latency_benchmark_{output['timestamp']}.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
