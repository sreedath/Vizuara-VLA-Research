"""
Computational Overhead Analysis.

Measures the inference latency and memory overhead of OOD detection
vs baseline forward pass. Compares: baseline (logits only),
hidden states extraction, attention extraction, full feature
extraction, and the post-processing cost of cosine distance.

Experiment 85 in the CalibDrive series.
"""
import os
import json
import datetime
import time
import numpy as np
import torch
from PIL import Image

RESULTS_DIR = "/workspace/Vizuara-VLA-Research/experiments"
os.makedirs(RESULTS_DIR, exist_ok=True)
SIZE = (256, 256)


def create_highway(idx):
    rng = np.random.default_rng(idx * 5001)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]
    img[SIZE[0]//2:] = [80, 80, 80]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [255, 255, 255]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def main():
    print("=" * 70, flush=True)
    print("COMPUTATIONAL OVERHEAD ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    from transformers import AutoModelForVision2Seq, AutoProcessor
    print("Loading OpenVLA-7B...", flush=True)
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        "openvla/openvla-7b", trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.", flush=True)

    prompt = "In: What action should the robot take to drive forward at 25 m/s safely?\nOut:"
    image = Image.fromarray(create_highway(1000))
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)

    N_WARMUP = 3
    N_MEASURE = 10

    # GPU memory baseline
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()

    results = {}

    # 1. Baseline: logits only
    print("\n1. Baseline (logits only)...", flush=True)
    for _ in range(N_WARMUP):
        with torch.no_grad():
            _ = model(**inputs)
    torch.cuda.synchronize()

    times = []
    for _ in range(N_MEASURE):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            fwd = model(**inputs)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    results['baseline'] = {
        'mean_ms': float(np.mean(times) * 1000),
        'std_ms': float(np.std(times) * 1000),
        'min_ms': float(min(times) * 1000),
        'max_ms': float(max(times) * 1000),
    }
    print(f"  {results['baseline']['mean_ms']:.1f} ± {results['baseline']['std_ms']:.1f} ms", flush=True)

    # 2. Hidden states extraction
    print("2. Hidden states extraction...", flush=True)
    for _ in range(N_WARMUP):
        with torch.no_grad():
            _ = model(**inputs, output_hidden_states=True)
    torch.cuda.synchronize()

    times = []
    for _ in range(N_MEASURE):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            fwd = model(**inputs, output_hidden_states=True)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    results['hidden_states'] = {
        'mean_ms': float(np.mean(times) * 1000),
        'std_ms': float(np.std(times) * 1000),
        'min_ms': float(min(times) * 1000),
        'max_ms': float(max(times) * 1000),
    }
    print(f"  {results['hidden_states']['mean_ms']:.1f} ± {results['hidden_states']['std_ms']:.1f} ms", flush=True)

    # 3. Attention extraction
    print("3. Attention extraction...", flush=True)
    for _ in range(N_WARMUP):
        with torch.no_grad():
            _ = model(**inputs, output_attentions=True)
    torch.cuda.synchronize()

    times = []
    for _ in range(N_MEASURE):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            fwd = model(**inputs, output_attentions=True)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    results['attention'] = {
        'mean_ms': float(np.mean(times) * 1000),
        'std_ms': float(np.std(times) * 1000),
        'min_ms': float(min(times) * 1000),
        'max_ms': float(max(times) * 1000),
    }
    print(f"  {results['attention']['mean_ms']:.1f} ± {results['attention']['std_ms']:.1f} ms", flush=True)

    # 4. Full feature extraction (hidden + attention)
    print("4. Full feature extraction...", flush=True)
    for _ in range(N_WARMUP):
        with torch.no_grad():
            _ = model(**inputs, output_hidden_states=True, output_attentions=True)
    torch.cuda.synchronize()

    times = []
    for _ in range(N_MEASURE):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            fwd = model(**inputs, output_hidden_states=True, output_attentions=True)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    results['full_features'] = {
        'mean_ms': float(np.mean(times) * 1000),
        'std_ms': float(np.std(times) * 1000),
        'min_ms': float(min(times) * 1000),
        'max_ms': float(max(times) * 1000),
    }
    print(f"  {results['full_features']['mean_ms']:.1f} ± {results['full_features']['std_ms']:.1f} ms", flush=True)

    # 5. Post-processing overhead (cosine distance computation)
    print("5. Post-processing overhead...", flush=True)
    # Create centroid
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    h = fwd.hidden_states[-1][0, -1, :].float().cpu().numpy()
    centroid = h.copy()

    times = []
    for _ in range(1000):
        t0 = time.perf_counter()
        cos = 1.0 - float(np.dot(h / (np.linalg.norm(h) + 1e-10),
                                  centroid / (np.linalg.norm(centroid) + 1e-10)))
        times.append(time.perf_counter() - t0)
    results['cosine_postprocess'] = {
        'mean_us': float(np.mean(times) * 1e6),
        'std_us': float(np.std(times) * 1e6),
    }
    print(f"  {results['cosine_postprocess']['mean_us']:.1f} ± {results['cosine_postprocess']['std_us']:.1f} μs", flush=True)

    # 6. PCA-4 post-processing
    # Simulate PCA projection
    pca_matrix = np.random.randn(4, 4096).astype(np.float32)
    centroid_4d = pca_matrix @ centroid

    times = []
    for _ in range(1000):
        t0 = time.perf_counter()
        h_4d = pca_matrix @ h
        cos = 1.0 - float(np.dot(h_4d / (np.linalg.norm(h_4d) + 1e-10),
                                  centroid_4d / (np.linalg.norm(centroid_4d) + 1e-10)))
        times.append(time.perf_counter() - t0)
    results['pca4_postprocess'] = {
        'mean_us': float(np.mean(times) * 1e6),
        'std_us': float(np.std(times) * 1e6),
    }
    print(f"  PCA-4: {results['pca4_postprocess']['mean_us']:.1f} ± {results['pca4_postprocess']['std_us']:.1f} μs", flush=True)

    # GPU memory
    torch.cuda.synchronize()
    mem_peak = torch.cuda.max_memory_allocated()
    results['memory'] = {
        'baseline_mb': mem_before / 1e6,
        'peak_mb': mem_peak / 1e6,
        'overhead_mb': (mem_peak - mem_before) / 1e6,
    }

    # Compute overheads
    baseline_ms = results['baseline']['mean_ms']
    for mode in ['hidden_states', 'attention', 'full_features']:
        overhead_ms = results[mode]['mean_ms'] - baseline_ms
        overhead_pct = (overhead_ms / baseline_ms) * 100
        results[mode]['overhead_ms'] = overhead_ms
        results[mode]['overhead_pct'] = overhead_pct

    # Summary
    print("\n" + "=" * 70, flush=True)
    print("LATENCY SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"  Baseline:          {results['baseline']['mean_ms']:.1f} ms", flush=True)
    print(f"  Hidden states:     {results['hidden_states']['mean_ms']:.1f} ms "
          f"(+{results['hidden_states']['overhead_ms']:.1f} ms, "
          f"+{results['hidden_states']['overhead_pct']:.1f}%)", flush=True)
    print(f"  Attention:         {results['attention']['mean_ms']:.1f} ms "
          f"(+{results['attention']['overhead_ms']:.1f} ms, "
          f"+{results['attention']['overhead_pct']:.1f}%)", flush=True)
    print(f"  Full features:     {results['full_features']['mean_ms']:.1f} ms "
          f"(+{results['full_features']['overhead_ms']:.1f} ms, "
          f"+{results['full_features']['overhead_pct']:.1f}%)", flush=True)
    print(f"  Cosine distance:   {results['cosine_postprocess']['mean_us']:.1f} μs", flush=True)
    print(f"  PCA-4 + cosine:    {results['pca4_postprocess']['mean_us']:.1f} μs", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'latency_analysis',
        'experiment_number': 85,
        'timestamp': timestamp,
        'n_warmup': N_WARMUP,
        'n_measure': N_MEASURE,
        'gpu': torch.cuda.get_device_name(0),
        'results': results,
    }
    output_path = os.path.join(RESULTS_DIR, f"latency_analysis_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
