"""
Computational Overhead Profiling.

Measures the wall-clock time for:
1. Standard VLA inference (generate actions)
2. Hidden state extraction (our OOD detection method)
3. Centroid computation
4. Cosine distance scoring
5. End-to-end detection pipeline

Experiment 131 in the CalibDrive series.
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
    rng = np.random.default_rng(idx * 24001)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]
    img[SIZE[0]//2:] = [80, 80, 80]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [255, 255, 255]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def main():
    print("=" * 70, flush=True)
    print("COMPUTATIONAL OVERHEAD PROFILING", flush=True)
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
    N_WARMUP = 3
    N_TRIALS = 20

    # Prepare test image
    img = Image.fromarray(create_highway(9999))

    # Warmup
    print("\nWarming up...", flush=True)
    for _ in range(N_WARMUP):
        inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            model(**inputs, output_hidden_states=True)
    torch.cuda.synchronize()

    # Test 1: Standard forward pass (no hidden states)
    print(f"\n--- Standard Forward ({N_TRIALS} trials) ---", flush=True)
    times_forward = []
    for i in range(N_TRIALS):
        inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(**inputs)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_forward.append(t1 - t0)
    print(f"  Forward: {np.mean(times_forward)*1000:.1f} +/- {np.std(times_forward)*1000:.1f} ms", flush=True)

    # Test 2: Forward pass WITH hidden states
    print(f"\n--- Forward + Hidden States ({N_TRIALS} trials) ---", flush=True)
    times_hidden = []
    for i in range(N_TRIALS):
        inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            fwd = model(**inputs, output_hidden_states=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_hidden.append(t1 - t0)
    print(f"  Forward+Hidden: {np.mean(times_hidden)*1000:.1f} +/- {np.std(times_hidden)*1000:.1f} ms", flush=True)

    # Test 3: Hidden state extraction (last layer, last token)
    print(f"\n--- Hidden State Extraction ({N_TRIALS} trials) ---", flush=True)
    times_extract = []
    for i in range(N_TRIALS):
        inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            fwd = model(**inputs, output_hidden_states=True)
        h = fwd.hidden_states[-1][0, -1, :].float().cpu().numpy()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_extract.append(t1 - t0)
    print(f"  Extract: {np.mean(times_extract)*1000:.1f} +/- {np.std(times_extract)*1000:.1f} ms", flush=True)

    # Test 4: Autoregressive generation (action tokens)
    print(f"\n--- Action Generation ({N_TRIALS} trials) ---", flush=True)
    times_generate = []
    for i in range(N_TRIALS):
        inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_generate.append(t1 - t0)
    print(f"  Generate: {np.mean(times_generate)*1000:.1f} +/- {np.std(times_generate)*1000:.1f} ms", flush=True)

    # Test 5: Preprocessing only
    print(f"\n--- Preprocessing ({N_TRIALS} trials) ---", flush=True)
    times_preprocess = []
    for i in range(N_TRIALS):
        t0 = time.perf_counter()
        inputs = processor(prompt, img)
        t1 = time.perf_counter()
        times_preprocess.append(t1 - t0)
    print(f"  Preprocess: {np.mean(times_preprocess)*1000:.1f} +/- {np.std(times_preprocess)*1000:.1f} ms", flush=True)

    # Test 6: Cosine distance computation (pure numpy)
    print(f"\n--- Cosine Distance ({N_TRIALS*100} trials) ---", flush=True)
    centroid = np.random.randn(4096).astype(np.float32)
    embedding = np.random.randn(4096).astype(np.float32)
    times_cosine = []
    for i in range(N_TRIALS * 100):
        t0 = time.perf_counter()
        cos = float(1 - np.dot(embedding, centroid) / (np.linalg.norm(embedding) * np.linalg.norm(centroid) + 1e-10))
        t1 = time.perf_counter()
        times_cosine.append(t1 - t0)
    print(f"  Cosine: {np.mean(times_cosine)*1e6:.1f} +/- {np.std(times_cosine)*1e6:.1f} us", flush=True)

    # Test 7: Full OOD detection pipeline
    print(f"\n--- Full OOD Pipeline ({N_TRIALS} trials) ---", flush=True)
    cal_centroid = np.random.randn(4096).astype(np.float32)
    times_pipeline = []
    for i in range(N_TRIALS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        # 1. Preprocess
        inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)
        # 2. Forward + hidden states
        with torch.no_grad():
            fwd = model(**inputs, output_hidden_states=True)
        # 3. Extract
        h = fwd.hidden_states[-1][0, -1, :].float().cpu().numpy()
        # 4. Score
        score = float(1 - np.dot(h, cal_centroid) / (np.linalg.norm(h) * np.linalg.norm(cal_centroid) + 1e-10))
        # 5. Threshold
        is_ood = score > 0.105
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_pipeline.append(t1 - t0)
    print(f"  Pipeline: {np.mean(times_pipeline)*1000:.1f} +/- {np.std(times_pipeline)*1000:.1f} ms", flush=True)

    # Summary
    print("\n--- Summary ---", flush=True)
    overhead_hidden = (np.mean(times_hidden) - np.mean(times_forward)) * 1000
    overhead_pct = overhead_hidden / (np.mean(times_forward) * 1000) * 100
    detection_overhead = np.mean(times_cosine) * 1000  # cosine distance in ms

    print(f"  Standard forward: {np.mean(times_forward)*1000:.1f} ms", flush=True)
    print(f"  Forward + hidden: {np.mean(times_hidden)*1000:.1f} ms", flush=True)
    print(f"  Hidden state overhead: {overhead_hidden:.1f} ms ({overhead_pct:.1f}%)", flush=True)
    print(f"  Action generation: {np.mean(times_generate)*1000:.1f} ms", flush=True)
    print(f"  Cosine distance: {np.mean(times_cosine)*1e6:.1f} us", flush=True)
    print(f"  Full pipeline: {np.mean(times_pipeline)*1000:.1f} ms", flush=True)

    # GPU info
    if torch.cuda.is_available():
        print(f"\n  GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"  GPU Memory: {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'compute_overhead',
        'experiment_number': 131,
        'timestamp': timestamp,
        'n_trials': N_TRIALS,
        'timings_ms': {
            'forward': {'mean': float(np.mean(times_forward)*1000), 'std': float(np.std(times_forward)*1000)},
            'forward_hidden': {'mean': float(np.mean(times_hidden)*1000), 'std': float(np.std(times_hidden)*1000)},
            'extract': {'mean': float(np.mean(times_extract)*1000), 'std': float(np.std(times_extract)*1000)},
            'generate': {'mean': float(np.mean(times_generate)*1000), 'std': float(np.std(times_generate)*1000)},
            'preprocess': {'mean': float(np.mean(times_preprocess)*1000), 'std': float(np.std(times_preprocess)*1000)},
            'cosine_us': {'mean': float(np.mean(times_cosine)*1e6), 'std': float(np.std(times_cosine)*1e6)},
            'pipeline': {'mean': float(np.mean(times_pipeline)*1000), 'std': float(np.std(times_pipeline)*1000)},
        },
        'overhead': {
            'hidden_state_ms': float(overhead_hidden),
            'hidden_state_pct': float(overhead_pct),
        },
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A',
        'gpu_memory_gb': float(torch.cuda.max_memory_allocated()/1e9) if torch.cuda.is_available() else 0.0,
    }
    output_path = os.path.join(RESULTS_DIR, f"compute_overhead_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
