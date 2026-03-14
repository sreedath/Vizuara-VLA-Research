"""
Latency Profiling: Zero-Overhead OOD Detection.

Measures exact inference time with and without:
1. output_hidden_states=True
2. output_scores=True
3. Both enabled

Demonstrates that CalibDrive's OOD detection adds zero computational
overhead because hidden states and scores are already computed during
generation — we just choose to return them.

Experiment 60 in the CalibDrive series.
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
    print("LATENCY PROFILING", flush=True)
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
    n_warmup = 3
    n_measure = 15

    configs = {
        'baseline': {'output_hidden_states': False, 'output_scores': False},
        'scores_only': {'output_hidden_states': False, 'output_scores': True},
        'hidden_only': {'output_hidden_states': True, 'output_scores': False},
        'both': {'output_hidden_states': True, 'output_scores': True},
    }

    results = {}
    for config_name, kwargs in configs.items():
        print(f"\n  Config: {config_name}", flush=True)
        print(f"    Settings: {kwargs}", flush=True)

        times = []
        for i in range(n_warmup + n_measure):
            img = Image.fromarray(create_highway(i + 100))
            inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=7, do_sample=False,
                    return_dict_in_generate=True,
                    **kwargs,
                )
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            elapsed = t1 - t0
            if i >= n_warmup:
                times.append(elapsed)
                if i % 5 == 0:
                    print(f"    [{i-n_warmup+1}/{n_measure}] {elapsed*1000:.1f}ms", flush=True)

        mean_ms = np.mean(times) * 1000
        std_ms = np.std(times) * 1000
        min_ms = min(times) * 1000
        max_ms = max(times) * 1000
        results[config_name] = {
            'mean_ms': float(mean_ms),
            'std_ms': float(std_ms),
            'min_ms': float(min_ms),
            'max_ms': float(max_ms),
            'times_ms': [float(t * 1000) for t in times],
        }
        print(f"    Mean: {mean_ms:.1f} ± {std_ms:.1f} ms (min={min_ms:.1f}, max={max_ms:.1f})",
              flush=True)

    # Post-processing overhead
    print("\n  Post-processing overhead measurement:", flush=True)
    img = Image.fromarray(create_highway(999))
    inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=7, do_sample=False,
            return_dict_in_generate=True,
            output_hidden_states=True, output_scores=True,
        )

    # Time cosine distance computation
    if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
        last_step = outputs.hidden_states[-1]
        if isinstance(last_step, tuple):
            hidden = last_step[-1][0, -1, :].float().cpu().numpy()
        else:
            hidden = last_step[0, -1, :].float().cpu().numpy()

        centroid = np.random.randn(hidden.shape[0])
        t0 = time.perf_counter()
        for _ in range(1000):
            cos_dist = 1.0 - float(np.dot(
                hidden / (np.linalg.norm(hidden) + 1e-10),
                centroid / (np.linalg.norm(centroid) + 1e-10)))
        t1 = time.perf_counter()
        cosine_us = (t1 - t0) / 1000 * 1e6
        print(f"    Cosine distance: {cosine_us:.2f} µs", flush=True)

    # Time action mass computation
    if hasattr(outputs, 'scores') and outputs.scores:
        vocab_size = outputs.scores[0].shape[-1]
        action_start = vocab_size - 256
        t0 = time.perf_counter()
        for _ in range(1000):
            for score in outputs.scores[:7]:
                logits = score[0].float()
                probs = torch.softmax(logits, dim=0)
                mass = float(probs[action_start:].sum())
        t1 = time.perf_counter()
        mass_us = (t1 - t0) / 1000 * 1e6
        print(f"    Action mass (7 dims): {mass_us:.2f} µs", flush=True)

    # Summary
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 70, flush=True)

    baseline = results['baseline']['mean_ms']
    for name, r in results.items():
        overhead = r['mean_ms'] - baseline
        pct = (overhead / baseline) * 100
        print(f"  {name:<20} {r['mean_ms']:>7.1f} ms  "
              f"(overhead: {overhead:+.1f} ms, {pct:+.1f}%)", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown"
    output = {
        'experiment': 'latency_profile',
        'experiment_number': 60,
        'timestamp': timestamp,
        'gpu': gpu_name,
        'n_warmup': n_warmup,
        'n_measure': n_measure,
        'results': results,
        'post_processing': {
            'cosine_distance_us': float(cosine_us) if 'cosine_us' in dir() else None,
            'action_mass_us': float(mass_us) if 'mass_us' in dir() else None,
        },
    }
    output_path = os.path.join(RESULTS_DIR, f"latency_profile_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
