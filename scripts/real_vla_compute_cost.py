"""
Computational Cost Analysis on Real OpenVLA-7B.

Measures wall-clock time and memory for different UQ methods:
1. Baseline inference (action only, no scores/hidden)
2. With output_scores (for action mass)
3. With output_hidden_states (for cosine distance)
4. With both scores + hidden states
5. MC Dropout (N=5, 10, 20 passes)
6. Full forward pass (for attention patterns)

Also measures:
- Centroid computation time
- Cosine distance computation time
- Action mass extraction time
- OOD decision time (end-to-end)

Experiment 46 in the CalibDrive series.
"""
import os
import json
import time
import datetime
import numpy as np
import torch
from PIL import Image

RESULTS_DIR = "/workspace/Vizuara-VLA-Research/experiments"
os.makedirs(RESULTS_DIR, exist_ok=True)
SIZE = (256, 256)


def create_test_image(idx):
    rng = np.random.default_rng(idx * 9001)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]
    img[SIZE[0]//2:] = [80, 80, 80]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def main():
    print("=" * 70, flush=True)
    print("COMPUTATIONAL COST ANALYSIS", flush=True)
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

    # Warmup
    print("\nWarmup (2 inferences)...", flush=True)
    for i in range(2):
        img = Image.fromarray(create_test_image(i + 100))
        inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=7, do_sample=False)
    torch.cuda.synchronize()

    N_TRIALS = 10
    results = {}

    # Method 1: Baseline (no scores, no hidden)
    print("\n1. Baseline inference...", flush=True)
    times = []
    for i in range(N_TRIALS):
        img = Image.fromarray(create_test_image(i + 200))
        inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=7, do_sample=False)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    results['baseline'] = {'mean': np.mean(times), 'std': np.std(times), 'times': times}
    print(f"  Baseline: {np.mean(times)*1000:.1f} ± {np.std(times)*1000:.1f} ms", flush=True)

    # Method 2: With output_scores (action mass)
    print("2. With output_scores...", flush=True)
    times = []
    for i in range(N_TRIALS):
        img = Image.fromarray(create_test_image(i + 300))
        inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=7, do_sample=False,
                                     output_scores=True, return_dict_in_generate=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        # Extract action mass
        vocab_size = outputs.scores[0].shape[-1]
        action_start = vocab_size - 256
        masses = []
        for score in outputs.scores[:7]:
            probs = torch.softmax(score[0].float(), dim=0)
            masses.append(float(probs[action_start:].sum()))
        t2 = time.perf_counter()
        times.append(t1 - t0)
    results['scores_only'] = {'mean': np.mean(times), 'std': np.std(times), 'times': times}
    print(f"  Scores: {np.mean(times)*1000:.1f} ± {np.std(times)*1000:.1f} ms", flush=True)

    # Method 3: With output_hidden_states (cosine distance)
    print("3. With output_hidden_states...", flush=True)
    times = []
    hidden_extract_times = []
    for i in range(N_TRIALS):
        img = Image.fromarray(create_test_image(i + 400))
        inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=7, do_sample=False,
                                     output_hidden_states=True, return_dict_in_generate=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        # Extract hidden state
        last_step = outputs.hidden_states[-1]
        if isinstance(last_step, tuple):
            hidden = last_step[-1][0, -1, :].float().cpu().numpy()
        else:
            hidden = last_step[0, -1, :].float().cpu().numpy()
        t2 = time.perf_counter()
        times.append(t1 - t0)
        hidden_extract_times.append(t2 - t1)
    results['hidden_only'] = {'mean': np.mean(times), 'std': np.std(times), 'times': times}
    print(f"  Hidden: {np.mean(times)*1000:.1f} ± {np.std(times)*1000:.1f} ms "
          f"(+{np.mean(hidden_extract_times)*1000:.2f} ms extraction)", flush=True)

    # Method 4: Both scores + hidden states
    print("4. Both scores + hidden states...", flush=True)
    times = []
    for i in range(N_TRIALS):
        img = Image.fromarray(create_test_image(i + 500))
        inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=7, do_sample=False,
                                     output_scores=True, output_hidden_states=True,
                                     return_dict_in_generate=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    results['both'] = {'mean': np.mean(times), 'std': np.std(times), 'times': times}
    print(f"  Both: {np.mean(times)*1000:.1f} ± {np.std(times)*1000:.1f} ms", flush=True)

    # Method 5: MC Dropout
    print("5. MC Dropout...", flush=True)
    # Enable dropout
    n_dropout = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.20
            module.train()
            n_dropout += 1

    for n_passes in [5, 10, 20]:
        times = []
        for i in range(3):  # fewer trials for MC Dropout
            img = Image.fromarray(create_test_image(i + 600 + n_passes))
            inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for p in range(n_passes):
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=7, do_sample=False,
                                             output_scores=True, return_dict_in_generate=True)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)
        results[f'mc_dropout_{n_passes}'] = {'mean': np.mean(times), 'std': np.std(times),
                                              'times': times, 'n_passes': n_passes}
        print(f"  MC Dropout (N={n_passes}): {np.mean(times)*1000:.1f} ± "
              f"{np.std(times)*1000:.1f} ms total, "
              f"{np.mean(times)/n_passes*1000:.1f} ms/pass", flush=True)

    # Reset dropout
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
            module.eval()

    # Post-processing costs
    print("\n6. Post-processing costs...", flush=True)

    # Generate calibration centroid
    cal_hidden = []
    for i in range(25):
        img = Image.fromarray(create_test_image(i + 700))
        inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=7, do_sample=False,
                                     output_hidden_states=True, return_dict_in_generate=True)
        last_step = outputs.hidden_states[-1]
        if isinstance(last_step, tuple):
            hidden = last_step[-1][0, -1, :].float().cpu().numpy()
        else:
            hidden = last_step[0, -1, :].float().cpu().numpy()
        cal_hidden.append(hidden)

    # Centroid computation
    t0 = time.perf_counter()
    for _ in range(1000):
        centroid = np.mean(cal_hidden, axis=0)
    t1 = time.perf_counter()
    centroid_time = (t1 - t0) / 1000
    print(f"  Centroid computation (25 samples): {centroid_time*1e6:.1f} μs", flush=True)

    # Cosine distance computation
    test_hidden = cal_hidden[0]
    t0 = time.perf_counter()
    for _ in range(10000):
        a = test_hidden / (np.linalg.norm(test_hidden) + 1e-10)
        b = centroid / (np.linalg.norm(centroid) + 1e-10)
        cos_dist = 1.0 - float(np.dot(a, b))
    t1 = time.perf_counter()
    cosine_time = (t1 - t0) / 10000
    print(f"  Cosine distance: {cosine_time*1e6:.1f} μs", flush=True)

    # Per-scene min cosine (4 centroids)
    centroids_4 = [np.mean(cal_hidden[i*6:(i+1)*6], axis=0) for i in range(4)]
    t0 = time.perf_counter()
    for _ in range(10000):
        min_cos = min(1.0 - float(np.dot(
            test_hidden / (np.linalg.norm(test_hidden) + 1e-10),
            c / (np.linalg.norm(c) + 1e-10)
        )) for c in centroids_4)
    t1 = time.perf_counter()
    perscene_time = (t1 - t0) / 10000
    print(f"  Per-scene min cosine (4 centroids): {perscene_time*1e6:.1f} μs", flush=True)

    # GPU memory
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated() / 1e9
        mem_reserved = torch.cuda.memory_reserved() / 1e9
        mem_max = torch.cuda.max_memory_allocated() / 1e9
        print(f"\n  GPU Memory: allocated={mem_allocated:.2f}GB, "
              f"reserved={mem_reserved:.2f}GB, peak={mem_max:.2f}GB", flush=True)
        results['gpu_memory'] = {
            'allocated_gb': mem_allocated,
            'reserved_gb': mem_reserved,
            'peak_gb': mem_max,
        }

    # Summary table
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY: End-to-End OOD Detection Latency", flush=True)
    print("=" * 70, flush=True)

    baseline_ms = results['baseline']['mean'] * 1000
    print(f"\n  {'Method':<40} {'Latency (ms)':>15} {'Overhead':>12}", flush=True)
    print("  " + "-" * 70, flush=True)
    print(f"  {'Baseline (action only)':<40} {baseline_ms:>12.1f} ms {'':>12}", flush=True)

    for name, key in [
        ('+ Action mass', 'scores_only'),
        ('+ Cosine distance', 'hidden_only'),
        ('+ Both (cos + mass)', 'both'),
    ]:
        ms = results[key]['mean'] * 1000
        overhead = (ms - baseline_ms) / baseline_ms * 100
        print(f"  {name:<40} {ms:>12.1f} ms {overhead:>+10.1f}%", flush=True)

    for n in [5, 10, 20]:
        ms = results[f'mc_dropout_{n}']['mean'] * 1000
        overhead = (ms - baseline_ms) / baseline_ms * 100
        print(f"  {'MC Dropout (N=' + str(n) + ')':<40} {ms:>12.1f} ms "
              f"{overhead:>+10.1f}%", flush=True)

    print(f"\n  Post-processing (negligible):", flush=True)
    print(f"    Centroid computation: {centroid_time*1e6:.1f} μs (one-time)", flush=True)
    print(f"    Cosine distance: {cosine_time*1e6:.1f} μs", flush=True)
    print(f"    Per-scene min (4): {perscene_time*1e6:.1f} μs", flush=True)

    results['post_processing'] = {
        'centroid_us': centroid_time * 1e6,
        'cosine_us': cosine_time * 1e6,
        'perscene_us': perscene_time * 1e6,
    }

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'compute_cost',
        'experiment_number': 46,
        'timestamp': timestamp,
        'n_trials': N_TRIALS,
        'results': {k: {kk: vv for kk, vv in v.items() if kk != 'times'}
                    for k, v in results.items()},
    }
    output_path = os.path.join(RESULTS_DIR, f"compute_cost_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
