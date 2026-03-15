"""
Computational Cost Profile for OOD Detection Pipeline.

This experiment precisely measures the computational cost of different
components of the OOD detection pipeline, to demonstrate the "zero overhead"
claim and quantify the full cost of deployment.

Analyses:
1. Forward Pass Timing: standard vs. output_hidden_states=True (20 warmup, 50 timed)
2. Hidden State Extraction Cost: time just the tensor slice for last token, layer 3
3. Cosine Distance Computation: time 1000 repetitions on (4096,) vectors
4. Full Detection Pipeline Timing: forward pass + extract + cosine + threshold
5. Memory Overhead: GPU memory with and without output_hidden_states=True
6. Batch Size Scaling: overhead at batch sizes 1, 2, 4

Experiment 452 in the CalibDrive series.
"""
import torch
import json
import os
import sys
import numpy as np
import time
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor
from datetime import datetime

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"
)
os.makedirs(RESULTS_DIR, exist_ok=True)


def make_image(seed=42):
    rng = np.random.RandomState(seed)
    return Image.fromarray(rng.randint(0, 255, (224, 224, 3), dtype=np.uint8))


def apply_corruption(image, ctype, severity=1.0):
    arr = np.array(image).astype(np.float32) / 255.0
    if ctype == "fog":
        arr = arr * (1 - 0.6 * severity) + 0.6 * severity
    elif ctype == "night":
        arr = arr * max(0.01, 1.0 - 0.95 * severity)
    elif ctype == "noise":
        arr = arr + np.random.RandomState(42).randn(*arr.shape) * 0.3 * severity
        arr = np.clip(arr, 0, 1)
    elif ctype == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=10 * severity))
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))


def time_forward_pass(model, processor, prompt, use_hidden_states, n_warmup, n_timed):
    """Run timed forward passes and return stats dict (times in ms)."""
    times_ms = []
    for i in range(n_warmup + n_timed):
        img = make_image(seed=i)
        inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(
                **inputs,
                output_hidden_states=use_hidden_states,
            )
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        if i >= n_warmup:
            times_ms.append((t1 - t0) * 1000.0)
        if i % 10 == 0:
            phase = "warmup" if i < n_warmup else f"timed {i - n_warmup + 1}/{n_timed}"
            print(
                f"    [{phase}] {(t1 - t0) * 1000:.1f} ms  "
                f"(hidden_states={use_hidden_states})",
                flush=True,
            )
    arr = np.array(times_ms)
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "times_ms": [float(v) for v in arr],
        "n_warmup": n_warmup,
        "n_timed": n_timed,
        "use_hidden_states": use_hidden_states,
    }


def time_hidden_state_extraction(hidden_states_tuple, layer_idx=3, n_reps=10000):
    """Time the tensor slice that extracts the last token embedding from one layer."""
    # Pre-select to avoid any Python overhead outside the loop
    layer_hs = hidden_states_tuple[layer_idx]  # shape: (batch, seq_len, hidden)
    times_us = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        _ = layer_hs[0, -1, :]
        t1 = time.perf_counter()
        times_us.append((t1 - t0) * 1e6)
    arr = np.array(times_us)
    return {
        "operation": "hidden_state_slice_last_token_layer3",
        "n_reps": n_reps,
        "mean_us": float(arr.mean()),
        "std_us": float(arr.std()),
        "min_us": float(arr.min()),
        "max_us": float(arr.max()),
    }


def time_cosine_distance(embedding_dim=4096, n_reps=1000):
    """Time cosine distance between two (embedding_dim,) numpy vectors."""
    rng = np.random.RandomState(7)
    vec = rng.randn(embedding_dim).astype(np.float32)
    centroid = rng.randn(embedding_dim).astype(np.float32)
    times_us = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        v_norm = vec / (np.linalg.norm(vec) + 1e-10)
        c_norm = centroid / (np.linalg.norm(centroid) + 1e-10)
        _dist = 1.0 - float(np.dot(v_norm, c_norm))
        t1 = time.perf_counter()
        times_us.append((t1 - t0) * 1e6)
    arr = np.array(times_us)
    return {
        "operation": "cosine_distance_numpy",
        "embedding_dim": embedding_dim,
        "n_reps": n_reps,
        "mean_us": float(arr.mean()),
        "std_us": float(arr.std()),
        "min_us": float(arr.min()),
        "max_us": float(arr.max()),
    }


def time_full_pipeline(model, processor, prompt, centroid_np, layer_idx=3, threshold=0.3, n_reps=20):
    """
    Time the complete OOD detection pipeline per inference call:
      forward pass (output_hidden_states=True) →
      extract last-token embedding from layer_idx →
      compute cosine distance to centroid →
      compare against threshold.
    """
    img = make_image(seed=999)
    inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)

    # warmup
    for _ in range(5):
        with torch.no_grad():
            model(**inputs, output_hidden_states=True)
    torch.cuda.synchronize()

    times_ms = []
    breakdown_extract_us = []
    breakdown_cosine_us = []
    breakdown_threshold_us = []

    for i in range(n_reps):
        img_i = make_image(seed=i + 500)
        inputs_i = processor(prompt, img_i).to(model.device, dtype=torch.bfloat16)

        torch.cuda.synchronize()
        t_start = time.perf_counter()

        # 1. Forward pass with hidden states
        with torch.no_grad():
            outputs = model(**inputs_i, output_hidden_states=True)

        torch.cuda.synchronize()
        t_after_forward = time.perf_counter()

        # 2. Extract embedding
        t_e0 = time.perf_counter()
        embedding = outputs.hidden_states[layer_idx][0, -1, :].float().cpu().numpy()
        t_e1 = time.perf_counter()

        # 3. Cosine distance
        t_c0 = time.perf_counter()
        e_norm = embedding / (np.linalg.norm(embedding) + 1e-10)
        c_norm = centroid_np / (np.linalg.norm(centroid_np) + 1e-10)
        dist = 1.0 - float(np.dot(e_norm, c_norm))
        t_c1 = time.perf_counter()

        # 4. Threshold comparison
        t_th0 = time.perf_counter()
        _is_ood = dist > threshold
        t_th1 = time.perf_counter()

        t_end = time.perf_counter()
        times_ms.append((t_end - t_start) * 1000.0)
        breakdown_extract_us.append((t_e1 - t_e0) * 1e6)
        breakdown_cosine_us.append((t_c1 - t_c0) * 1e6)
        breakdown_threshold_us.append((t_th1 - t_th0) * 1e6)

        if i % 5 == 0:
            print(f"    [full pipeline {i + 1}/{n_reps}] {times_ms[-1]:.1f} ms", flush=True)

    arr = np.array(times_ms)
    return {
        "operation": "full_ood_detection_pipeline",
        "layer_idx": layer_idx,
        "threshold": threshold,
        "n_reps": n_reps,
        "total_mean_ms": float(arr.mean()),
        "total_std_ms": float(arr.std()),
        "total_min_ms": float(arr.min()),
        "total_max_ms": float(arr.max()),
        "breakdown": {
            "extract_embedding_mean_us": float(np.mean(breakdown_extract_us)),
            "cosine_distance_mean_us": float(np.mean(breakdown_cosine_us)),
            "threshold_comparison_mean_us": float(np.mean(breakdown_threshold_us)),
        },
    }


def measure_memory_overhead(model, processor, prompt):
    """
    Compare peak GPU memory usage with and without output_hidden_states=True.
    Returns a dict with bytes allocated for each configuration.
    """
    results = {}
    img = make_image(seed=123)
    inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)

    for use_hs in (False, True):
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        with torch.no_grad():
            _ = model(**inputs, output_hidden_states=use_hs)
        torch.cuda.synchronize()
        after = torch.cuda.max_memory_allocated()
        peak_delta = after - before
        key = "with_hidden_states" if use_hs else "without_hidden_states"
        results[key] = {
            "peak_allocated_bytes": int(after),
            "delta_from_baseline_bytes": int(peak_delta),
            "delta_from_baseline_mb": float(peak_delta / 1024 / 1024),
        }
        print(
            f"    output_hidden_states={use_hs}: peak={after / 1024 / 1024:.1f} MB, "
            f"delta={peak_delta / 1024 / 1024:.2f} MB",
            flush=True,
        )

    if "with_hidden_states" in results and "without_hidden_states" in results:
        overhead_bytes = (
            results["with_hidden_states"]["peak_allocated_bytes"]
            - results["without_hidden_states"]["peak_allocated_bytes"]
        )
        results["memory_overhead_bytes"] = int(overhead_bytes)
        results["memory_overhead_mb"] = float(overhead_bytes / 1024 / 1024)

    return results


def time_batch_scaling(model, processor, prompt, batch_sizes=(1, 2, 4), n_warmup=5, n_timed=20):
    """
    Measure forward-pass time (with output_hidden_states=True) at different batch sizes.
    Uses multiple copies of the same image for simplicity.
    """
    results = {}
    for bs in batch_sizes:
        print(f"    Batch size {bs}...", flush=True)
        imgs = [make_image(seed=bs * 10 + j) for j in range(bs)]
        # Process each image individually and stack inputs
        input_list = [processor(prompt, img) for img in imgs]

        # Build a batched input dict by stacking tensors
        def stack_inputs(ilist):
            keys = ilist[0].keys()
            batched = {}
            for k in keys:
                tensors = [inp[k] for inp in ilist]
                # handle both (seq,) and (1, seq, ...) shapes
                if tensors[0].dim() == 1:
                    batched[k] = torch.stack(tensors, dim=0)
                else:
                    batched[k] = torch.cat(tensors, dim=0)
            return batched

        try:
            batched_inputs = stack_inputs(input_list)
        except Exception as exc:
            print(f"      Skipping batch_size={bs}: could not stack inputs ({exc})", flush=True)
            results[str(bs)] = {"error": str(exc)}
            continue

        batched_inputs = {
            k: v.to(model.device, dtype=torch.bfloat16)
            if v.is_floating_point()
            else v.to(model.device)
            for k, v in batched_inputs.items()
        }

        # warmup
        for _ in range(n_warmup):
            with torch.no_grad():
                model(**batched_inputs, output_hidden_states=True)
        torch.cuda.synchronize()

        times_ms = []
        for _ in range(n_timed):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                model(**batched_inputs, output_hidden_states=True)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

        arr = np.array(times_ms)
        results[str(bs)] = {
            "batch_size": bs,
            "mean_ms": float(arr.mean()),
            "std_ms": float(arr.std()),
            "min_ms": float(arr.min()),
            "max_ms": float(arr.max()),
            "per_sample_mean_ms": float(arr.mean() / bs),
        }
        print(
            f"      mean={arr.mean():.1f} ms  per-sample={arr.mean() / bs:.1f} ms",
            flush=True,
        )
    return results


def main():
    print("=" * 70, flush=True)
    print("EXPERIMENT 452: COMPUTATIONAL COST PROFILE", flush=True)
    print("OOD Detection Pipeline — Zero-Overhead Claim Verification", flush=True)
    print("=" * 70, flush=True)

    # ------------------------------------------------------------------ #
    # Model loading
    # ------------------------------------------------------------------ #
    print("\nLoading OpenVLA-7B...", flush=True)
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()
    print("Model loaded.", flush=True)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"GPU: {gpu_name}", flush=True)

    prompt = "In: What action should the robot take to pick up the object?\nOut:"

    N_WARMUP = 20
    N_TIMED = 50

    results = {
        "experiment": "computational_cost_profile",
        "experiment_number": 452,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "gpu": gpu_name,
        "model": "openvla/openvla-7b",
        "prompt": prompt,
    }

    # ------------------------------------------------------------------ #
    # Analysis 1: Forward Pass Timing
    # ------------------------------------------------------------------ #
    print(f"\n[1/6] Forward Pass Timing  (warmup={N_WARMUP}, timed={N_TIMED})", flush=True)
    print("  Without output_hidden_states:", flush=True)
    timing_no_hs = time_forward_pass(
        model, processor, prompt,
        use_hidden_states=False,
        n_warmup=N_WARMUP,
        n_timed=N_TIMED,
    )
    print("  With output_hidden_states=True:", flush=True)
    timing_with_hs = time_forward_pass(
        model, processor, prompt,
        use_hidden_states=True,
        n_warmup=N_WARMUP,
        n_timed=N_TIMED,
    )

    overhead_ms = timing_with_hs["mean_ms"] - timing_no_hs["mean_ms"]
    overhead_pct = (overhead_ms / timing_no_hs["mean_ms"]) * 100.0
    print(
        f"  Overhead: {overhead_ms:+.2f} ms  ({overhead_pct:+.2f}%)",
        flush=True,
    )
    results["forward_pass_timing"] = {
        "without_hidden_states": timing_no_hs,
        "with_hidden_states": timing_with_hs,
        "overhead_ms": float(overhead_ms),
        "overhead_percent": float(overhead_pct),
    }

    # ------------------------------------------------------------------ #
    # Analysis 2: Hidden State Extraction Cost
    # ------------------------------------------------------------------ #
    print("\n[2/6] Hidden State Extraction Cost", flush=True)
    # Run one forward pass to get actual hidden states
    img_ref = make_image(seed=0)
    inputs_ref = processor(prompt, img_ref).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        out_ref = model(**inputs_ref, output_hidden_states=True)

    extraction_result = time_hidden_state_extraction(
        out_ref.hidden_states, layer_idx=3, n_reps=10000
    )
    print(
        f"  Tensor slice (layer 3, last token): "
        f"{extraction_result['mean_us']:.3f} ± {extraction_result['std_us']:.3f} µs",
        flush=True,
    )
    results["hidden_state_extraction"] = extraction_result

    # ------------------------------------------------------------------ #
    # Analysis 3: Cosine Distance Computation
    # ------------------------------------------------------------------ #
    print("\n[3/6] Cosine Distance Computation  (1000 repetitions)", flush=True)
    # Use the real embedding dimension from the model
    hidden_dim = out_ref.hidden_states[3].shape[-1]
    cosine_result = time_cosine_distance(embedding_dim=hidden_dim, n_reps=1000)
    print(
        f"  Cosine distance on ({hidden_dim},) vectors: "
        f"{cosine_result['mean_us']:.3f} ± {cosine_result['std_us']:.3f} µs",
        flush=True,
    )
    results["cosine_distance_timing"] = cosine_result

    # ------------------------------------------------------------------ #
    # Analysis 4: Full Detection Pipeline Timing
    # ------------------------------------------------------------------ #
    print("\n[4/6] Full Detection Pipeline Timing  (20 repetitions)", flush=True)
    # Create a synthetic centroid from the reference embedding
    ref_embedding = out_ref.hidden_states[3][0, -1, :].float().cpu().numpy()
    centroid_np = ref_embedding + np.random.RandomState(1).randn(*ref_embedding.shape).astype(np.float32) * 0.05

    pipeline_result = time_full_pipeline(
        model, processor, prompt,
        centroid_np=centroid_np,
        layer_idx=3,
        threshold=0.3,
        n_reps=20,
    )
    print(
        f"  Total pipeline mean: {pipeline_result['total_mean_ms']:.2f} ms",
        flush=True,
    )
    print("  Breakdown (post-forward-pass overhead only):", flush=True)
    for op, val in pipeline_result["breakdown"].items():
        print(f"    {op}: {val:.3f} µs", flush=True)
    results["full_pipeline_timing"] = pipeline_result

    # ------------------------------------------------------------------ #
    # Analysis 5: Memory Overhead
    # ------------------------------------------------------------------ #
    print("\n[5/6] Memory Overhead", flush=True)
    memory_result = measure_memory_overhead(model, processor, prompt)
    results["memory_overhead"] = memory_result
    if "memory_overhead_mb" in memory_result:
        print(
            f"  Memory overhead from output_hidden_states=True: "
            f"{memory_result['memory_overhead_mb']:.2f} MB",
            flush=True,
        )

    # ------------------------------------------------------------------ #
    # Analysis 6: Batch Size Scaling
    # ------------------------------------------------------------------ #
    print("\n[6/6] Batch Size Scaling  (warmup=5, timed=20)", flush=True)
    batch_result = time_batch_scaling(
        model, processor, prompt,
        batch_sizes=(1, 2, 4),
        n_warmup=5,
        n_timed=20,
    )
    results["batch_size_scaling"] = batch_result

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 70, flush=True)

    baseline_ms = timing_no_hs["mean_ms"]
    print(f"  Baseline forward pass (no hidden states): {baseline_ms:.2f} ms", flush=True)
    print(
        f"  Forward pass + hidden states:              "
        f"{timing_with_hs['mean_ms']:.2f} ms  "
        f"(overhead: {overhead_ms:+.2f} ms, {overhead_pct:+.2f}%)",
        flush=True,
    )
    print(
        f"  Hidden state extraction (layer 3):         "
        f"{extraction_result['mean_us']:.3f} µs",
        flush=True,
    )
    print(
        f"  Cosine distance computation:               "
        f"{cosine_result['mean_us']:.3f} µs",
        flush=True,
    )
    print(
        f"  Threshold comparison:                      "
        f"{pipeline_result['breakdown']['threshold_comparison_mean_us']:.3f} µs",
        flush=True,
    )
    total_post_overhead_us = (
        extraction_result["mean_us"]
        + cosine_result["mean_us"]
        + pipeline_result["breakdown"]["threshold_comparison_mean_us"]
    )
    print(
        f"  Total post-forward-pass overhead:          {total_post_overhead_us:.3f} µs  "
        f"({total_post_overhead_us / 1000:.4f} ms)",
        flush=True,
    )
    results["summary"] = {
        "baseline_forward_pass_ms": float(baseline_ms),
        "with_hidden_states_ms": float(timing_with_hs["mean_ms"]),
        "forward_pass_overhead_ms": float(overhead_ms),
        "forward_pass_overhead_percent": float(overhead_pct),
        "post_forward_overhead_total_us": float(total_post_overhead_us),
        "post_forward_overhead_total_ms": float(total_post_overhead_us / 1000.0),
        "detection_adds_zero_overhead": abs(overhead_pct) < 1.0,
    }

    # ------------------------------------------------------------------ #
    # Save results
    # ------------------------------------------------------------------ #
    timestamp = results["timestamp"]
    output_path = os.path.join(RESULTS_DIR, f"computational_profile_{timestamp}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
