"""
Experiment 210: Quantization Effect on OOD Detection
Does INT8 or INT4 quantization of OpenVLA affect hidden-state OOD detection?
Tests bfloat16 (baseline) vs INT8 vs INT4 quantization.
"""
import torch, json, numpy as np, os, sys
from datetime import datetime
from pathlib import Path
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
    elif name == 'blur':
        return img.filter(ImageFilter.GaussianBlur(radius=5))
    elif name == 'noise':
        arr = arr + rng.normal(0, 40, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def extract_hidden(model, processor, image, prompt, layers, quantized=False):
    inputs = processor(prompt, image)
    if quantized:
        inputs = inputs.to("cuda")
    else:
        inputs = inputs.to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return {l: fwd.hidden_states[l][0, -1, :].float().cpu().numpy() for l in layers}

def cosine_dist(a, b):
    return 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def compute_auroc(id_scores, ood_scores):
    id_scores = np.asarray(id_scores)
    ood_scores = np.asarray(ood_scores)
    n_id, n_ood = len(id_scores), len(ood_scores)
    if n_id == 0 or n_ood == 0:
        return 0.5
    count = sum(float(np.sum(o > id_scores) + 0.5 * np.sum(o == id_scores)) for o in ood_scores)
    return count / (n_id * n_ood)

def run_detection(model, processor, layers, prompt, base_imgs, rng, n_cal, n_test, quantized=False):
    corruption_types = ['fog', 'night', 'blur', 'noise']

    # Calibration
    id_embeds = {l: [] for l in layers}
    for i in range(n_cal):
        h = extract_hidden(model, processor, base_imgs[i], prompt, layers, quantized=quantized)
        for l in layers:
            id_embeds[l].append(h[l])
    centroids = {l: np.mean(id_embeds[l], axis=0) for l in layers}

    # Test
    results = {}
    for l in layers:
        id_scores = []
        for i in range(n_cal, n_cal + n_test):
            h = extract_hidden(model, processor, base_imgs[i], prompt, [l], quantized=quantized)
            id_scores.append(cosine_dist(h[l], centroids[l]))

        per_corr = {}
        all_ood = []
        for ctype in corruption_types:
            ood_scores = []
            for i in range(n_test):
                img = apply_corruption(base_imgs[i], ctype, rng)
                h = extract_hidden(model, processor, img, prompt, [l], quantized=quantized)
                d = cosine_dist(h[l], centroids[l])
                ood_scores.append(d)
                all_ood.append(d)
            per_corr[ctype] = round(compute_auroc(id_scores, ood_scores), 4)

        overall = round(compute_auroc(id_scores, all_ood), 4)
        results[f"L{l}"] = {
            "overall_auroc": overall,
            "per_corruption": per_corr,
            "id_mean_dist": round(float(np.mean(id_scores)), 6),
            "ood_mean_dist": round(float(np.mean(all_ood)), 6),
        }
    return results

def main():
    print("=" * 60)
    print("Experiment 210: Quantization Effect on OOD Detection")
    print("=" * 60)

    from transformers import AutoModelForVision2Seq, AutoProcessor

    layers = [1, 3, 32]
    prompt = "In: What action should the robot take to drive forward?\nOut:"
    n_cal, n_test = 10, 8
    rng = np.random.default_rng(42)
    base_imgs = [make_driving_image() for _ in range(20)]

    all_results = {}

    # --- BFloat16 baseline ---
    print("\n--- BFloat16 (baseline) ---")
    model_bf16 = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model_bf16.eval()

    bf16_results = run_detection(model_bf16, processor, layers, prompt, base_imgs, rng, n_cal, n_test)
    all_results["bfloat16"] = bf16_results
    for l in layers:
        print(f"  L{l}: AUROC={bf16_results[f'L{l}']['overall_auroc']}")

    # Get bf16 memory usage
    bf16_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"  GPU memory: {bf16_mem:.2f} GB")
    all_results["bfloat16"]["gpu_memory_gb"] = round(bf16_mem, 2)

    # Free bf16 model
    del model_bf16
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # --- INT8 ---
    print("\n--- INT8 quantization ---")
    try:
        from transformers import BitsAndBytesConfig
        quant_config_8 = BitsAndBytesConfig(load_in_8bit=True)
        model_int8 = AutoModelForVision2Seq.from_pretrained(
            "openvla/openvla-7b",
            quantization_config=quant_config_8,
            device_map="auto", trust_remote_code=True)
        model_int8.eval()

        int8_results = run_detection(model_int8, processor, layers, prompt, base_imgs, rng, n_cal, n_test, quantized=True)
        all_results["int8"] = int8_results
        for l in layers:
            print(f"  L{l}: AUROC={int8_results[f'L{l}']['overall_auroc']}")

        int8_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"  GPU memory: {int8_mem:.2f} GB")
        all_results["int8"]["gpu_memory_gb"] = round(int8_mem, 2)

        del model_int8
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    except Exception as e:
        print(f"  INT8 failed: {e}")
        all_results["int8"] = {"error": str(e)}

    # --- INT4 ---
    print("\n--- INT4 quantization ---")
    try:
        quant_config_4 = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        model_int4 = AutoModelForVision2Seq.from_pretrained(
            "openvla/openvla-7b",
            quantization_config=quant_config_4,
            device_map="auto", trust_remote_code=True)
        model_int4.eval()

        int4_results = run_detection(model_int4, processor, layers, prompt, base_imgs, rng, n_cal, n_test, quantized=True)
        all_results["int4"] = int4_results
        for l in layers:
            print(f"  L{l}: AUROC={int4_results[f'L{l}']['overall_auroc']}")

        int4_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"  GPU memory: {int4_mem:.2f} GB")
        all_results["int4"]["gpu_memory_gb"] = round(int4_mem, 2)

        del model_int4
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  INT4 failed: {e}")
        all_results["int4"] = {"error": str(e)}

    # Cross-precision centroid comparison
    # (if both bf16 and int8 succeeded, compare their centroids)
    print("\n--- Cross-precision analysis ---")

    output = {
        "experiment": "quantization_effect",
        "experiment_number": 210,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "n_cal": n_cal,
        "n_test": n_test,
        "layers": [1, 3, 32],
        "results": all_results,
    }

    out_path = f"/workspace/Vizuara-VLA-Research/experiments/quantization_effect_{output['timestamp']}.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
