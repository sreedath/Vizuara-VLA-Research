"""
Experiment 238: Output Logit Distribution Analysis
How do output logits change under corruption?
Tests whether corruption causes predictable logit distribution shifts
(entropy, top-k probability mass, action token probability).
"""
import torch, json, numpy as np, os
from datetime import datetime
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
    elif name == 'noise':
        arr = arr + rng.normal(0, 30, arr.shape)
    elif name == 'blur':
        return img.filter(ImageFilter.GaussianBlur(radius=5))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def main():
    print("=" * 60)
    print("Experiment 238: Output Logit Distribution Analysis")
    print("=" * 60)

    from transformers import AutoModelForVision2Seq, AutoProcessor
    print("Loading OpenVLA-7B...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to drive forward?\nOut:"
    base_img = make_driving_image()
    rng = np.random.default_rng(42)

    # Action token range
    ACTION_START = 31744
    ACTION_END = 31999

    def get_logit_stats(image):
        inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[0, -1, :].float().cpu().numpy()

        # Softmax
        logits_shifted = logits - logits.max()
        probs = np.exp(logits_shifted) / np.exp(logits_shifted).sum()

        # Entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        # Top-1 and top-5 probability
        sorted_probs = np.sort(probs)[::-1]
        top1_prob = float(sorted_probs[0])
        top5_prob = float(sorted_probs[:5].sum())
        top10_prob = float(sorted_probs[:10].sum())

        # Action token probability mass
        action_probs = probs[ACTION_START:ACTION_END+1]
        action_mass = float(action_probs.sum())

        # Top action token
        top_action_idx = ACTION_START + int(np.argmax(action_probs))
        top_action_prob = float(action_probs.max())

        # Top predicted token (global)
        top_token = int(np.argmax(probs))

        # KL divergence from uniform over action tokens
        uniform_action = np.ones(256) / 256
        kl_from_uniform = float(np.sum(action_probs * np.log((action_probs + 1e-10) / uniform_action)))

        return {
            "entropy": round(entropy, 4),
            "top1_prob": round(top1_prob, 6),
            "top5_prob": round(top5_prob, 6),
            "top10_prob": round(top10_prob, 6),
            "action_mass": round(action_mass, 6),
            "top_action_token": top_action_idx,
            "top_action_prob": round(top_action_prob, 6),
            "top_token": top_token,
            "kl_action_from_uniform": round(kl_from_uniform, 4),
        }

    # Clean
    print("\n--- Clean ---")
    clean_stats = get_logit_stats(base_img)
    print(f"  {clean_stats}")

    # Corrupted
    corruption_types = ['fog', 'night', 'noise', 'blur']
    results = {"clean": clean_stats}

    for ctype in corruption_types:
        print(f"\n--- {ctype} ---")
        rng_local = np.random.default_rng(42)
        img = apply_corruption(base_img, ctype, rng_local)
        stats = get_logit_stats(img)
        results[ctype] = stats
        print(f"  {stats}")

        # Compare to clean
        entropy_change = stats["entropy"] - clean_stats["entropy"]
        action_mass_change = stats["action_mass"] - clean_stats["action_mass"]
        print(f"  Entropy change: {entropy_change:+.4f}")
        print(f"  Action mass change: {action_mass_change:+.6f}")

    output = {
        "experiment": "logit_analysis",
        "experiment_number": 238,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "results": results,
    }

    out_path = f"/workspace/Vizuara-VLA-Research/experiments/logit_analysis_{output['timestamp']}.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
