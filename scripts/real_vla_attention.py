"""
Experiment 216: Attention Pattern Analysis
How does OOD corruption change the model's attention distributions?
Compares attention entropy and attention distance between clean and corrupted inputs.
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
    elif name == 'blur':
        return img.filter(ImageFilter.GaussianBlur(radius=5))
    elif name == 'noise':
        arr = arr + rng.normal(0, 40, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def compute_attention_stats(attn_weights):
    """Compute statistics from attention weights [n_heads, seq_len, seq_len]."""
    # Average over heads
    avg_attn = attn_weights.mean(dim=0)  # [seq_len, seq_len]
    
    # Entropy of attention distribution (per query token, averaged)
    eps = 1e-10
    entropy = -(avg_attn * torch.log(avg_attn + eps)).sum(dim=-1).mean().item()
    
    # Attention concentration: max attention weight per query (averaged)
    max_attn = avg_attn.max(dim=-1).values.mean().item()
    
    # Per-head entropy
    head_entropies = -(attn_weights * torch.log(attn_weights + eps)).sum(dim=-1).mean(dim=-1)
    
    return {
        'entropy': entropy,
        'max_attn': max_attn,
        'head_entropy_mean': head_entropies.mean().item(),
        'head_entropy_std': head_entropies.std().item(),
    }

def main():
    print("=" * 60)
    print("Experiment 216: Attention Pattern Analysis")
    print("=" * 60)

    from transformers import AutoModelForVision2Seq, AutoProcessor
    print("Loading OpenVLA-7B...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to drive forward?\nOut:"
    rng = np.random.default_rng(42)
    n_samples = 8
    base_imgs = [make_driving_image() for _ in range(n_samples)]
    corruption_types = ['fog', 'night', 'blur', 'noise']
    
    # Layers to analyze attention at
    attn_layers = [0, 1, 3, 7, 15, 31]

    print("\n--- Extracting attention patterns ---")
    results = {}
    
    # Clean baseline
    clean_stats = {l: [] for l in attn_layers}
    for i in range(n_samples):
        inputs = processor(prompt, base_imgs[i]).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            fwd = model(**inputs, output_attentions=True)
        for l in attn_layers:
            if l < len(fwd.attentions):
                stats = compute_attention_stats(fwd.attentions[l][0])
                clean_stats[l].append(stats)
        if (i+1) % 4 == 0:
            print(f"  Clean: {i+1}/{n_samples}")
    
    # Average clean stats
    clean_avg = {}
    for l in attn_layers:
        if clean_stats[l]:
            clean_avg[f"L{l}"] = {
                'entropy': round(np.mean([s['entropy'] for s in clean_stats[l]]), 4),
                'max_attn': round(np.mean([s['max_attn'] for s in clean_stats[l]]), 4),
                'head_entropy_mean': round(np.mean([s['head_entropy_mean'] for s in clean_stats[l]]), 4),
            }
    results['clean'] = clean_avg
    print(f"  Clean entropy: {clean_avg}")

    # Corrupted
    for ctype in corruption_types:
        corr_stats = {l: [] for l in attn_layers}
        for i in range(n_samples):
            img = apply_corruption(base_imgs[i], ctype, rng)
            inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)
            with torch.no_grad():
                fwd = model(**inputs, output_attentions=True)
            for l in attn_layers:
                if l < len(fwd.attentions):
                    stats = compute_attention_stats(fwd.attentions[l][0])
                    corr_stats[l].append(stats)
        
        corr_avg = {}
        for l in attn_layers:
            if corr_stats[l]:
                clean_ent = np.mean([s['entropy'] for s in clean_stats[l]])
                corr_ent = np.mean([s['entropy'] for s in corr_stats[l]])
                corr_avg[f"L{l}"] = {
                    'entropy': round(corr_ent, 4),
                    'max_attn': round(np.mean([s['max_attn'] for s in corr_stats[l]]), 4),
                    'head_entropy_mean': round(np.mean([s['head_entropy_mean'] for s in corr_stats[l]]), 4),
                    'entropy_delta': round(corr_ent - clean_ent, 4),
                    'entropy_ratio': round(corr_ent / (clean_ent + 1e-10), 4),
                }
        results[ctype] = corr_avg
        print(f"  {ctype}: {corr_avg}")

    output = {
        "experiment": "attention_analysis",
        "experiment_number": 216,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "n_samples": n_samples,
        "attn_layers": attn_layers,
        "results": results,
    }

    out_path = f"/workspace/Vizuara-VLA-Research/experiments/attention_analysis_{output['timestamp']}.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
