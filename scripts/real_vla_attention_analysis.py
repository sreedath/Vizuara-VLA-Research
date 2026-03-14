"""
Attention Pattern Analysis for OOD Detection.

Examines whether attention patterns differ between ID and OOD inputs.
Specifically:
1. Attention entropy (how diffuse attention is)
2. Fraction of attention on image vs text tokens
3. Attention concentration on specific image regions

Experiment 63 in the CalibDrive series.
"""
import os
import json
import datetime
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

def create_urban(idx):
    rng = np.random.default_rng(idx * 5002)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [135, 206, 235]
    img[SIZE[0]//3:SIZE[0]//2] = [139, 119, 101]
    img[SIZE[0]//2:] = [60, 60, 60]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_noise(idx):
    rng = np.random.default_rng(idx * 5003)
    return rng.integers(0, 256, (*SIZE, 3), dtype=np.uint8)

def create_indoor(idx):
    rng = np.random.default_rng(idx * 5004)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:] = [200, 180, 160]
    img[SIZE[0]//2:] = [139, 90, 43]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_inverted(idx):
    return 255 - create_highway(idx + 3000)

def create_blackout(idx):
    return np.zeros((*SIZE, 3), dtype=np.uint8)


def extract_attention_stats(model, processor, image, prompt):
    """Extract attention statistics from a forward pass."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)

    with torch.no_grad():
        out = model(**inputs, output_attentions=True)

    if hasattr(out, 'attentions') and out.attentions is not None:
        stats = {}
        for layer_idx in [-1, -4, -8, -16]:
            if abs(layer_idx) > len(out.attentions):
                continue
            attn = out.attentions[layer_idx]  # (1, heads, seq, seq)
            attn = attn[0].float().cpu().numpy()  # (heads, seq, seq)

            n_heads, seq_len, _ = attn.shape
            last_token_attn = attn[:, -1, :]  # (heads, seq)

            # Entropy per head
            entropies = []
            for h in range(n_heads):
                a = last_token_attn[h]
                a = a + 1e-10
                ent = -np.sum(a * np.log(a))
                entropies.append(float(ent))

            mean_entropy = np.mean(entropies)
            max_attns = [float(np.max(last_token_attn[h])) for h in range(n_heads)]
            mean_max_attn = np.mean(max_attns)

            top5_masses = []
            for h in range(n_heads):
                sorted_a = np.sort(last_token_attn[h])[::-1]
                top5_masses.append(float(sorted_a[:5].sum()))

            layer_name = f"L{len(out.attentions) + layer_idx}"
            stats[layer_name] = {
                'mean_entropy': float(mean_entropy),
                'std_entropy': float(np.std(entropies)),
                'mean_max_attn': float(mean_max_attn),
                'mean_top5_mass': float(np.mean(top5_masses)),
                'seq_len': int(seq_len),
            }

        return stats
    else:
        return None


def main():
    print("=" * 70, flush=True)
    print("ATTENTION PATTERN ANALYSIS", flush=True)
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

    test_fns = {
        'highway': (create_highway, False, 8),
        'urban': (create_urban, False, 8),
        'noise': (create_noise, True, 6),
        'indoor': (create_indoor, True, 6),
        'inverted': (create_inverted, True, 6),
        'blackout': (create_blackout, True, 6),
    }

    all_data = []
    cnt = 0
    total = sum(v[2] for v in test_fns.values())

    for scene, (fn, is_ood, n) in test_fns.items():
        for i in range(n):
            cnt += 1
            print(f"  [{cnt}/{total}] {scene}_{i}...", flush=True)
            try:
                stats = extract_attention_stats(
                    model, processor,
                    Image.fromarray(fn(i + 200)), prompt
                )
                if stats:
                    all_data.append({
                        'scenario': scene, 'is_ood': is_ood, 'stats': stats
                    })
                else:
                    print(f"    No attention data for {scene}_{i}", flush=True)
            except Exception as e:
                print(f"    Error: {e}", flush=True)
                all_data.append({
                    'scenario': scene, 'is_ood': is_ood,
                    'stats': None, 'error': str(e)
                })

    # Analysis
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    valid = [d for d in all_data if d.get('stats')]
    if not valid:
        print("  No valid attention data collected.", flush=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output = {
            'experiment': 'attention_analysis',
            'experiment_number': 63,
            'timestamp': timestamp,
            'n_samples': len(all_data),
            'n_valid': 0,
            'error': 'No attention data available',
        }
        output_path = os.path.join(RESULTS_DIR, f"attention_analysis_{timestamp}.json")
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nSaved to {output_path}", flush=True)
        return

    id_data = [d for d in valid if not d['is_ood']]
    ood_data = [d for d in valid if d['is_ood']]

    layers = list(valid[0]['stats'].keys())

    for layer in layers:
        print(f"\n  Layer {layer}:", flush=True)
        id_ent = [d['stats'][layer]['mean_entropy'] for d in id_data]
        ood_ent = [d['stats'][layer]['mean_entropy'] for d in ood_data]
        id_max = [d['stats'][layer]['mean_max_attn'] for d in id_data]
        ood_max = [d['stats'][layer]['mean_max_attn'] for d in ood_data]

        print(f"    Entropy: ID={np.mean(id_ent):.3f}±{np.std(id_ent):.3f}, "
              f"OOD={np.mean(ood_ent):.3f}±{np.std(ood_ent):.3f}", flush=True)
        print(f"    Max Attn: ID={np.mean(id_max):.4f}±{np.std(id_max):.4f}, "
              f"OOD={np.mean(ood_max):.4f}±{np.std(ood_max):.4f}", flush=True)

        from sklearn.metrics import roc_auc_score
        labels = [0]*len(id_data) + [1]*len(ood_data)
        for metric_name, id_vals, ood_vals in [
            ('entropy', id_ent, ood_ent),
            ('max_attn', id_max, ood_max),
        ]:
            try:
                auroc = roc_auc_score(labels, id_vals + ood_vals)
                auroc_neg = roc_auc_score(labels, [-v for v in id_vals + ood_vals])
                best = max(auroc, auroc_neg)
                print(f"    {metric_name} AUROC: {best:.3f}", flush=True)
            except Exception:
                pass

    # Per-scenario
    print("\n  Per-scenario attention entropy (last layer):", flush=True)
    last_layer = layers[-1] if layers else None
    if last_layer:
        for scene in sorted(set(d['scenario'] for d in valid)):
            scene_data = [d for d in valid if d['scenario'] == scene]
            ent = [d['stats'][last_layer]['mean_entropy'] for d in scene_data]
            print(f"    {scene:<12}: entropy={np.mean(ent):.3f}±{np.std(ent):.3f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'attention_analysis',
        'experiment_number': 63,
        'timestamp': timestamp,
        'n_samples': len(all_data),
        'n_valid': len(valid),
        'layers_analyzed': layers,
        'summary': {},
    }
    if last_layer:
        for scene in sorted(set(d['scenario'] for d in valid)):
            scene_data = [d for d in valid if d['scenario'] == scene]
            output['summary'][scene] = {
                'n': len(scene_data),
                'mean_entropy': float(np.mean([d['stats'][last_layer]['mean_entropy']
                                                for d in scene_data])),
                'mean_max_attn': float(np.mean([d['stats'][last_layer]['mean_max_attn']
                                                 for d in scene_data])),
            }

    output_path = os.path.join(RESULTS_DIR, f"attention_analysis_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
