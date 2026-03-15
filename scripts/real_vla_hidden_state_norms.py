"""
Hidden State Norm Analysis.

Analyzes the L2 norm of hidden state vectors across layers for ID vs OOD inputs.
Tests whether norm-based statistics (mean, variance, ratio) can serve as
simple OOD signals without computing centroids.

Key question: Do OOD inputs produce hidden states with different norms?

Experiment 137 in the CalibDrive series.
"""
import os
import json
import datetime
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

RESULTS_DIR = "/workspace/Vizuara-VLA-Research/experiments"
os.makedirs(RESULTS_DIR, exist_ok=True)
SIZE = (256, 256)


def create_highway(idx):
    rng = np.random.default_rng(idx * 30001)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]
    img[SIZE[0]//2:] = [80, 80, 80]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [255, 255, 255]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_urban(idx):
    rng = np.random.default_rng(idx * 30002)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [135, 206, 235]
    img[SIZE[0]//3:SIZE[0]//2] = [139, 119, 101]
    img[SIZE[0]//2:] = [60, 60, 60]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_noise(idx):
    rng = np.random.default_rng(idx * 30003)
    return rng.integers(0, 256, (*SIZE, 3), dtype=np.uint8)

def create_indoor(idx):
    rng = np.random.default_rng(idx * 30004)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:] = [200, 180, 160]
    img[SIZE[0]//2:, :] = [100, 80, 60]
    img[:SIZE[0]//3, SIZE[1]//3:2*SIZE[1]//3] = [150, 200, 255]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_twilight(idx):
    rng = np.random.default_rng(idx * 30010)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [70, 50, 80]
    img[SIZE[0]//2:] = [60, 60, 60]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [200, 200, 100]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_snow(idx):
    rng = np.random.default_rng(idx * 30014)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [200, 200, 210]
    img[SIZE[0]//2:] = [220, 220, 230]
    img[SIZE[0]//2:, SIZE[1]//2-2:SIZE[1]//2+2] = [180, 180, 190]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def main():
    print("=" * 70, flush=True)
    print("HIDDEN STATE NORM ANALYSIS", flush=True)
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

    categories = {
        'highway': (create_highway, 'ID'),
        'urban': (create_urban, 'ID'),
        'noise': (create_noise, 'OOD'),
        'indoor': (create_indoor, 'OOD'),
        'twilight': (create_twilight, 'OOD'),
        'snow': (create_snow, 'OOD'),
    }

    # Layers to analyze
    layer_indices = [0, 3, 8, 16, 24, 28, 32]

    print("\n--- Collecting hidden state norms ---", flush=True)
    all_norms = {l: {'id': [], 'ood': []} for l in layer_indices}
    labels = []

    for cat_name, (fn, group) in categories.items():
        print(f"  {cat_name} ({group})...", flush=True)
        for i in range(10):
            img = Image.fromarray(fn(i + 3500))
            inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)
            with torch.no_grad():
                fwd = model(**inputs, output_hidden_states=True)
            if hasattr(fwd, 'hidden_states') and fwd.hidden_states:
                for layer_idx in layer_indices:
                    if layer_idx < len(fwd.hidden_states):
                        h = fwd.hidden_states[layer_idx][0, -1, :].float().cpu().numpy()
                        norm = float(np.linalg.norm(h))
                        if group == 'ID':
                            all_norms[layer_idx]['id'].append(norm)
                        else:
                            all_norms[layer_idx]['ood'].append(norm)

    # Analyze per layer
    print("\n--- Per-Layer Norm Analysis ---", flush=True)
    results = {}
    for layer_idx in layer_indices:
        id_norms = np.array(all_norms[layer_idx]['id'])
        ood_norms = np.array(all_norms[layer_idx]['ood'])

        # AUROC using norm as score
        labels = np.array([0]*len(id_norms) + [1]*len(ood_norms))
        scores = np.concatenate([id_norms, ood_norms])
        try:
            auroc_high = float(roc_auc_score(labels, scores))  # higher norm = OOD
            auroc_low = float(roc_auc_score(labels, -scores))   # lower norm = OOD
            auroc = max(auroc_high, auroc_low)
            direction = 'higher' if auroc_high >= auroc_low else 'lower'
        except:
            auroc = 0.5
            direction = 'neither'

        d = float(abs(np.mean(ood_norms) - np.mean(id_norms)) / (np.std(id_norms) + 1e-10))

        results[str(layer_idx)] = {
            'layer': layer_idx,
            'id_mean': float(np.mean(id_norms)),
            'id_std': float(np.std(id_norms)),
            'ood_mean': float(np.mean(ood_norms)),
            'ood_std': float(np.std(ood_norms)),
            'auroc': auroc,
            'd_prime': d,
            'direction': direction,
        }
        print(f"  Layer {layer_idx:2d}: ID norm={np.mean(id_norms):.1f}±{np.std(id_norms):.1f}, "
              f"OOD norm={np.mean(ood_norms):.1f}±{np.std(ood_norms):.1f}, "
              f"AUROC={auroc:.4f} ({direction}), d={d:.2f}", flush=True)

    # Multi-layer norm feature
    print("\n--- Multi-Layer Norm Features ---", flush=True)
    # Collect norm vectors
    id_norm_vecs = []
    ood_norm_vecs = []
    for i in range(len(all_norms[layer_indices[0]]['id'])):
        vec = [all_norms[l]['id'][i] for l in layer_indices]
        id_norm_vecs.append(vec)
    for i in range(len(all_norms[layer_indices[0]]['ood'])):
        vec = [all_norms[l]['ood'][i] for l in layer_indices]
        ood_norm_vecs.append(vec)

    id_norm_vecs = np.array(id_norm_vecs)
    ood_norm_vecs = np.array(ood_norm_vecs)
    all_vecs = np.vstack([id_norm_vecs, ood_norm_vecs])
    labels = np.array([0]*len(id_norm_vecs) + [1]*len(ood_norm_vecs))

    # L2 from ID centroid of norm vectors
    id_centroid = np.mean(id_norm_vecs, axis=0)
    scores = np.array([np.linalg.norm(v - id_centroid) for v in all_vecs])
    multi_auroc = float(roc_auc_score(labels, scores))
    id_s = scores[labels == 0]
    ood_s = scores[labels == 1]
    multi_d = float((np.mean(ood_s) - np.mean(id_s)) / (np.std(id_s) + 1e-10))
    print(f"  Multi-layer norm vector: AUROC={multi_auroc:.4f}, d={multi_d:.2f}", flush=True)

    # Norm ratio features
    if 3 in layer_indices and 32 in layer_indices:
        id_ratios = np.array(all_norms[3]['id']) / (np.array(all_norms[32]['id']) + 1e-10)
        ood_ratios = np.array(all_norms[3]['ood']) / (np.array(all_norms[32]['ood']) + 1e-10)
        labels_ratio = np.array([0]*len(id_ratios) + [1]*len(ood_ratios))
        scores_ratio = np.concatenate([id_ratios, ood_ratios])
        ratio_auroc = float(roc_auc_score(labels_ratio, scores_ratio))
        ratio_auroc = max(ratio_auroc, 1 - ratio_auroc)
        ratio_d = float(abs(np.mean(ood_ratios) - np.mean(id_ratios)) / (np.std(id_ratios) + 1e-10))
        print(f"  Layer 3/32 norm ratio: AUROC={ratio_auroc:.4f}, d={ratio_d:.2f}", flush=True)
        results['norm_ratio_3_32'] = {
            'auroc': ratio_auroc,
            'd_prime': ratio_d,
            'id_mean': float(np.mean(id_ratios)),
            'ood_mean': float(np.mean(ood_ratios)),
        }

    results['multi_layer'] = {
        'auroc': multi_auroc,
        'd_prime': multi_d,
    }

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'hidden_state_norms',
        'experiment_number': 137,
        'timestamp': timestamp,
        'layer_indices': layer_indices,
        'results': results,
    }
    output_path = os.path.join(RESULTS_DIR, f"hidden_state_norms_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
