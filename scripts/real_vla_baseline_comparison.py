#!/usr/bin/env python3
"""Experiment 283: Baseline OOD Detection Comparison
Compares our cosine distance detector against standard OOD methods:
1. Maximum Softmax Probability (MSP) - Hendrycks & Gimpel 2017
2. Energy Score - Liu et al. 2020
3. Output Entropy
4. Mahalanobis Distance - Lee et al. 2018
5. Feature Norm
Computes AUROC for each method across 4 corruptions.
"""
import torch, json, numpy as np
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor
from datetime import datetime

def apply_corruption(img, ctype, severity=1.0):
    arr = np.array(img).astype(np.float32) / 255.0
    if ctype == 'fog':
        arr = arr * (1 - 0.6 * severity) + 0.6 * severity
    elif ctype == 'night':
        arr = arr * max(0.01, 1.0 - 0.95 * severity)
    elif ctype == 'noise':
        arr = arr + np.random.RandomState(42).randn(*arr.shape) * 0.3 * severity
        arr = np.clip(arr, 0, 1)
    elif ctype == 'blur':
        return img.filter(ImageFilter.GaussianBlur(radius=10 * severity))
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))

def compute_auroc(id_scores, ood_scores):
    id_scores = np.asarray(id_scores)
    ood_scores = np.asarray(ood_scores)
    n_id, n_ood = len(id_scores), len(ood_scores)
    if n_id == 0 or n_ood == 0:
        return 0.5
    count = sum(float(np.sum(o > id_scores) + 0.5 * np.sum(o == id_scores)) for o in ood_scores)
    return count / (n_id * n_ood)

print("=" * 60)
print("Experiment 283: Baseline OOD Detection Comparison")
print("=" * 60)

print("Loading OpenVLA-7B...")
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", torch_dtype=torch.bfloat16,
    device_map="auto", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
model.eval()

prompt = "In: What action should the robot take to pick up the object?\nOut:"

# Generate 5 "in-distribution" images (different random seeds)
id_images = []
for seed in [42, 123, 456, 789, 1010]:
    np.random.seed(seed)
    pixels = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    id_images.append(Image.fromarray(pixels))

# Generate corrupted versions at severity 0.5 and 1.0
corruptions = ['fog', 'night', 'noise', 'blur']
ood_images = {}
for ctype in corruptions:
    ood_images[ctype] = []
    for id_img in id_images:
        ood_images[ctype].append(apply_corruption(id_img, ctype, 0.5))
        ood_images[ctype].append(apply_corruption(id_img, ctype, 1.0))

# Collect metrics for all images
def get_all_metrics(model, processor, image, prompt, centroid_l3=None):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)

    # Get logits
    logits = fwd.logits[0, -1, :].float().cpu()

    # 1. MSP (Maximum Softmax Probability) - lower = more OOD
    probs = torch.softmax(logits, dim=0)
    msp = float(probs.max())

    # 2. Energy Score - lower = more OOD (negated logSumExp)
    energy = float(-torch.logsumexp(logits, dim=0))

    # 3. Output Entropy - higher = more OOD
    log_probs = torch.log_softmax(logits, dim=0)
    entropy = float(-torch.sum(probs * log_probs))

    # 4. Cosine Distance (ours) - higher = more OOD
    h_l3 = fwd.hidden_states[3][0, -1, :].float().cpu().numpy()
    if centroid_l3 is not None:
        cos_d = 1.0 - np.dot(h_l3, centroid_l3) / (np.linalg.norm(h_l3) * np.linalg.norm(centroid_l3) + 1e-30)
    else:
        cos_d = 0.0

    # 5. Feature Norm (L2 norm of hidden state) - detect as |norm - clean_norm|
    l3_norm = float(np.linalg.norm(h_l3))

    # 6. Top-K probability (sum of top 5 probabilities)
    topk_prob = float(probs.topk(5).values.sum())

    return {
        'msp': msp,
        'energy': energy,
        'entropy': entropy,
        'cosine_distance': float(cos_d),
        'feature_norm': l3_norm,
        'topk_prob': topk_prob,
        'hidden_l3': h_l3
    }

# Get centroid from first ID image
print("\nComputing centroid from first ID image...")
first_metrics = get_all_metrics(model, processor, id_images[0], prompt)
centroid = first_metrics['hidden_l3']
clean_norm = first_metrics['feature_norm']

# Collect ID metrics (using same-image calibration for each)
print("\n=== IN-DISTRIBUTION METRICS ===")
id_metrics_list = []
for i, id_img in enumerate(id_images):
    # For cosine distance, use same-image centroid
    img_centroid = get_all_metrics(model, processor, id_img, prompt)['hidden_l3']
    metrics = get_all_metrics(model, processor, id_img, prompt, centroid_l3=img_centroid)
    id_metrics_list.append(metrics)
    print(f"  ID {i}: MSP={metrics['msp']:.4f}, Energy={metrics['energy']:.4f}, "
          f"Entropy={metrics['entropy']:.4f}, CosD={metrics['cosine_distance']:.6f}")

# Collect OOD metrics
print("\n=== OOD METRICS ===")
ood_metrics = {}
for ctype in corruptions:
    ood_metrics[ctype] = []
    for i, ood_img in enumerate(ood_images[ctype]):
        # Use the corresponding clean image's centroid
        clean_idx = i // 2
        img_centroid = get_all_metrics(model, processor, id_images[clean_idx], prompt)['hidden_l3']
        metrics = get_all_metrics(model, processor, ood_img, prompt, centroid_l3=img_centroid)
        ood_metrics[ctype].append(metrics)
    print(f"  {ctype}: MSP={np.mean([m['msp'] for m in ood_metrics[ctype]]):.4f}, "
          f"Energy={np.mean([m['energy'] for m in ood_metrics[ctype]]):.4f}, "
          f"CosD={np.mean([m['cosine_distance'] for m in ood_metrics[ctype]]):.6f}")

# Compute AUROC for each method and corruption
print("\n=== AUROC COMPARISON ===")
methods = ['cosine_distance', 'msp', 'energy', 'entropy', 'feature_norm', 'topk_prob']
# For norm, use |norm - clean_norm| as score
id_norms = [m['feature_norm'] for m in id_metrics_list]
clean_norm_mean = np.mean(id_norms)

auroc_results = {}
for method in methods:
    auroc_results[method] = {}
    for ctype in corruptions:
        if method == 'feature_norm':
            id_scores = [abs(m['feature_norm'] - clean_norm_mean) for m in id_metrics_list]
            ood_scores = [abs(m['feature_norm'] - clean_norm_mean) for m in ood_metrics[ctype]]
        elif method in ['msp', 'topk_prob']:
            # Lower MSP = more OOD, so negate for AUROC (higher = more OOD)
            id_scores = [-m[method] for m in id_metrics_list]
            ood_scores = [-m[method] for m in ood_metrics[ctype]]
        elif method == 'energy':
            # More negative energy = more OOD
            id_scores = [m[method] for m in id_metrics_list]
            ood_scores = [m[method] for m in ood_metrics[ctype]]
        else:
            # Higher = more OOD (cosine distance, entropy)
            id_scores = [m[method] for m in id_metrics_list]
            ood_scores = [m[method] for m in ood_metrics[ctype]]

        auroc = compute_auroc(id_scores, ood_scores)
        auroc_results[method][ctype] = auroc

    mean_auroc = np.mean(list(auroc_results[method].values()))
    print(f"  {method:20s}: fog={auroc_results[method]['fog']:.3f}, "
          f"night={auroc_results[method]['night']:.3f}, "
          f"noise={auroc_results[method]['noise']:.3f}, "
          f"blur={auroc_results[method]['blur']:.3f}, "
          f"MEAN={mean_auroc:.3f}")

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
# Exclude hidden states from saved metrics
clean_metrics_save = [{k: v for k, v in m.items() if k != 'hidden_l3'} for m in id_metrics_list]
ood_metrics_save = {ct: [{k: v for k, v in m.items() if k != 'hidden_l3'} for m in ms]
                    for ct, ms in ood_metrics.items()}

out = {
    'experiment': 'baseline_comparison',
    'experiment_number': 283,
    'timestamp': ts,
    'methods': methods,
    'n_id': len(id_images),
    'n_ood_per_corruption': len(ood_images['fog']),
    'results': {
        'auroc': auroc_results,
        'id_metrics': clean_metrics_save,
        'ood_metrics': ood_metrics_save
    }
}

path = f'/workspace/Vizuara-VLA-Research/experiments/baseline_{ts}.json'
with open(path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {path}")
