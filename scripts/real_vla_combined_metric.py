#!/usr/bin/env python3
"""Experiment 274: Combined Metric for Robust Detection
Tests combining cosine distance with other metrics (L2 norm change,
attention entropy change, action token entropy) to create a more
robust composite detector.
"""
import torch, json, numpy as np
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor
from datetime import datetime

def extract_all(model, processor, image, prompt):
    """Extract hidden state, attention, and logits in one forward pass."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True, output_attentions=True)

    h3 = fwd.hidden_states[3][0, -1, :].float().cpu().numpy()
    h31 = fwd.hidden_states[31][0, -1, :].float().cpu().numpy()

    # L31 attention entropy (last head avg)
    attn = fwd.attentions[31][0].float().cpu().numpy()  # (n_heads, seq, seq)
    last_attn = attn[:, -1, :]  # attention from last token
    entropies = []
    for h_idx in range(attn.shape[0]):
        p = last_attn[h_idx]
        p = p[p > 0]
        ent = -np.sum(p * np.log(p))
        entropies.append(ent)
    attn_entropy = float(np.mean(entropies))

    # Next token logits
    logits = fwd.logits[0, -1, 31744:32000].float().cpu().numpy()
    logits_shifted = logits - logits.max()
    probs = np.exp(logits_shifted) / np.exp(logits_shifted).sum()
    output_entropy = float(-np.sum(probs[probs > 0] * np.log(probs[probs > 0])))
    top1_prob = float(probs.max())

    return {
        'h3': h3, 'h31': h31,
        'l2_norm_31': float(np.linalg.norm(h31)),
        'attn_entropy_31': attn_entropy,
        'output_entropy': output_entropy,
        'top1_prob': top1_prob
    }

def cosine_distance(a, b):
    return 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

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
    if n_id == 0 or n_ood == 0: return 0.5
    count = sum(float(np.sum(o > id_scores) + 0.5 * np.sum(o == id_scores)) for o in ood_scores)
    return count / (n_id * n_ood)

print("=" * 60)
print("Experiment 274: Combined Metric for Robust Detection")
print("=" * 60)

print("Loading OpenVLA-7B...")
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", torch_dtype=torch.bfloat16,
    device_map="auto", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
model.eval()

prompt = "In: What action should the robot take to pick up the object?\nOut:"

# Generate 5 diverse scenes
scenes = []
for seed in range(5):
    np.random.seed(seed * 100)
    pixels = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    scenes.append(Image.fromarray(pixels))

corruptions = ['fog', 'night', 'noise', 'blur']

results = {}
all_metrics = {'id': {m: [] for m in ['cos_d_3', 'cos_d_31', 'norm_change', 'attn_ent_change', 'output_ent', 'top1_prob']}}
for c in corruptions:
    all_metrics[c] = {m: [] for m in all_metrics['id']}

# Extract clean references
clean_data = []
for scene in scenes:
    clean = extract_all(model, processor, scene, prompt)
    clean_data.append(clean)

# ID metrics (each scene compared to itself)
for i, scene in enumerate(scenes):
    ref = extract_all(model, processor, scene, prompt)
    d3 = cosine_distance(ref['h3'], clean_data[i]['h3'])
    d31 = cosine_distance(ref['h31'], clean_data[i]['h31'])
    norm_change = abs(ref['l2_norm_31'] - clean_data[i]['l2_norm_31']) / clean_data[i]['l2_norm_31']
    attn_change = abs(ref['attn_entropy_31'] - clean_data[i]['attn_entropy_31'])
    all_metrics['id']['cos_d_3'].append(d3)
    all_metrics['id']['cos_d_31'].append(d31)
    all_metrics['id']['norm_change'].append(norm_change)
    all_metrics['id']['attn_ent_change'].append(attn_change)
    all_metrics['id']['output_ent'].append(ref['output_entropy'])
    all_metrics['id']['top1_prob'].append(ref['top1_prob'])

# OOD metrics
for ctype in corruptions:
    print(f"\n--- {ctype} ---")
    for i, scene in enumerate(scenes):
        corrupted = apply_corruption(scene, ctype)
        corr = extract_all(model, processor, corrupted, prompt)

        d3 = cosine_distance(corr['h3'], clean_data[i]['h3'])
        d31 = cosine_distance(corr['h31'], clean_data[i]['h31'])
        norm_change = abs(corr['l2_norm_31'] - clean_data[i]['l2_norm_31']) / clean_data[i]['l2_norm_31']
        attn_change = abs(corr['attn_entropy_31'] - clean_data[i]['attn_entropy_31'])

        all_metrics[ctype]['cos_d_3'].append(d3)
        all_metrics[ctype]['cos_d_31'].append(d31)
        all_metrics[ctype]['norm_change'].append(norm_change)
        all_metrics[ctype]['attn_ent_change'].append(attn_change)
        all_metrics[ctype]['output_ent'].append(corr['output_entropy'])
        all_metrics[ctype]['top1_prob'].append(corr['top1_prob'])

    for m in all_metrics[ctype]:
        vals = all_metrics[ctype][m]
        print(f"  {m}: mean={np.mean(vals):.6f}")

# Compute AUROC for each individual metric
print("\n=== INDIVIDUAL METRIC AUROC ===")
metric_aurocs = {}
for metric_name in ['cos_d_3', 'cos_d_31', 'norm_change', 'attn_ent_change']:
    id_vals = all_metrics['id'][metric_name]
    for ctype in corruptions:
        ood_vals = all_metrics[ctype][metric_name]
        auroc = compute_auroc(id_vals, ood_vals)
        key = f"{metric_name}_{ctype}"
        metric_aurocs[key] = float(auroc)
        print(f"  {metric_name} vs {ctype}: AUROC={auroc:.3f}")

# Combined metric: weighted sum
print("\n=== COMBINED METRIC AUROC ===")
combined_aurocs = {}
for w3, w31, w_norm, w_attn in [(1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1),
                                  (1,1,0,0), (1,0,1,0), (1,0,0,1), (1,1,1,1)]:
    weight_name = f"w3={w3}_w31={w31}_wnorm={w_norm}_wattn={w_attn}"
    for ctype in corruptions:
        id_combined = [w3*all_metrics['id']['cos_d_3'][i] +
                       w31*all_metrics['id']['cos_d_31'][i] +
                       w_norm*all_metrics['id']['norm_change'][i] +
                       w_attn*all_metrics['id']['attn_ent_change'][i]
                       for i in range(5)]
        ood_combined = [w3*all_metrics[ctype]['cos_d_3'][i] +
                        w31*all_metrics[ctype]['cos_d_31'][i] +
                        w_norm*all_metrics[ctype]['norm_change'][i] +
                        w_attn*all_metrics[ctype]['attn_ent_change'][i]
                        for i in range(5)]
        auroc = compute_auroc(id_combined, ood_combined)
        key = f"{weight_name}_{ctype}"
        combined_aurocs[key] = float(auroc)

    all_aurocs = [combined_aurocs[f"{weight_name}_{c}"] for c in corruptions]
    print(f"  {weight_name}: mean AUROC={np.mean(all_aurocs):.3f}, min={min(all_aurocs):.3f}")

results = {
    'metrics': {k: {m: [float(v) for v in vals] for m, vals in d.items()} for k, d in all_metrics.items()},
    'individual_aurocs': metric_aurocs,
    'combined_aurocs': combined_aurocs,
}

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
out = {
    'experiment': 'combined_metric',
    'experiment_number': 274,
    'timestamp': ts,
    'results': results
}

path = f'/workspace/Vizuara-VLA-Research/experiments/combined_metric_{ts}.json'
with open(path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {path}")
