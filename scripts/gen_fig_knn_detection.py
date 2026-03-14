"""Generate Figure 89: KNN-Based OOD Detection."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/knn_detection_20260314_213133.json") as f:
    data = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel (a): KNN AUROC by k and metric
ax = axes[0]
metrics = ['cosine', 'euclidean', 'manhattan']
k_values = [1, 3, 5, 10, 15, 20]
colors = {'cosine': '#4CAF50', 'euclidean': '#2196F3', 'manhattan': '#FF9800'}
markers = {'cosine': 'o', 'euclidean': 's', 'manhattan': 'D'}

for metric in metrics:
    aurocs = []
    ks = []
    for k in k_values:
        key = f"{metric}_k{k}"
        if key in data['knn_results']:
            aurocs.append(data['knn_results'][key]['auroc'])
            ks.append(k)
    ax.plot(ks, aurocs, color=colors[metric], marker=markers[metric],
            label=metric.capitalize(), linewidth=2, markersize=8, alpha=0.8)

ax.axhline(y=data['centroid_auroc'], color='red', linestyle='--', linewidth=1.5,
           label=f'Centroid ({data["centroid_auroc"]:.3f})')
ax.set_xlabel('k (neighbors)', fontsize=11)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(a) KNN AUROC by k and Metric', fontsize=12, fontweight='bold')
ax.set_ylim(0.95, 1.005)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel (b): Per-category KNN scores
ax = axes[1]
cats = ['highway', 'urban', 'noise', 'indoor', 'twilight', 'snow']
cat_labels = ['Highway\n(ID)', 'Urban\n(ID)', 'Noise\n(OOD)', 'Indoor\n(OOD)', 'Twilight\n(OOD)', 'Snow\n(OOD)']
cat_colors = ['#4CAF50', '#4CAF50', '#F44336', '#F44336', '#F44336', '#F44336']

positions = range(len(cats))
box_data = [data['per_category'][c]['scores'] for c in cats]
bp = ax.boxplot(box_data, positions=positions, patch_artist=True, widths=0.6)
for patch, color in zip(bp['boxes'], cat_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax.set_xticks(positions)
ax.set_xticklabels(cat_labels, fontsize=8)
ax.set_ylabel('KNN Cosine Distance (k=1)', fontsize=11)
ax.set_title('(b) Per-Category Score Distribution', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add separator line between ID and OOD
max_id = max(max(data['per_category']['highway']['scores']),
             max(data['per_category']['urban']['scores']))
min_ood = min(min(data['per_category'][c]['scores']) for c in ['noise', 'indoor', 'twilight', 'snow'])
gap_center = (max_id + min_ood) / 2
ax.axhline(y=gap_center, color='red', linestyle='--', alpha=0.5, label=f'Gap center ({gap_center:.3f})')
ax.legend(fontsize=8)

# Panel (c): ID/OOD separation ratio by k
ax = axes[2]
for metric in ['cosine']:
    ratios = []
    ks = []
    for k in k_values:
        key = f"{metric}_k{k}"
        if key in data['knn_results']:
            r = data['knn_results'][key]
            ratio = r['ood_mean'] / (r['id_mean'] + 1e-10)
            ratios.append(ratio)
            ks.append(k)
    ax.bar(range(len(ks)), ratios, color='#2196F3', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([f'k={k}' for k in ks], fontsize=9)

ax.set_ylabel('OOD/ID Score Ratio', fontsize=11)
ax.set_title('(c) Separation Ratio (Cosine KNN)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

for i, r in enumerate(ratios):
    ax.text(i, r + 0.2, f'{r:.1f}×', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig89_knn_detection.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig89_knn_detection.pdf', dpi=200, bbox_inches='tight')
print("Saved fig89_knn_detection.png/pdf")
