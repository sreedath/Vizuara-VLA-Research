"""Generate Figure 24: Multi-Centroid OOD Detection Comparison."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/multi_centroid_20260314_155725.json"
OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

with open(RESULTS) as f:
    data = json.load(f)

aurocs = data['method_aurocs']

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Panel (a): Overall AUROC comparison
ax = axes[0]
methods_sorted = sorted(aurocs.items(), key=lambda x: -x[1])
names = [m[0] for m in methods_sorted]
vals = [m[1] for m in methods_sorted]
colors_bar = ['#2196F3' if v >= 0.99 else '#64B5F6' if v >= 0.97 else '#90CAF9' for v in vals]

bars = ax.barh(range(len(names)), vals, color=colors_bar, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=9)
ax.set_xlabel('AUROC', fontsize=11)
ax.set_title('(a) Method Comparison', fontsize=12, fontweight='bold')
ax.set_xlim(0.95, 1.0)
ax.axvline(x=0.994, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Global centroid')
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

for bar, val in zip(bars, vals):
    ax.text(val - 0.001, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', ha='right', va='center', fontsize=9, fontweight='bold')

# Panel (b): Per-OOD-type AUROC for top 3 methods
ax = axes[1]
ood_types = ['noise', 'blank', 'indoor', 'inverted', 'checker', 'blackout']
# Hardcode from results
per_type = {
    'Global centroid': [1.000, 0.997, 1.000, 0.970, 1.000, 1.000],
    'Per-scene (2)': [1.000, 0.990, 1.000, 0.983, 1.000, 1.000],
    'KMeans k=2': [1.000, 1.000, 1.000, 0.970, 1.000, 1.000],
}

x = np.arange(len(ood_types))
width = 0.25
colors_m = ['#2196F3', '#FF9800', '#4CAF50']

for idx, (name, vals) in enumerate(per_type.items()):
    ax.bar(x + idx * width - width, vals, width, label=name,
           color=colors_m[idx], alpha=0.8, edgecolor='black', linewidth=0.5)

ax.set_xlabel('OOD Type', fontsize=11)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(b) Per-Type Detection', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(ood_types, fontsize=9, rotation=30)
ax.set_ylim(0.95, 1.01)
ax.legend(fontsize=8, loc='lower left')
ax.grid(True, alpha=0.3, axis='y')

# Panel (c): Number of centroids vs AUROC
ax = axes[2]
n_centroids = [1, 2, 2, 3, 5]
centroid_aurocs = [
    aurocs['Global centroid'],
    aurocs['Per-scene (2 centroids)'],
    aurocs['KMeans k=2'],
    aurocs['KMeans k=3'],
    aurocs['KMeans k=5'],
]
centroid_names = ['Global\n(1)', 'Per-scene\n(2)', 'KMeans\nk=2', 'KMeans\nk=3', 'KMeans\nk=5']

ax.bar(range(len(centroid_names)), centroid_aurocs, color=['#2196F3', '#FF9800', '#4CAF50', '#9C27B0', '#F44336'],
       edgecolor='black', linewidth=0.5, alpha=0.8)
ax.set_xticks(range(len(centroid_names)))
ax.set_xticklabels(centroid_names, fontsize=9)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(c) Centroids vs Performance', fontsize=12, fontweight='bold')
ax.set_ylim(0.96, 1.0)
ax.axhline(y=aurocs['Global centroid'], color='red', linestyle='--', linewidth=1, alpha=0.7)
ax.grid(True, alpha=0.3, axis='y')

for i, v in enumerate(centroid_aurocs):
    ax.text(i, v + 0.001, f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig24_multi_centroid.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig24_multi_centroid.pdf', dpi=200, bbox_inches='tight')
print("Saved fig24_multi_centroid.png/pdf")
