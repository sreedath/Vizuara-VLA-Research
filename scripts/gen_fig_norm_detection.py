"""Generate Figure 82: Norm-Based OOD Detection."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/norm_detection_20260314_210133.json") as f:
    data = json.load(f)

results = data['results']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel (a): AUROC comparison across detection methods
ax = axes[0]
methods = ['l2_norm', 'l1_norm', 'linf_norm', 'l2_deviation', 'cosine', 'euclidean', 'cosine_plus_norm']
labels = ['L2\nNorm', 'L1\nNorm', 'L∞\nNorm', 'L2\nDeviation', 'Cosine\nDist', 'Euclidean\nDist', 'Cosine\n+Norm']
aurocs = [results[m]['auroc'] for m in methods]
colors = ['#FF9800' if a < 0.95 else '#4CAF50' if a >= 1.0 else '#2196F3' for a in aurocs]

bars = ax.bar(range(len(methods)), aurocs, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(a) Detection Method Comparison', fontsize=12, fontweight='bold')
ax.set_ylim(0.5, 1.05)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
ax.grid(True, alpha=0.3, axis='y')

for i, a in enumerate(aurocs):
    ax.text(i, a + 0.01, f'{a:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# Panel (b): Per-category norm distributions
ax = axes[1]
cats = ['indoor', 'noise', 'snow', 'twilight']
cat_norms = results['per_category_norms']
id_mean = results['id_mean_norm']

x = np.arange(len(cats) + 1)
means = [id_mean] + [cat_norms[c]['mean'] for c in cats]
stds = [results['l2_norm']['id_std']] + [cat_norms[c]['std'] for c in cats]
cat_labels = ['ID'] + cats
cat_colors = ['#4CAF50'] + ['#F44336']*4

ax.bar(x, means, yerr=stds, color=cat_colors, alpha=0.7, edgecolor='black', linewidth=0.5, capsize=3)
ax.set_xticks(x)
ax.set_xticklabels(cat_labels, fontsize=10)
ax.set_ylabel('L2 Norm', fontsize=11)
ax.set_title('(b) Embedding Norm by Category', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

for i, (m, s) in enumerate(zip(means, stds)):
    ax.text(i, m + s + 1, f'{m:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Panel (c): Per-category norm-based AUROC
ax = axes[2]
cat_aurocs = [cat_norms[c]['auroc'] for c in cats]
cat_colors_auc = ['#4CAF50' if a >= 1.0 else '#2196F3' if a >= 0.95 else '#FF9800' for a in cat_aurocs]

ax.bar(range(len(cats)), cat_aurocs, color=cat_colors_auc, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(cats)))
ax.set_xticklabels(cats, fontsize=10)
ax.set_ylabel('AUROC (L2 norm only)', fontsize=11)
ax.set_title('(c) Per-Category Norm Detection', fontsize=12, fontweight='bold')
ax.set_ylim(0.8, 1.05)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
ax.grid(True, alpha=0.3, axis='y')

for i, a in enumerate(cat_aurocs):
    ax.text(i, a + 0.005, f'{a:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig82_norm_detection.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig82_norm_detection.pdf', dpi=200, bbox_inches='tight')
print("Saved fig82_norm_detection.png/pdf")
