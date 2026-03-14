"""Generate Figure 102: Mahalanobis Distance Detection — Refined."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/mahalanobis_refined_20260314_223236.json") as f:
    data = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

res = data['results']

# Panel (a): Method comparison — AUROC and d
ax = axes[0]
methods = ['cosine', 'diagonal_mahalanobis', 'euclidean', 'ledoit_wolf', 'oas',
           'pca_16_mahalanobis', 'pca_8_mahalanobis', 'pca_4_mahalanobis']
labels = ['Cosine', 'Diagonal\nMaha.', 'Euclidean', 'Ledoit-Wolf', 'OAS',
          'PCA-16\nMaha.', 'PCA-8\nMaha.', 'PCA-4\nMaha.']
aurocs = [res[m]['auroc'] for m in methods]
ds = [res[m]['d'] for m in methods]

colors = ['#4CAF50' if a >= 1.0 else '#FF9800' if a >= 0.95 else '#F44336' for a in aurocs]
x = np.arange(len(methods))
bars = ax.bar(x, ds, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
for i, (bar, auroc) in enumerate(zip(bars, aurocs)):
    ax.text(bar.get_x() + bar.get_width()/2,
            max(bar.get_height(), 0) + 0.5,
            f'AUROC\n{auroc:.3f}', ha='center', va='bottom', fontsize=6)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7, rotation=45, ha='right')
ax.set_ylabel("Cohen's d", fontsize=11)
ax.set_title('(a) Distance Metric Comparison', fontsize=12, fontweight='bold')
ax.axhline(y=0, color='black', linewidth=0.5)
ax.grid(True, alpha=0.3, axis='y')

# Panel (b): Per-category for top 3 methods
ax = axes[1]
cats = ['highway', 'urban', 'snow', 'indoor', 'twilight', 'noise']
groups = [data['per_category'][c]['group'] for c in cats]

cos_scores = [data['per_category'][c]['cosine_mean'] for c in cats]
euc_scores_norm = [data['per_category'][c]['euclidean_mean'] / 100 for c in cats]  # normalize
diag_scores_norm = [data['per_category'][c]['diagonal_maha_mean'] / 60000 for c in cats]  # normalize

x2 = np.arange(len(cats))
width = 0.25
ax.bar(x2 - width, cos_scores, width, label='Cosine', color='#4CAF50', alpha=0.8)
ax.bar(x2, euc_scores_norm, width, label='Euclidean (÷100)', color='#2196F3', alpha=0.8)
ax.bar(x2 + width, diag_scores_norm, width, label='Diag. Maha. (÷60K)', color='#FF9800', alpha=0.8)
ax.set_xticks(x2)
ax.set_xticklabels(cats, fontsize=9)
ax.set_ylabel('Normalized Score', fontsize=11)
ax.set_title('(b) Per-Category Scores (Normalized)', fontsize=12, fontweight='bold')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3, axis='y')

# Panel (c): Full-dim methods ranked by d
ax = axes[2]
# Only full-dim methods
full_methods = ['cosine', 'diagonal_mahalanobis', 'euclidean', 'ledoit_wolf']
full_labels = ['Cosine Distance', 'Diagonal Mahalanobis', 'Euclidean Distance', 'Ledoit-Wolf Maha.']
full_ds = [res[m]['d'] for m in full_methods]
full_aurocs = [res[m]['auroc'] for m in full_methods]

sorted_idx = np.argsort(full_ds)[::-1]
colors_sorted = ['#4CAF50' if full_aurocs[i] >= 1.0 else '#FF9800' for i in sorted_idx]

bars = ax.barh(range(len(full_methods)),
               [full_ds[i] for i in sorted_idx],
               color=colors_sorted, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(full_methods)))
ax.set_yticklabels([full_labels[i] for i in sorted_idx], fontsize=9)
for i, (bar, idx) in enumerate(zip(bars, sorted_idx)):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f'AUROC={full_aurocs[idx]:.3f}', va='center', fontsize=9)
ax.set_xlabel("Cohen's d", fontsize=11)
ax.set_title('(c) Full-Dim Methods Ranked', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig102_mahalanobis_refined.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig102_mahalanobis_refined.pdf', dpi=200, bbox_inches='tight')
print("Saved fig102_mahalanobis_refined.png/pdf")
