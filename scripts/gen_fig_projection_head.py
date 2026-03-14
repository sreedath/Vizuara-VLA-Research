"""Generate Figure 96: Projection Head Analysis."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/projection_head_20260314_215415.json") as f:
    data = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
results = data['results']

# Panel (a): PCA projection d by dims
ax = axes[0]
pca_keys = sorted([k for k in results if k.startswith('pca_')], key=lambda k: results[k]['dims'])
pca_dims = [results[k]['dims'] for k in pca_keys]
pca_ds = [results[k]['d'] for k in pca_keys]
pca_aurocs = [results[k]['auroc'] for k in pca_keys]

ax.semilogx(pca_dims, pca_ds, 'o-', color='#2196F3', linewidth=2, markersize=8, label='PCA')

# Add whitened
wh_keys = sorted([k for k in results if k.startswith('whitened_')], key=lambda k: results[k]['dims'])
wh_dims = [results[k]['dims'] for k in wh_keys]
wh_ds = [results[k]['d'] for k in wh_keys]
ax.semilogx(wh_dims, wh_ds, 's-', color='#FF9800', linewidth=2, markersize=8, label='Whitened PCA')

ax.axhline(y=results['baseline']['d'], color='green', linestyle='--', linewidth=1.5,
           label=f'Full dim (d={results["baseline"]["d"]:.1f})')
ax.set_xlabel('Projection Dimensions', fontsize=11)
ax.set_ylabel("Cohen's d", fontsize=11)
ax.set_title("(a) PCA Projection: Cohen's d", fontsize=12, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel (b): Method comparison
ax = axes[1]
methods = ['baseline', 'pca_4', 'pca_8', 'pca_32', 'whitened_8', 'lda']
labels = [f'Full\n({results["baseline"]["dims"]}D)',
          f'PCA-4\n(4D)', f'PCA-8\n(8D)', f'PCA-32\n(32D)',
          f'Wh-PCA-8\n(8D)', f'LDA\n(1D)']
ds = [results[m]['d'] for m in methods]
aurocs = [results[m]['auroc'] for m in methods]
colors = ['#4CAF50' if a >= 1.0 else '#FF9800' if a >= 0.95 else '#F44336' for a in aurocs]

bars = ax.bar(range(len(methods)), ds, color=colors, alpha=0.7,
              edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Cohen's d", fontsize=11)
ax.set_title('(b) Method Comparison', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

for i, (d_val, a) in enumerate(zip(ds, aurocs)):
    ax.text(i, d_val + 1, f'{a:.3f}', ha='center', fontsize=7, fontweight='bold')

# Panel (c): Random vs PCA
ax = axes[2]
rand_keys = sorted([k for k in results if k.startswith('random_')], key=lambda k: results[k]['dims'])
rand_dims = [results[k]['dims'] for k in rand_keys]
rand_ds = [results[k]['d_mean'] for k in rand_keys]
rand_stds = [results[k]['d_std'] for k in rand_keys]

ax.errorbar(rand_dims, rand_ds, yerr=rand_stds, fmt='D-', color='#9E9E9E',
            linewidth=1.5, markersize=5, capsize=3, label='Random')
ax.plot(pca_dims, pca_ds, 'o-', color='#2196F3', linewidth=2, markersize=8, label='PCA')
ax.axhline(y=results['baseline']['d'], color='green', linestyle='--', linewidth=1.5,
           label='Full dim')

ax.set_xlabel('Dimensions', fontsize=11)
ax.set_ylabel("Cohen's d", fontsize=11)
ax.set_title('(c) PCA vs Random Projection', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig96_projection_head.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig96_projection_head.pdf', dpi=200, bbox_inches='tight')
print("Saved fig96_projection_head.png/pdf")
