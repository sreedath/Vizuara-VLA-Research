"""Generate Figure 26: PCA Dimensionality Reduction Analysis."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/pca_reduction_20260314_161339.json"
OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

with open(RESULTS) as f:
    data = json.load(f)

pca_results = {int(k): v for k, v in data['pca_results'].items()}
rp_results = {int(k): v for k, v in data['rp_results'].items()}

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Panel (a): AUROC vs dimension for PCA and random projection
ax = axes[0]

# PCA
pca_dims = sorted(pca_results.keys())
pca_aurocs = [pca_results[d] for d in pca_dims]
ax.plot(pca_dims, pca_aurocs, 'b-o', linewidth=2, markersize=8, label='PCA', zorder=3)

# Random projection
rp_dims = sorted(rp_results.keys())
rp_aurocs = [rp_results[d] for d in rp_dims]
ax.plot(rp_dims, rp_aurocs, 'r-s', linewidth=2, markersize=8, label='Random Proj.', zorder=3)

ax.axhline(y=pca_results[4096], color='gray', linestyle='--', linewidth=1.5,
           label=f'Full 4096-d ({pca_results[4096]:.3f})')
ax.set_xscale('log', base=2)
ax.set_xlabel('Dimensions', fontsize=11)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(a) AUROC vs Dimensionality', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='lower right')
ax.set_ylim(0.55, 1.02)
ax.grid(True, alpha=0.3)
ax.set_xticks([2, 8, 32, 128, 512, 2048, 4096])
ax.set_xticklabels(['2', '8', '32', '128', '512', '2K', '4K'], fontsize=9)

# Panel (b): First k vs Last k PCA components
ax = axes[1]
ks = [4, 8, 16]
first_k = [0.932, 0.979, 0.974]
last_k = [0.444, 0.456, 0.833]

x = np.arange(len(ks))
width = 0.35
ax.bar(x - width/2, first_k, width, label='First k', color='#2196F3',
       edgecolor='black', linewidth=0.5, alpha=0.8)
ax.bar(x + width/2, last_k, width, label='Last k', color='#FF5722',
       edgecolor='black', linewidth=0.5, alpha=0.8)

ax.set_xticks(x)
ax.set_xticklabels([f'k={k}' for k in ks], fontsize=10)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(b) First vs Last PCA Components', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.set_ylim(0, 1.1)
ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
ax.grid(True, alpha=0.3, axis='y')

for i, (f, l) in enumerate(zip(first_k, last_k)):
    ax.text(i - width/2, f + 0.02, f'{f:.3f}', ha='center', fontsize=9, fontweight='bold')
    ax.text(i + width/2, l + 0.02, f'{l:.3f}', ha='center', fontsize=9, fontweight='bold')

# Panel (c): Top individual PCA components
ax = axes[2]
top_comps = data['top_components'][:10]
comp_ids = [f'PC{c[0]}' for c in top_comps]
comp_aurocs = [c[1] for c in top_comps]

colors_bar = ['#2196F3' if a >= 0.8 else '#64B5F6' if a >= 0.75 else '#90CAF9' for a in comp_aurocs]
bars = ax.barh(range(len(comp_ids)), comp_aurocs, color=colors_bar,
               edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(comp_ids)))
ax.set_yticklabels(comp_ids, fontsize=9)
ax.set_xlabel('AUROC', fontsize=11)
ax.set_title('(c) Individual Component AUROC', fontsize=12, fontweight='bold')
ax.set_xlim(0.6, 0.9)
ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

for bar, val in zip(bars, comp_aurocs):
    ax.text(val - 0.01, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', ha='right', va='center', fontsize=8, fontweight='bold', color='white')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig26_pca_reduction.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig26_pca_reduction.pdf', dpi=200, bbox_inches='tight')
print("Saved fig26_pca_reduction.png/pdf")
