"""Generate Figure 78: Multi-Layer Hidden State Fusion."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/multi_layer_fusion_20260314_204644.json") as f:
    data = json.load(f)

strategies = data['strategies']
pca = data['pca_results']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel (a): Cohen's d for concatenation strategies
ax = axes[0]
names = ['last_layer', 'layer_24', 'early_late', 'every_4th', 'last_4', 'last_8', 'weighted_avg']
labels = ['Last\nLayer', 'L24', 'Early+\nLate', 'Every\n4th', 'Last\n4', 'Last\n8', 'Weighted\nAvg']
ds = [strategies[n]['cohens_d'] for n in names]
dims = [strategies[n].get('total_dim', 4096) for n in names]

colors = ['#2196F3' if d < 5.5 else '#4CAF50' if d < 5.8 else '#FF9800' for d in ds]
bars = ax.bar(range(len(names)), ds, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Cohen's d", fontsize=11)
ax.set_title("(a) Concatenation Strategies", fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

for i, (d, dim) in enumerate(zip(ds, dims)):
    ax.text(i, d + 0.1, f'd={d:.2f}\n{dim//1024}K', ha='center', va='bottom', fontsize=7, fontweight='bold')

# Panel (b): PCA reduction of multi-layer features — the star result
ax = axes[1]
pca_names = ['pca_4', 'pca_8', 'pca_16', 'pca_32']
pca_dims = [4, 8, 16, 32]
pca_ds = [pca[n]['cohens_d'] for n in pca_names]
pca_vars = [pca[n]['explained_var'] for n in pca_names]

color_pca = ['#FF9800', '#F44336', '#F44336', '#F44336']
bars = ax.bar(range(len(pca_names)), pca_ds, color=color_pca, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(pca_names)))
ax.set_xticklabels([f'PCA-{d}' for d in pca_dims], fontsize=10)
ax.set_ylabel("Cohen's d", fontsize=11)
ax.set_title("(b) PCA of Multi-Layer Features", fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add baseline reference
ax.axhline(y=5.73, color='#2196F3', linestyle='--', alpha=0.7, linewidth=1.5, label='Last layer baseline (d=5.73)')
ax.legend(fontsize=8, loc='lower right')

for i, (d, v) in enumerate(zip(pca_ds, pca_vars)):
    ax.text(i, d + 0.3, f'd={d:.1f}\n{v:.1%} var', ha='center', va='bottom', fontsize=8, fontweight='bold')

# Panel (c): Dimension vs d tradeoff
ax = axes[2]
# All strategies on one plot
all_dims_raw = [strategies[n].get('total_dim', 4096) for n in names]
all_ds_raw = [strategies[n]['cohens_d'] for n in names]

# PCA points
pca_dim_vals = pca_dims
pca_d_vals = pca_ds

ax.scatter(all_dims_raw, all_ds_raw, c='#2196F3', s=80, zorder=3, label='Concatenation', edgecolors='black', linewidth=0.5)
ax.scatter(pca_dim_vals, pca_d_vals, c='#F44336', s=100, marker='*', zorder=4, label='PCA reduction', edgecolors='black', linewidth=0.5)

# Label key points
for i, n in enumerate(names):
    if n in ['last_layer', 'last_4', 'every_4th']:
        ax.annotate(labels[i].replace('\n', ' '), (all_dims_raw[i], all_ds_raw[i]),
                    textcoords='offset points', xytext=(10, 5), fontsize=7)

for i, d in enumerate(pca_dims):
    ax.annotate(f'PCA-{d}', (d, pca_d_vals[i]),
                textcoords='offset points', xytext=(10, -10), fontsize=7, color='darkred')

ax.set_xlabel('Feature Dimensions', fontsize=11)
ax.set_ylabel("Cohen's d", fontsize=11)
ax.set_title("(c) Dimension-Efficiency Tradeoff", fontsize=12, fontweight='bold')
ax.set_xscale('log')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig78_multi_layer_fusion.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig78_multi_layer_fusion.pdf', dpi=200, bbox_inches='tight')
print("Saved fig78_multi_layer_fusion.png/pdf")
