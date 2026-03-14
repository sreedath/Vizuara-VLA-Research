"""Generate Figure 84: Ensemble Weight Optimization."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/ensemble_optimization_20260314_210917.json") as f:
    data = json.load(f)

grid = data['grid_results']
indiv = data['individual_aurocs']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel (a): Heatmap of Cohen's d for cosine vs norm (attn=0)
ax = axes[0]
cos_vals = sorted(set(r['w_cos'] for r in grid))
norm_vals = sorted(set(r['w_norm'] for r in grid))

# Filter for attn=0 grid
d_map = {}
for r in grid:
    if abs(r['w_attn']) < 0.05:
        key = (round(r['w_cos'], 1), round(r['w_norm'], 1))
        d_map[key] = r['cohens_d']

cos_range = np.arange(0, 1.05, 0.1)
norm_range = np.arange(0, 1.05, 0.1)
heatmap = np.full((len(norm_range), len(cos_range)), np.nan)

for i, n in enumerate(norm_range):
    for j, c in enumerate(cos_range):
        key = (round(c, 1), round(n, 1))
        if key in d_map:
            heatmap[i, j] = d_map[key]

im = ax.imshow(heatmap, cmap='RdYlGn', aspect='auto', origin='lower',
               extent=[-0.05, 1.05, -0.05, 1.05])
ax.set_xlabel('Cosine Weight', fontsize=11)
ax.set_ylabel('Norm Weight', fontsize=11)
ax.set_title("(a) Cohen's d (attn=0)", fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.8, label="Cohen's d")

# Mark best
ax.plot(0.9, 0.1, '*', color='red', markersize=15, markeredgecolor='black', markeredgewidth=0.5)
ax.annotate('Best\nd=6.15', xy=(0.9, 0.1), xytext=(0.6, 0.3),
           fontsize=9, fontweight='bold', color='darkred',
           arrowprops=dict(arrowstyle='->', color='darkred'))

# Panel (b): Individual vs ensemble comparison
ax = axes[1]
methods = ['Cosine\nonly', 'Attn max\nonly', 'Norm dev\nonly', 'Best\nensemble', 'Pure\ncosine']
aurocs = [indiv['cosine'], indiv['attn_max'], indiv['norm_deviation'],
          data['best']['auroc'], indiv['cosine']]
ds = [5.79, 1.26, None, 6.15, 5.79]  # From grid results

colors = ['#2196F3', '#FF9800', '#FF9800', '#4CAF50', '#2196F3']
bars = ax.bar(range(len(methods)), aurocs, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods, fontsize=9)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(b) Individual vs Ensemble', fontsize=12, fontweight='bold')
ax.set_ylim(0.8, 1.02)
ax.grid(True, alpha=0.3, axis='y')

for i, a in enumerate(aurocs):
    ax.text(i, a + 0.003, f'{a:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Panel (c): Top 10 ensembles ranked by d
ax = axes[2]
top = data['top_10']
top_labels = [f"c={r['w_cos']:.1f}\na={r['w_attn']:.1f}\nn={r['w_norm']:.1f}" for r in top]
top_ds = [r['cohens_d'] for r in top]
colors_top = ['#4CAF50' if d > 5.5 else '#2196F3' for d in top_ds]

ax.barh(range(len(top)), top_ds, color=colors_top, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(top)))
ax.set_yticklabels(top_labels, fontsize=7)
ax.set_xlabel("Cohen's d", fontsize=11)
ax.set_title("(c) Top 10 Ensembles", fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

for i, d in enumerate(top_ds):
    ax.text(d + 0.05, i, f'{d:.2f}', va='center', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig84_ensemble_optimization.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig84_ensemble_optimization.pdf', dpi=200, bbox_inches='tight')
print("Saved fig84_ensemble_optimization.png/pdf")
