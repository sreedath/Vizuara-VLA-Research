"""Generate Figure 91: Feature Importance via Dimension Ablation."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/feature_importance_20260314_213746.json") as f:
    data = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel (a): Block ablation — importance scores
ax = axes[0]
blocks = sorted(data['block_ablation'].keys(), key=lambda k: int(k.split('_')[1]))
importances = [data['block_ablation'][b]['importance'] for b in blocks]
only_ds = [data['block_ablation'][b]['only_d'] for b in blocks]
block_ids = [int(b.split('_')[1]) for b in blocks]

colors = ['#F44336' if imp > 2 else '#FF9800' if imp > 0 else '#4CAF50' for imp in importances]
ax.bar(block_ids, importances, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_xlabel('Block Index (256 dims each)', fontsize=11)
ax.set_ylabel('Importance (d drop when removed)', fontsize=11)
ax.set_title('(a) Block Importance Scores', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Panel (b): Feature selection — d vs n_dims
ax = axes[1]
# Variance selection
var_data = data['variance_selection']
var_dims = sorted(var_data.keys(), key=lambda k: var_data[k]['n_dims'])
var_ns = [var_data[k]['n_dims'] for k in var_dims]
var_ds = [var_data[k]['d'] for k in var_dims]

# Mean-diff selection
md_data = data['mean_diff_selection']
md_dims = sorted(md_data.keys(), key=lambda k: md_data[k]['n_dims'])
md_ns = [md_data[k]['n_dims'] for k in md_dims]
md_ds = [md_data[k]['d'] for k in md_dims]

# Random selection
rand_data = data['random_subsets']
rand_dims = sorted(rand_data.keys(), key=lambda k: rand_data[k]['n_dims'])
rand_ns = [rand_data[k]['n_dims'] for k in rand_dims]
rand_ds = [rand_data[k]['d_mean'] for k in rand_dims]
rand_stds = [rand_data[k]['d_std'] for k in rand_dims]

ax.semilogx(md_ns, md_ds, 'o-', color='#F44336', label='Mean-diff selection', linewidth=2, markersize=6)
ax.semilogx(var_ns, var_ds, 's-', color='#2196F3', label='Variance selection', linewidth=2, markersize=6)
ax.errorbar(rand_ns, rand_ds, yerr=rand_stds, fmt='D-', color='#9E9E9E',
            label='Random (5 seeds)', linewidth=1.5, markersize=5, capsize=3)
ax.axhline(y=data['baseline']['d'], color='green', linestyle='--', linewidth=1.5,
           label=f'Full {data["dim"]} dims (d={data["baseline"]["d"]:.1f})')
ax.set_xlabel('Number of Dimensions', fontsize=11)
ax.set_ylabel("Cohen's d", fontsize=11)
ax.set_title('(b) Feature Selection: d vs Dimensions', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel (c): Single-block d values
ax = axes[2]
sorted_blocks = sorted(blocks, key=lambda b: data['block_ablation'][b]['only_d'], reverse=True)
block_labels = [f'{data["block_ablation"][b]["start"]}-{data["block_ablation"][b]["end"]}'
                for b in sorted_blocks]
block_ds = [data['block_ablation'][b]['only_d'] for b in sorted_blocks]

colors_bar = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(sorted_blocks)))
ax.barh(range(len(sorted_blocks)), block_ds, color=colors_bar, alpha=0.8,
        edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(sorted_blocks)))
ax.set_yticklabels(block_labels, fontsize=7)
ax.set_xlabel("Cohen's d (block only)", fontsize=11)
ax.set_title('(c) Single-Block Detection Power', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig91_feature_importance.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig91_feature_importance.pdf', dpi=200, bbox_inches='tight')
print("Saved fig91_feature_importance.png/pdf")
