"""Generate Figure 37: Systematic Ablation Study."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Panel (a): Cumulative pipeline ablation
ax = axes[0]
stages = ['Global cos\n(1 frame)', '+ Per-scene\ncentroids', '+ Temporal\naggregation', '+ Action mass\ncombination']
aurocs = [0.618, 0.820, 0.945, 0.965]
deltas = [0, 0.203, 0.125, 0.020]
colors = ['#FF9800', '#4CAF50', '#2196F3', '#1565C0']

bars = ax.bar(range(len(stages)), aurocs, color=colors, edgecolor='black',
              linewidth=0.5, alpha=0.85)
ax.set_xticks(range(len(stages)))
ax.set_xticklabels(stages, fontsize=9)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(a) Cumulative Pipeline Ablation', fontsize=12, fontweight='bold')
ax.set_ylim(0.4, 1.1)
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0.5, color='red', linestyle=':', alpha=0.3)

for i, (bar, val, d) in enumerate(zip(bars, aurocs, deltas)):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.3f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
    if d > 0:
        ax.annotate(f'+{d:.3f}', xy=(i, val - 0.05),
                    fontsize=9, fontweight='bold', color='green', ha='center')

# Arrow showing total gain
ax.annotate('', xy=(3, 0.965), xytext=(0, 0.618),
            arrowprops=dict(arrowstyle='->', color='red', lw=2, ls='--'))
ax.text(1.5, 0.52, 'Total: +0.347', fontsize=10, fontweight='bold',
        color='red', ha='center')

# Panel (b): Leave-one-out ablation
ax = axes[1]
components = ['Full\npipeline', 'w/o\nper-scene', 'w/o\ntemporal', 'w/o\nmass', 'w/o cosine\n(mass only)']
loo_aurocs = [0.965, 0.847, 0.847, 0.945, 0.690]
drops = [0, 0.118, 0.118, 0.020, 0.275]
loo_colors = ['#1565C0', '#F44336', '#F44336', '#FF9800', '#F44336']

bars = ax.bar(range(len(components)), loo_aurocs, color=loo_colors, edgecolor='black',
              linewidth=0.5, alpha=0.85)
ax.set_xticks(range(len(components)))
ax.set_xticklabels(components, fontsize=9)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(b) Leave-One-Out Ablation', fontsize=12, fontweight='bold')
ax.set_ylim(0.5, 1.1)
ax.grid(True, alpha=0.3, axis='y')

for i, (bar, val, d) in enumerate(zip(bars, loo_aurocs, drops)):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.3f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
    if d > 0:
        ax.text(bar.get_x() + bar.get_width()/2, val - 0.04,
                f'-{d:.3f}', ha='center', fontsize=9, fontweight='bold', color='red')

# Panel (c): Per-OOD type performance heatmap
ax = axes[2]
methods = ['Global cos\n(1f)', 'Per-scene\ncos (1f)', 'Action\nmass (1f)',
           'Per-scene\ncos (5s)', 'Full\npipeline']
ood_types = ['offroad', 'flooded', 'tunnel', 'snow']
data_grid = np.array([
    [0.850, 0.300, 0.690, 0.630],
    [0.910, 0.600, 0.940, 0.830],
    [0.680, 0.750, 0.700, 0.630],
    [1.000, 0.780, 1.000, 1.000],
    [1.000, 0.860, 1.000, 1.000],
])

im = ax.imshow(data_grid, cmap='RdYlGn', aspect='auto', vmin=0.2, vmax=1.0)
ax.set_xticks(range(len(ood_types)))
ax.set_xticklabels(ood_types, fontsize=10)
ax.set_yticks(range(len(methods)))
ax.set_yticklabels(methods, fontsize=8)
ax.set_title('(c) Per-OOD Type AUROC', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, label='AUROC', shrink=0.8)

for i in range(len(methods)):
    for j in range(len(ood_types)):
        val = data_grid[i, j]
        color = 'white' if val < 0.5 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=9,
                fontweight='bold', color=color)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig37_ablation.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig37_ablation.pdf', dpi=200, bbox_inches='tight')
print("Saved fig37_ablation.png/pdf")
