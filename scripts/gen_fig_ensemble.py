"""Generate Figure 55: Ensemble OOD Detection."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# Panel (a): Ensemble strategies on all OOD (ranked)
ax = axes[0]
strategies = ['Max(cos+attn)', 'Avg(cos+attn)', 'Adaptive', 'W(0.6cos)', 'Avg(top 3)',
              'Product', 'Vote', 'Avg(all 5)', 'Cosine only', 'Attn only']
aurocs_all = [1.000, 1.000, 1.000, 1.000, 1.000, 0.985, 0.984, 0.966, 1.000, 0.890]
colors = ['#4CAF50']*5 + ['#FF9800']*2 + ['#2196F3'] + ['#FF5722', '#9E9E9E']

bars = ax.barh(range(len(strategies)), aurocs_all, 0.6, color=colors,
               edgecolor='black', linewidth=0.5, alpha=0.85)
ax.set_yticks(range(len(strategies)))
ax.set_yticklabels(strategies, fontsize=9)
ax.set_xlabel('AUROC (All OOD)', fontsize=11)
ax.set_title('(a) Ensemble Strategies\n(All OOD)', fontsize=11, fontweight='bold')
ax.set_xlim(0.85, 1.02)
ax.grid(True, alpha=0.3, axis='x')
for bar, v in zip(bars, aurocs_all):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2.,
            f'{v:.3f}', ha='left', va='center', fontsize=9, fontweight='bold')

# Panel (b): Near-OOD comparison (where ensembles help most)
ax = axes[1]
strategies_near = ['Max(cos+attn)', 'Avg(cos+attn)', 'Adaptive', 'W(0.6cos)',
                   'W(0.4cos)', 'Avg(top 3)', 'Product', 'Vote',
                   'Avg(all 5)', 'Cosine only', 'Attn only']
aurocs_near = [1.000, 1.000, 1.000, 1.000, 0.996, 0.974, 0.976, 0.931, 0.916, 1.000, 0.824]
colors_n = ['#4CAF50']*4 + ['#8BC34A'] + ['#FF9800']*2 + ['#F44336'] + ['#2196F3'] + ['#FF5722', '#9E9E9E']

bars = ax.barh(range(len(strategies_near)), aurocs_near, 0.6, color=colors_n,
               edgecolor='black', linewidth=0.5, alpha=0.85)
ax.set_yticks(range(len(strategies_near)))
ax.set_yticklabels(strategies_near, fontsize=8)
ax.set_xlabel('AUROC (Near-OOD)', fontsize=11)
ax.set_title('(b) Ensemble Strategies\n(Near-OOD)', fontsize=11, fontweight='bold')
ax.set_xlim(0.78, 1.02)
ax.grid(True, alpha=0.3, axis='x')
for bar, v in zip(bars, aurocs_near):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2.,
            f'{v:.3f}', ha='left', va='center', fontsize=8, fontweight='bold')
ax.annotate('Attention alone\nfails near-OOD', xy=(0.824, 10), xytext=(0.85, 8),
            fontsize=8, color='red', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='red'))

# Panel (c): Weight sweep
ax = axes[2]
cos_weights = np.arange(0, 1.05, 0.1)
all_aurocs = [0.890, 0.955, 0.993, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000]
near_aurocs = [0.824, 0.875, 0.924, 0.975, 0.996, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000]

ax.plot(cos_weights, all_aurocs, 'bo-', linewidth=2, markersize=8, label='All OOD')
ax.plot(cos_weights, near_aurocs, 'rs-', linewidth=2, markersize=8, label='Near-OOD')
ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.3)
ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5, label='Equal weight')
ax.fill_between(cos_weights, 0.8, 1.0, where=[a >= 1.0 and n >= 1.0 for a, n in zip(all_aurocs, near_aurocs)],
                alpha=0.1, color='green')
ax.set_xlabel('Cosine Weight (1 - attn weight)', fontsize=11)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(c) Cosine-Attention Weight Sweep', fontsize=11, fontweight='bold')
ax.set_ylim(0.8, 1.02)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.annotate('Sweet spot:\ncos_w ≥ 0.5', xy=(0.5, 1.0), xytext=(0.2, 0.88),
            fontsize=9, color='green', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='green'))

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig55_ensemble.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig55_ensemble.pdf', dpi=200, bbox_inches='tight')
print("Saved fig55_ensemble.png/pdf")
