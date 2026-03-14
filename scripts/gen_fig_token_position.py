"""Generate Figure 95: Token Position Analysis."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/token_position_analysis_20260314_215009.json") as f:
    data = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel (a): Pooling strategy comparison
ax = axes[0]
pool_data = data['pooling_results']
methods = ['last', 'mean', 'last_quarter_mean', 'second_last', 'max', 'middle', 'first']
labels = ['Last\nToken', 'Mean\nPool', 'Last-Qtr\nMean', '2nd-Last\nToken', 'Max\nPool', 'Middle\nToken', 'First\nToken']
aurocs = [pool_data[m]['auroc'] for m in methods]
ds = [pool_data[m]['d'] for m in methods]
colors = ['#4CAF50' if a >= 1.0 else '#FF9800' if a >= 0.95 else '#F44336' for a in aurocs]

bars = ax.bar(range(len(methods)), ds, color=colors, alpha=0.7,
              edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Cohen's d", fontsize=11)
ax.set_title("(a) Pooling Strategy: Cohen's d", fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

for i, (d_val, auroc) in enumerate(zip(ds, aurocs)):
    ax.text(i, d_val + 1, f'{auroc:.3f}', ha='center', fontsize=7, fontweight='bold')

# Panel (b): Position-by-position AUROC
ax = axes[1]
pos_data = data['position_results']
positions = sorted([int(k) for k in pos_data.keys()])
pos_aurocs = [pos_data[str(p)]['auroc'] for p in positions]
pos_ds = [pos_data[str(p)]['d'] for p in positions]

ax.plot(positions, pos_aurocs, 'o-', color='#2196F3', linewidth=1.5, markersize=4)
ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.3)
ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.3, label='Random')
ax.set_xlabel('Token Position', fontsize=11)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(b) AUROC by Token Position', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

# Panel (c): d by position
ax = axes[2]
colors_pos = ['#F44336' if d < 5 else '#FF9800' if d < 20 else '#4CAF50' for d in pos_ds]
ax.bar(range(len(positions)), pos_ds, color=colors_pos, alpha=0.7,
       edgecolor='black', linewidth=0.3)
ax.set_xticks(range(0, len(positions), 3))
ax.set_xticklabels([str(positions[i]) for i in range(0, len(positions), 3)], fontsize=7)
ax.set_xlabel('Token Position', fontsize=11)
ax.set_ylabel("Cohen's d", fontsize=11)
ax.set_title("(c) Separation by Token Position", fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig95_token_position.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig95_token_position.pdf', dpi=200, bbox_inches='tight')
print("Saved fig95_token_position.png/pdf")
