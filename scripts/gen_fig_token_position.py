"""Generate Figure 63: Token Position Analysis."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

positions = [5, 25, 50, 75, 90, 95, 100]
aurocs = [1.000, 0.719, 0.949, 0.922, 0.859, 1.000, 1.000]
cohens_d = [10.94, 1.01, 2.27, 1.13, 0.57, 6.70, 8.52]

# Panel (a): AUROC and Cohen's d by position
ax = axes[0]
colors = ['#4CAF50' if a >= 0.95 else '#FF9800' if a >= 0.7 else '#F44336' for a in aurocs]
bars = ax.bar(range(len(positions)), aurocs, 0.6, color=colors,
              edgecolor='black', linewidth=0.5, alpha=0.85)
ax.set_xticks(range(len(positions)))
ax.set_xticklabels([f'{p}%' for p in positions], fontsize=10)
ax.set_xlabel('Token Position (% of sequence)', fontsize=11)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(a) AUROC by Token Position', fontsize=12, fontweight='bold')
ax.set_ylim(0.6, 1.05)
ax.axhline(y=0.5, color='red', linestyle=':', alpha=0.3)
ax.grid(True, alpha=0.3, axis='y')
for bar, v in zip(bars, aurocs):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.annotate('Visual tokens\n(best signal!)', xy=(0, 1.0), xytext=(1, 0.75),
            fontsize=8, color='green', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='green'))
ax.annotate('Text tokens\n(weakest)', xy=(4, 0.859), xytext=(2.5, 0.68),
            fontsize=8, color='orange', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='orange'))

# Panel (b): Effect size by position
ax = axes[1]
colors_d = plt.cm.YlOrRd(np.array(cohens_d) / max(cohens_d))
bars = ax.bar(range(len(positions)), cohens_d, 0.6, color=colors_d,
              edgecolor='black', linewidth=0.5, alpha=0.85)
ax.set_xticks(range(len(positions)))
ax.set_xticklabels([f'{p}%' for p in positions], fontsize=10)
ax.set_xlabel('Token Position (% of sequence)', fontsize=11)
ax.set_ylabel("Cohen's d", fontsize=11)
ax.set_title("(b) Separability by Position", fontsize=12, fontweight='bold')
ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.3, label='Large effect')
ax.grid(True, alpha=0.3, axis='y')
ax.legend(fontsize=8)
for bar, v in zip(bars, cohens_d):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
            f'{v:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.annotate('Visual tokens:\nd=10.94!', xy=(0, 10.94), xytext=(2, 9),
            fontsize=9, color='darkred', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='darkred'))

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig63_token_position.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig63_token_position.pdf', dpi=200, bbox_inches='tight')
print("Saved fig63_token_position.png/pdf")
