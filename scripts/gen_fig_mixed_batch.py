"""Generate Figure 72: Mixed Batch Detection."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

rates = [0.01, 0.05, 0.10, 0.25, 0.50]
aurocs = [1.000, 1.000, 1.000, 1.000, 1.000]
aps = [1.000, 1.000, 1.000, 1.000, 1.000]
precisions = [0.633, 0.633, 0.730, 0.927, 0.977]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Panel (a): AUROC and AP vs contamination rate
ax = axes[0]
ax.plot(rates, aurocs, 'go-', linewidth=2, markersize=12, label='AUROC')
ax.plot(rates, aps, 'bs--', linewidth=2, markersize=10, label='Average Precision')
ax.set_xlabel('OOD Contamination Rate', fontsize=11)
ax.set_ylabel('Score', fontsize=11)
ax.set_title('(a) Detection Quality vs Contamination', fontsize=12, fontweight='bold')
ax.set_ylim(0.9, 1.05)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.annotate('Perfect detection at\nALL contamination rates', xy=(0.25, 1.0), xytext=(0.15, 0.94),
            fontsize=11, color='darkgreen', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='darkgreen'))

# Panel (b): Precision at threshold
ax = axes[1]
colors = plt.cm.RdYlGn(np.array(precisions))
bars = ax.bar(range(len(rates)), precisions, 0.6, color=colors,
              edgecolor='black', linewidth=0.5, alpha=0.85)
ax.set_xticks(range(len(rates)))
ax.set_xticklabels([f'{r:.0%}' for r in rates], fontsize=11)
ax.set_xlabel('OOD Contamination Rate', fontsize=11)
ax.set_ylabel('Precision at μ+3σ Threshold', fontsize=11)
ax.set_title('(b) Precision vs Base Rate', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0.5, 1.05)

for bar, v in zip(bars, precisions):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{v:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.annotate('Low base rate\n→ more FPs\n(base rate fallacy)', xy=(0, 0.633), xytext=(1.5, 0.58),
            fontsize=9, color='darkred', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='darkred'))

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig72_mixed_batch.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig72_mixed_batch.pdf', dpi=200, bbox_inches='tight')
print("Saved fig72_mixed_batch.png/pdf")
