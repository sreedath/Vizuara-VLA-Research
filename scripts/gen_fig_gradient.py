"""Generate Figure 62: Gradient-Based OOD Detection."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Panel (a): AUROC comparison
ax = axes[0]
methods = ['Cosine\n(hidden)', 'Total Grad\nNorm', 'Mean Grad\nNorm', 'Max Grad\nNorm', 'Last Layer\nGrad']
aurocs = [1.000, 0.371, 0.371, 0.277, 0.113]
colors = ['#4CAF50' if a > 0.8 else '#FF9800' if a > 0.5 else '#F44336' for a in aurocs]

bars = ax.bar(range(len(methods)), aurocs, 0.6, color=colors,
              edgecolor='black', linewidth=0.5, alpha=0.85)
ax.axhline(y=0.5, color='red', linestyle=':', alpha=0.3, label='Random')
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods, fontsize=9)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(a) Gradient vs Hidden-State Detection', fontsize=12, fontweight='bold')
ax.set_ylim(0, 1.1)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')
for bar, v in zip(bars, aurocs):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
            f'{v:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.annotate('Gradients fail!\nWorse than random', xy=(2, 0.277), xytext=(3, 0.6),
            fontsize=9, color='red', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='red'))

# Panel (b): Per-scenario gradient norms
ax = axes[1]
scenarios = ['Highway', 'Urban', 'Noise', 'Indoor', 'Blackout']
grad_norms = [16002, 13367, 14210, 15266, 4839]
is_ood = [False, False, True, True, True]
colors_s = ['#2196F3' if not o else '#F44336' for o in is_ood]

bars = ax.bar(range(len(scenarios)), grad_norms, 0.6, color=colors_s,
              edgecolor='black', linewidth=0.5, alpha=0.85)
ax.set_xticks(range(len(scenarios)))
ax.set_xticklabels(scenarios, fontsize=10)
ax.set_ylabel('Total Gradient Norm', fontsize=11)
ax.set_title('(b) Gradient Norms by Scenario', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#2196F3', label='ID'),
                   Patch(facecolor='#F44336', label='OOD')]
ax.legend(handles=legend_elements, fontsize=9)

ax.annotate('ID has LARGER\ngradients than OOD!', xy=(0, 16002), xytext=(2, 17000),
            fontsize=9, color='blue', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='blue'))
ax.annotate('Blackout: tiny\ngradients', xy=(4, 4839), xytext=(3, 8000),
            fontsize=9, color='red', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='red'))

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig62_gradient.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig62_gradient.pdf', dpi=200, bbox_inches='tight')
print("Saved fig62_gradient.png/pdf")
