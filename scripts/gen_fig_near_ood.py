"""Generate Figure 53: Near-OOD Detection Challenge."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# Panel (a): Per-scenario cosine distance
ax = axes[0]
scenarios = ['Highway', 'Urban', 'Twilight', 'Wet', 'Constr.', 'Occluded', 'Snow', 'Noise', 'Blackout']
cosines = [0.082, 0.091, 0.441, 0.263, 0.437, 0.423, 0.267, 0.448, 0.482]
types = ['id', 'id', 'near', 'near', 'near', 'near', 'near', 'far', 'far']
colors = ['#2196F3' if t == 'id' else '#FF9800' if t == 'near' else '#F44336' for t in types]

bars = ax.bar(range(len(scenarios)), cosines, 0.6, color=colors,
              edgecolor='black', linewidth=0.5, alpha=0.85)
ax.set_xticks(range(len(scenarios)))
ax.set_xticklabels(scenarios, fontsize=8, rotation=30, ha='right')
ax.set_ylabel('Cosine Distance from ID Centroid', fontsize=10)
ax.set_title('(a) Cosine Distance by Scenario', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#2196F3', label='ID'),
                   Patch(facecolor='#FF9800', label='Near-OOD'),
                   Patch(facecolor='#F44336', label='Far-OOD')]
ax.legend(handles=legend_elements, fontsize=8)
ax.annotate('Clear gap between\nID and near-OOD', xy=(2, 0.3), xytext=(4, 0.15),
            fontsize=8, color='green', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='green'))

# Panel (b): Near-OOD detection by method
ax = axes[1]
near_types = ['Twilight', 'Wet', 'Constr.', 'Occluded', 'Snow']
cosine_aurocs = [1.000, 1.000, 1.000, 1.000, 1.000]
attn_aurocs = [0.925, 1.000, 0.931, 0.475, 1.000]

x = np.arange(len(near_types))
width = 0.35
bars1 = ax.bar(x - width/2, cosine_aurocs, width, label='Cosine (calibrated)',
               color='#2196F3', edgecolor='black', linewidth=0.5, alpha=0.85)
bars2 = ax.bar(x + width/2, attn_aurocs, width, label='Attn Max (cal-free)',
               color='#4CAF50', edgecolor='black', linewidth=0.5, alpha=0.85)
ax.axhline(y=0.5, color='red', linestyle=':', alpha=0.3)
ax.set_xticks(x)
ax.set_xticklabels(near_types, fontsize=9)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(b) Near-OOD Detection by Method', fontsize=12, fontweight='bold')
ax.set_ylim(0.3, 1.1)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')
ax.annotate('Attention fails\non occluded!', xy=(3, 0.475), xytext=(1.5, 0.4),
            fontsize=8, color='red', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='red'))

# Panel (c): Far vs Near OOD comparison
ax = axes[2]
categories = ['Near-OOD', 'Far-OOD', 'All OOD']
cosine_all = [1.000, 1.000, 1.000]
attn_all = [0.866, 1.000, 0.897]

x = np.arange(len(categories))
width = 0.35
bars1 = ax.bar(x - width/2, cosine_all, width, label='Cosine',
               color='#2196F3', edgecolor='black', linewidth=0.5, alpha=0.85)
bars2 = ax.bar(x + width/2, attn_all, width, label='Attn Max',
               color='#4CAF50', edgecolor='black', linewidth=0.5, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=10)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(c) Near vs Far OOD Detection', fontsize=12, fontweight='bold')
ax.set_ylim(0.7, 1.1)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars1, cosine_all):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar, val in zip(bars2, attn_all):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig53_near_ood.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig53_near_ood.pdf', dpi=200, bbox_inches='tight')
print("Saved fig53_near_ood.png/pdf")
