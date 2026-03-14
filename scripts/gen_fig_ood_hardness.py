"""Generate Figure 74: OOD Hardness Spectrum."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

# Interpolation data
alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
interp_auroc = [0.417, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000]
interp_dist = [0.086, 0.199, 0.219, 0.224, 0.236, 0.248, 0.261, 0.266, 0.266, 0.268, 0.345]

# Color shift data
shifts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
color_auroc = [0.427, 0.802, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000]
color_dist = [0.086, 0.095, 0.127, 0.177, 0.203, 0.214, 0.215, 0.214, 0.232, 0.259, 0.286]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel (a): AUROC vs interpolation level
ax = axes[0]
ax.plot(alphas, interp_auroc, 'ro-', linewidth=2, markersize=8, label='Highway→Indoor')
ax.plot(shifts, color_auroc, 'bs--', linewidth=2, markersize=8, label='Color Shift')
ax.axhline(y=0.95, color='green', linestyle=':', alpha=0.3)
ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)
ax.set_xlabel('Perturbation Level', fontsize=11)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(a) Detection Boundary', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.3, 1.1)

ax.annotate('Sharp transition\nat α=0.1!', xy=(0.1, 1.0), xytext=(0.3, 0.6),
            fontsize=10, color='darkred', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5))
ax.annotate('Color: gradual\ntransition', xy=(0.1, 0.802), xytext=(0.4, 0.5),
            fontsize=9, color='darkblue', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='darkblue'))

# Panel (b): Cosine distance vs perturbation
ax = axes[1]
ax.plot(alphas, interp_dist, 'ro-', linewidth=2, markersize=8, label='Highway→Indoor')
ax.plot(shifts, color_dist, 'bs--', linewidth=2, markersize=8, label='Color Shift')

# ID reference line
id_mean = 0.088
ax.axhline(y=id_mean, color='green', linestyle='-', linewidth=2, alpha=0.5, label=f'ID mean ({id_mean:.3f})')
ax.axhline(y=0.105, color='orange', linestyle='--', alpha=0.5, label='μ+3σ threshold')

ax.set_xlabel('Perturbation Level', fontsize=11)
ax.set_ylabel('Cosine Distance', fontsize=11)
ax.set_title('(b) Embedding Space Trajectory', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel (c): Phase diagram
ax = axes[2]
# Combine both experiments into scatter
all_dists = interp_dist + color_dist
all_aurocs = interp_auroc + color_auroc
all_colors = ['red']*len(interp_dist) + ['blue']*len(color_dist)
all_labels_exp = ['Interp']*len(interp_dist) + ['Color']*len(color_dist)

for d, a, c, lbl in zip(all_dists, all_aurocs, all_colors, all_labels_exp):
    ax.scatter(d, a, c=c, s=80, alpha=0.7, edgecolors='black', linewidth=0.3)

ax.set_xlabel('Cosine Distance to Centroid', fontsize=11)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(c) Phase Diagram', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Detection boundary
ax.axvline(x=0.105, color='orange', linestyle='--', alpha=0.5, label='μ+3σ')
ax.axhline(y=0.95, color='green', linestyle=':', alpha=0.3, label='AUROC=0.95')

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='red', label='Interpolation'),
                   Patch(facecolor='blue', label='Color shift')]
ax.legend(handles=legend_elements, fontsize=9, loc='lower right')

ax.fill_between([0, 0.105], 0, 1.1, alpha=0.05, color='green')
ax.fill_between([0.105, 0.4], 0, 1.1, alpha=0.05, color='red')
ax.text(0.05, 0.5, 'ID\nzone', fontsize=10, color='green', fontweight='bold', ha='center')
ax.text(0.25, 0.5, 'OOD\nzone', fontsize=10, color='red', fontweight='bold', ha='center')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig74_ood_hardness.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig74_ood_hardness.pdf', dpi=200, bbox_inches='tight')
print("Saved fig74_ood_hardness.png/pdf")
