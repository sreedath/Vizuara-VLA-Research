"""Generate Figure 65: Leave-One-Out OOD Generalization."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

# Data from experiment 79
categories = ['noise', 'indoor', 'twilight', 'snow', 'blackout', 'inverted', 'ALL']
aurocs = [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000]
n_samples = [8, 8, 8, 8, 6, 8, 46]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Panel (a): Per-category AUROC
ax = axes[0]
colors = ['#2196F3', '#FF9800', '#9C27B0', '#00BCD4', '#F44336', '#4CAF50', '#333333']
bars = ax.bar(range(len(categories)), aurocs, 0.6, color=colors,
              edgecolor='black', linewidth=0.5, alpha=0.85)
ax.set_xticks(range(len(categories)))
ax.set_xticklabels(categories, fontsize=10, rotation=30, ha='right')
ax.set_xlabel('Held-Out OOD Category', fontsize=11)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(a) Leave-One-Out: Per-Category AUROC', fontsize=12, fontweight='bold')
ax.set_ylim(0.9, 1.05)
ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.3)
ax.grid(True, alpha=0.3, axis='y')

for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
            '1.000', ha='center', va='bottom', fontsize=9, fontweight='bold', color='green')

ax.annotate('Perfect generalization\nto ALL unseen categories', xy=(3, 1.0), xytext=(3, 0.94),
            fontsize=10, color='darkgreen', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='darkgreen'))

# Panel (b): Sample counts and generalization summary
ax = axes[1]
# Create a visual summary showing the leave-one-out design
cats_short = ['noise', 'indoor', 'twilight', 'snow', 'blackout', 'inverted']
n_cats = len(cats_short)

# Create matrix: rows = held-out category, cols = categories
# Green = train, Red = test (held out)
matrix = np.ones((n_cats, n_cats))
for i in range(n_cats):
    matrix[i, i] = 0  # held out

cmap = plt.cm.colors.ListedColormap(['#FF5252', '#4CAF50'])
im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)

ax.set_xticks(range(n_cats))
ax.set_xticklabels(cats_short, fontsize=9, rotation=45, ha='right')
ax.set_yticks(range(n_cats))
ax.set_yticklabels([f'Hold out\n{c}' for c in cats_short], fontsize=8)
ax.set_xlabel('OOD Categories', fontsize=11)
ax.set_title('(b) Leave-One-Out Design\n(Green=Train, Red=Test)', fontsize=12, fontweight='bold')

# Add AUROC text in each held-out cell
for i in range(n_cats):
    ax.text(i, i, '1.000', ha='center', va='center', fontsize=9,
            fontweight='bold', color='white')

# Add legend patches
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#4CAF50', label='Available for cal.'),
                   Patch(facecolor='#FF5252', label='Held out (test)')]
ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig65_leave_one_out.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig65_leave_one_out.pdf', dpi=200, bbox_inches='tight')
print("Saved fig65_leave_one_out.png/pdf")
