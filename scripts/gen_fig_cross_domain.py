"""Generate Figure 48: Cross-Domain Transfer Analysis."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# Panel (a): Cross-calibration AUROC matrix
ax = axes[0]
cal_names = ['Highway\nOnly', 'Urban\nOnly', 'Mixed']
test_names = ['Highway\nTest', 'Urban\nTest', 'Overall']
data = np.array([
    [0.975, 0.530, 0.752],
    [0.682, 0.988, 0.835],
    [0.905, 0.938, 0.921],
])

im = ax.imshow(data, cmap='RdYlGn', vmin=0.4, vmax=1.0, aspect='auto')
ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(test_names, fontsize=9)
ax.set_yticklabels(cal_names, fontsize=9)
ax.set_title('(a) Cross-Domain Transfer AUROC', fontsize=12, fontweight='bold')

for i in range(3):
    for j in range(3):
        color = 'white' if data[i, j] < 0.6 else 'black'
        ax.text(j, i, f'{data[i, j]:.3f}', ha='center', va='center',
                fontsize=11, fontweight='bold', color=color)

plt.colorbar(im, ax=ax, shrink=0.8)

# Highlight failure cells
for i, j in [(0, 1), (1, 0)]:
    rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False,
                          edgecolor='red', linewidth=3)
    ax.add_patch(rect)

# Panel (b): Per-OOD type with different calibrations
ax = axes[1]
ood_types = ['Noise', 'Indoor', 'Inverted', 'Blackout']
hw_cal = [1.000, 0.970, 0.930, 1.000]  # Highway test
ur_cal = [1.000, 0.980, 0.970, 1.000]  # Urban test
mx_hw = [1.000, 0.940, 0.680, 1.000]   # Mixed → Highway test
mx_ur = [1.000, 0.860, 0.890, 1.000]   # Mixed → Urban test

x = np.arange(len(ood_types))
width = 0.2
bars1 = ax.bar(x - 1.5*width, hw_cal, width, label='HW cal → HW test',
               color='#2196F3', edgecolor='black', linewidth=0.5, alpha=0.85)
bars2 = ax.bar(x - 0.5*width, ur_cal, width, label='UR cal → UR test',
               color='#4CAF50', edgecolor='black', linewidth=0.5, alpha=0.85)
bars3 = ax.bar(x + 0.5*width, mx_hw, width, label='Mixed → HW test',
               color='#FF9800', edgecolor='black', linewidth=0.5, alpha=0.85)
bars4 = ax.bar(x + 1.5*width, mx_ur, width, label='Mixed → UR test',
               color='#9C27B0', edgecolor='black', linewidth=0.5, alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(ood_types, fontsize=10)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(b) Same-Domain vs Cross-Domain', fontsize=12, fontweight='bold')
ax.set_ylim(0.5, 1.1)
ax.legend(fontsize=7, loc='lower left')
ax.grid(True, alpha=0.3, axis='y')

# Panel (c): Domain centroid distances
ax = axes[2]
labels = ['HW↔UR', 'HW↔Mix', 'UR↔Mix', 'Noise↔Mix', 'Indoor↔Mix', 'Inverted↔Mix', 'Blackout↔Mix']
distances = [0.694, 0.245, 0.145, 0.654, 0.570, 0.477, 0.842]
colors = ['#9E9E9E', '#9E9E9E', '#9E9E9E', '#F44336', '#FF9800', '#9C27B0', '#333333']

bars = ax.barh(range(len(labels)), distances, 0.6, color=colors,
               edgecolor='black', linewidth=0.5, alpha=0.85)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel('Cosine Distance Between Centroids', fontsize=11)
ax.set_title('(c) Domain Centroid Distances', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# Annotate
ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.3, label='Detection boundary')
ax.legend(fontsize=8)

for bar, v in zip(bars, distances):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2.,
            f'{v:.3f}', ha='left', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig48_cross_domain.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig48_cross_domain.pdf', dpi=200, bbox_inches='tight')
print("Saved fig48_cross_domain.png/pdf")
