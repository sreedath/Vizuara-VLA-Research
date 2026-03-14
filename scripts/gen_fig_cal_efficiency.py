"""Generate Figure 42: Calibration Sample Efficiency."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Data
cal_sizes = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30]
means = [0.746, 0.758, 0.793, 0.856, 0.872, 0.929, 0.913, 0.921, 0.912, 0.933]
stds = [0.148, 0.154, 0.085, 0.051, 0.084, 0.027, 0.023, 0.025, 0.015, 0.000]
mins = [0.494, 0.483, 0.717, 0.795, 0.709, 0.880, 0.883, 0.875, 0.897, 0.933]
maxs = [0.933, 0.941, 0.911, 0.939, 0.939, 0.953, 0.938, 0.945, 0.939, 0.933]

# Panel (a): AUROC vs calibration size
ax = axes[0]
ax.plot(cal_sizes, means, 'b-o', linewidth=2, markersize=8, zorder=3, label='Mean AUROC')
ax.fill_between(cal_sizes, mins, maxs, alpha=0.2, color='#2196F3', label='Min-Max range')
ax.fill_between(cal_sizes,
                [m-s for m,s in zip(means, stds)],
                [m+s for m,s in zip(means, stds)],
                alpha=0.3, color='#2196F3', label='±1 std')

# Mark 95% threshold
max_auroc = 0.933
threshold_95 = 0.95 * max_auroc
ax.axhline(y=threshold_95, color='green', linestyle='--', alpha=0.7,
           label=f'95% of max ({threshold_95:.3f})')
ax.axvline(x=10, color='green', linestyle=':', alpha=0.5)
ax.annotate('N=10', xy=(10, threshold_95), xytext=(14, threshold_95 - 0.03),
            fontsize=10, fontweight='bold', color='green',
            arrowprops=dict(arrowstyle='->', color='green'))

ax.set_xlabel('Number of Calibration Samples', fontsize=11)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(a) Sample Efficiency Curve', fontsize=12, fontweight='bold')
ax.set_ylim(0.4, 1.0)
ax.legend(fontsize=8, loc='lower right')
ax.grid(True, alpha=0.3)

# Panel (b): Per-OOD type at different N
ax = axes[1]
ood_types = ['noise', 'indoor', 'inverted', 'blackout']
cal_levels = [1, 5, 10, 30]
per_type_data = {
    'noise': [0.519, 0.725, 0.725, 0.994],
    'indoor': [0.425, 0.550, 0.550, 0.887],
    'inverted': [0.475, 0.544, 0.562, 0.850],
    'blackout': [0.850, 1.000, 1.000, 1.000],
}

x = np.arange(len(cal_levels))
width = 0.2
colors = ['#F44336', '#FF9800', '#9C27B0', '#2196F3']

for i, (ood_type, aurocs) in enumerate(per_type_data.items()):
    bars = ax.bar(x + i*width, aurocs, width, label=ood_type,
                  color=colors[i], edgecolor='black', linewidth=0.5, alpha=0.85)

ax.set_xticks(x + 1.5*width)
ax.set_xticklabels([f'N={n}' for n in cal_levels], fontsize=11)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(b) Per-OOD Type vs Calibration Size', fontsize=12, fontweight='bold')
ax.set_ylim(0.3, 1.1)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0.5, color='red', linestyle=':', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig42_cal_efficiency.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig42_cal_efficiency.pdf', dpi=200, bbox_inches='tight')
print("Saved fig42_cal_efficiency.png/pdf")
