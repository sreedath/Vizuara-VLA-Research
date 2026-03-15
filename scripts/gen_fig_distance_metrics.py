"""Generate Figure 118: Distance Metric Comparison."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/distance_metrics_20260315_001023.json") as f:
    data = json.load(f)

results = data['results']
# Sort by d-prime descending
sorted_metrics = sorted(results.items(), key=lambda x: x[1]['d'], reverse=True)
names = [m[0] for m in sorted_metrics]
ds = [m[1]['d'] for m in sorted_metrics]
aurocs = [m[1]['auroc'] for m in sorted_metrics]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel A: D-prime by metric (bar chart)
ax = axes[0]
colors = ['#4CAF50' if d > 50 else '#2196F3' if d > 30 else '#FF9800' if d > 15 else '#F44336' for d in ds]
bars = ax.barh(range(len(names)), ds, color=colors, alpha=0.8)
ax.set_yticks(range(len(names)))
ax.set_yticklabels([n.replace('_', ' ').title() for n in names], fontsize=9)
ax.set_xlabel("D-prime (σ)")
ax.set_title("(A) Separation Strength by Metric")
ax.grid(True, alpha=0.3, axis='x')
for bar, d in zip(bars, ds):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f"{d:.1f}", va='center', fontsize=9)
ax.invert_yaxis()

# Panel B: Metric families comparison
ax = axes[1]
families = {
    'Direction-based': ['cosine', 'correlation', 'angular'],
    'Magnitude-based': ['euclidean', 'manhattan', 'chebyshev', 'bray_curtis'],
}
family_colors = {'Direction-based': '#4CAF50', 'Magnitude-based': '#2196F3'}
x_pos = 0
positions = []
tick_labels = []
bar_colors = []
bar_ds = []
for family_name, members in families.items():
    for m in members:
        positions.append(x_pos)
        tick_labels.append(m.replace('_', ' ').title())
        bar_colors.append(family_colors[family_name])
        bar_ds.append(results[m]['d'])
        x_pos += 1
    x_pos += 0.5  # gap between families

bars = ax.bar(positions, bar_ds, color=bar_colors, alpha=0.8)
ax.set_xticks(positions)
ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
ax.set_ylabel("D-prime (σ)")
ax.set_title("(B) Direction vs Magnitude Metrics")
ax.grid(True, alpha=0.3, axis='y')

# Add family labels
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#4CAF50', alpha=0.8, label='Direction-based'),
                   Patch(facecolor='#2196F3', alpha=0.8, label='Magnitude-based')]
ax.legend(handles=legend_elements, fontsize=8)

# Add d values
for bar, d in zip(bars, bar_ds):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{d:.1f}", ha='center', fontsize=8)

# Panel C: D-prime ratio relative to cosine
ax = axes[2]
cosine_d = results['cosine']['d']
ratios = [(n, results[n]['d'] / cosine_d * 100) for n in names if n != 'cosine']
ratio_names = [r[0].replace('_', ' ').title() for r in ratios]
ratio_vals = [r[1] for r in ratios]

bars = ax.barh(range(len(ratio_names)), ratio_vals, color='#9C27B0', alpha=0.7)
ax.axvline(x=100, color='green', linestyle='--', alpha=0.5, label='Cosine (100%)')
ax.set_yticks(range(len(ratio_names)))
ax.set_yticklabels(ratio_names, fontsize=9)
ax.set_xlabel("% of Cosine D-prime")
ax.set_title("(C) Relative to Cosine Baseline")
ax.grid(True, alpha=0.3, axis='x')
ax.legend(fontsize=8)
for bar, v in zip(bars, ratio_vals):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
            f"{v:.0f}%", va='center', fontsize=9)
ax.invert_yaxis()

plt.suptitle("Distance Metric Comparison (Exp 132)\nAll 7 metrics AUROC=1.000; Cosine/Correlation lead with d≈52",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/latex/figures/fig118_distance_metrics.png",
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig118")
