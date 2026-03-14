"""Generate Figure 111: Calibration Strategies Comparison."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/calibration_strategies_20260314_232850.json") as f:
    data = json.load(f)

strategies = data['strategies']
names = list(strategies.keys())
short_names = ['Centroid', 'Nearest\nNeighbor', 'Farthest\nNeighbor', 'Average\nto All', 'Per-Class\nCentroid', '3-NN\nDistance']
d_values = [strategies[n]['d'] for n in names]
auroc_values = [strategies[n]['auroc'] for n in names]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: D-prime comparison
ax = axes[0]
colors = ['#2196F3', '#4CAF50', '#FF5722', '#FF9800', '#9C27B0', '#00BCD4']
bars = ax.bar(range(len(names)), d_values, color=colors, alpha=0.8)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(short_names, fontsize=8)
ax.set_ylabel("D-prime")
ax.set_title("(A) D-prime by Calibration Strategy")
ax.grid(True, alpha=0.3, axis='y')

# Annotate values
for bar, d in zip(bars, d_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{d:.1f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

# Highlight best
best_idx = np.argmax(d_values)
bars[best_idx].set_edgecolor('gold')
bars[best_idx].set_linewidth(3)

# Panel B: Strategy taxonomy
ax = axes[1]
# Scatter: x = computational cost (relative), y = d-prime
costs = [1, 20, 20, 20, 2, 20]  # relative to centroid
ax.scatter(costs, d_values, c=colors, s=200, zorder=5, edgecolors='black')
for i, name in enumerate(short_names):
    offset_x = 0.5 if costs[i] < 10 else -1
    ax.annotate(name.replace('\n', ' '), (costs[i], d_values[i]),
                textcoords="offset points", xytext=(10, 5), fontsize=8)

ax.set_xlabel("Relative Computational Cost")
ax.set_ylabel("D-prime")
ax.set_title("(B) Cost-Effectiveness")
ax.set_xscale('log')
ax.grid(True, alpha=0.3)
ax.axhline(y=3, color='red', linestyle='--', alpha=0.3, label='Detection threshold')
ax.legend(fontsize=8)

# All AUROC = 1.0 annotation
ax.text(0.5, 0.02, "All strategies: AUROC = 1.000",
        transform=ax.transAxes, fontsize=11, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

plt.suptitle("Calibration Strategy Comparison (Exp 125)\nPer-class centroid optimal (d=58.2), all achieve AUROC=1.000",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/latex/figures/fig111_calibration_strategies.png",
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig111")
