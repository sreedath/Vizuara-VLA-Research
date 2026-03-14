"""Generate Figure 43: Deep Action Token Distribution Analysis."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# Data from action_dist_deep experiment (Experiment 57)
scenarios = ['highway', 'urban', 'noise', 'indoor', 'inverted', 'blackout']
is_ood = [False, False, True, True, True, True]
colors = ['#2196F3', '#2196F3', '#F44336', '#FF9800', '#9C27B0', '#333333']
labels_pretty = ['Highway', 'Urban', 'Noise', 'Indoor', 'Inverted', 'Blackout']

# Stats from JSON
mass_means = [0.968, 0.997, 0.877, 0.979, 0.915, 0.980]
entropy_means = [1.098, 0.998, 1.233, 0.720, 1.373, 3.099]
top1_means = [0.656, 0.706, 0.508, 0.770, 0.503, 0.273]
garbage_means = [0.031, 0.003, 0.123, 0.020, 0.085, 0.014]

# Panel (a): Action Mass vs Garbage Tokens
ax = axes[0]
for i, (m, g) in enumerate(zip(mass_means, garbage_means)):
    marker = 's' if is_ood[i] else 'o'
    edge = 'black' if is_ood[i] else 'navy'
    ax.scatter(m, g, c=colors[i], s=150, marker=marker, edgecolors=edge,
               linewidths=1.5, zorder=3, label=labels_pretty[i])
ax.set_xlabel('Action Mass (fraction on 256 bins)', fontsize=11)
ax.set_ylabel('Garbage Token Mass (top-5 non-action)', fontsize=11)
ax.set_title('(a) Action Mass vs Garbage Leakage', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)
ax.annotate('Noise: highest\ngarbage leakage', xy=(0.877, 0.123),
            xytext=(0.90, 0.10), fontsize=8, color='#F44336',
            arrowprops=dict(arrowstyle='->', color='#F44336'))

# Panel (b): Entropy comparison
ax = axes[1]
x = np.arange(len(scenarios))
bars = ax.bar(x, entropy_means, 0.6, color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(labels_pretty, fontsize=9, rotation=20, ha='right')
ax.set_ylabel('Action Distribution Entropy (nats)', fontsize=11)
ax.set_title('(b) Per-Scenario Action Entropy', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
# Annotate blackout
ax.annotate('Blackout: 3× higher\n(uniform-like)', xy=(5, 3.099),
            xytext=(3.5, 2.8), fontsize=8, color='#333333',
            arrowprops=dict(arrowstyle='->', color='#333333'))
# ID mean line
id_ent_mean = np.mean([entropy_means[0], entropy_means[1]])
ax.axhline(y=id_ent_mean, color='blue', linestyle='--', alpha=0.5, label=f'ID mean ({id_ent_mean:.2f})')
ax.legend(fontsize=8)

# Panel (c): Top-1 concentration
ax = axes[2]
bars = ax.bar(x, top1_means, 0.6, color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(labels_pretty, fontsize=9, rotation=20, ha='right')
ax.set_ylabel('Top-1 Action Probability', fontsize=11)
ax.set_title('(c) Action Concentration (Top-1)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
# ID mean line
id_top1_mean = np.mean([top1_means[0], top1_means[1]])
ax.axhline(y=id_top1_mean, color='blue', linestyle='--', alpha=0.5, label=f'ID mean ({id_top1_mean:.2f})')
ax.legend(fontsize=8)
ax.annotate('OOD: lower\nconcentration', xy=(4, 0.503),
            xytext=(2.5, 0.35), fontsize=8, color='#9C27B0',
            arrowprops=dict(arrowstyle='->', color='#9C27B0'))

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig43_action_dist_deep.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig43_action_dist_deep.pdf', dpi=200, bbox_inches='tight')
print("Saved fig43_action_dist_deep.png/pdf")
