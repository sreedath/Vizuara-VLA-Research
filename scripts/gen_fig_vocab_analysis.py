"""Generate Figure 15: Full Vocabulary Analysis."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# (a) Action mass by scenario
scenarios = ['Urban', 'Highway', 'Night', 'Rain', 'OOD\nNoise', 'OOD\nBlank']
action_masses = [0.9939, 0.9574, 0.8976, 0.8769, 0.8436, 0.8241]
colors = ['#2ecc71', '#27ae60', '#e67e22', '#e74c3c', '#8e44ad', '#9b59b6']

bars = axes[0].bar(scenarios, action_masses, color=colors, edgecolor='black', linewidth=0.5)
axes[0].set_ylabel('Action Mass (fraction on action bins)', fontsize=10)
axes[0].set_title('(a) Action Mass by Scenario', fontsize=11, fontweight='bold')
axes[0].set_ylim(0.75, 1.02)
axes[0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
axes[0].tick_params(axis='x', rotation=0, labelsize=8)

# Add value labels
for bar, val in zip(bars, action_masses):
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.003,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

# (b) AUROC comparison
signals = ['Neg Action\nMass', 'Full\nEntropy', 'Action\nEntropy']
aurocs = [0.949, 0.786, 0.763]
bar_colors = ['#e74c3c', '#3498db', '#2ecc71']

bars2 = axes[1].bar(signals, aurocs, color=bar_colors, edgecolor='black', linewidth=0.5)
axes[1].set_ylabel('AUROC (Easy vs OOD)', fontsize=10)
axes[1].set_title('(b) AUROC: Action Mass is Best Signal', fontsize=11, fontweight='bold')
axes[1].set_ylim(0.5, 1.05)
axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8, label='Random')

for bar, val in zip(bars2, aurocs):
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Add previous best comparison
axes[1].axhline(y=0.873, color='orange', linestyle=':', alpha=0.7, linewidth=1.5, label='Raw entropy w/ MC (0.873)')
axes[1].legend(fontsize=7, loc='lower right')

# (c) Per-dimension action mass
dims = ['Dim 0\n(lat)', 'Dim 1\n(long)', 'Dim 2\n(z)', 'Dim 3\n(roll)', 'Dim 4\n(pitch)', 'Dim 5\n(yaw)', 'Dim 6\n(grip)']
masses = [0.9837, 0.8198, 0.9345, 0.9533, 0.9606, 0.9650, 0.6858]
stds = [0.0214, 0.2231, 0.1468, 0.1340, 0.0908, 0.1202, 0.4264]

colors3 = plt.cm.RdYlGn(np.array(masses))
bars3 = axes[2].bar(dims, masses, yerr=stds, capsize=3, color=colors3, edgecolor='black', linewidth=0.5)
axes[2].set_ylabel('Action Mass (fraction)', fontsize=10)
axes[2].set_title('(c) Per-Dimension Action Mass', fontsize=11, fontweight='bold')
axes[2].set_ylim(0.0, 1.15)
axes[2].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
axes[2].tick_params(axis='x', rotation=0, labelsize=7)

# Highlight low mass dims
for i, (bar, val) in enumerate(zip(bars3, masses)):
    if val < 0.85:
        axes[2].text(bar.get_x() + bar.get_width()/2., 0.05,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='red')

plt.tight_layout()
plt.savefig('/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures/fig15_vocab_analysis.png',
            dpi=300, bbox_inches='tight')
print("Saved fig15_vocab_analysis.png")
