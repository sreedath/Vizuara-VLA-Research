"""Generate Figure 16: Selective Prediction with Action Mass Comparison."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# (a) AUROC comparison grouped by detection type
signals = ['Action Mass\n(T=1.0)', 'Action Mass\n(T=0.25)', 'Entropy\n(T=1.0)',
           'MC Entropy\n(p=0.20)', 'Aug Ensemble\nMass', 'Mass+MCEnt\n(combined)']
auroc_ood = [0.877, 0.869, 0.742, 0.824, 0.856, 0.873]
auroc_hard = [0.742, 0.777, 0.777, 0.620, 0.859, 0.799]

x = np.arange(len(signals))
width = 0.35

bars1 = axes[0].bar(x - width/2, auroc_ood, width, label='Easy vs OOD', color='#e74c3c', edgecolor='black', linewidth=0.5)
bars2 = axes[0].bar(x + width/2, auroc_hard, width, label='Easy vs Hard', color='#3498db', edgecolor='black', linewidth=0.5)

axes[0].set_ylabel('AUROC', fontsize=10)
axes[0].set_title('(a) AUROC by Signal and Detection Type', fontsize=11, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(signals, fontsize=7)
axes[0].legend(fontsize=8)
axes[0].set_ylim(0.5, 1.0)
axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

# (b) OOD rejection curves
coverages = [30, 40, 50, 60, 70, 80, 90, 100]
rej_action_mass = [92.5, 82.5, 72.5, 62.5, 55.0, 40.0, 20.0, 0.0]
rej_mc_entropy = [85.0, 82.5, 82.5, 65.0, 55.0, 37.5, 17.5, 0.0]
rej_aug_ensemble = [87.5, 77.5, 67.5, 57.5, 45.0, 32.5, 10.0, 0.0]
rej_combined = [92.5, 85.0, 72.5, 62.5, 57.5, 37.5, 20.0, 0.0]
rej_entropy = [72.5, 57.5, 50.0, 40.0, 22.5, 5.0, 0.0, 0.0]

axes[1].plot(coverages, rej_action_mass, 'o-', color='#e74c3c', linewidth=2, markersize=5, label='Action Mass (1 pass)')
axes[1].plot(coverages, rej_mc_entropy, 's-', color='#3498db', linewidth=2, markersize=5, label='MC Entropy (10 passes)')
axes[1].plot(coverages, rej_aug_ensemble, '^-', color='#2ecc71', linewidth=2, markersize=5, label='Aug Ensemble (5 passes)')
axes[1].plot(coverages, rej_combined, 'D-', color='#9b59b6', linewidth=2, markersize=5, label='Combined (11 passes)')
axes[1].plot(coverages, rej_entropy, 'v-', color='#95a5a6', linewidth=1.5, markersize=4, label='Entropy (1 pass)')

axes[1].set_xlabel('Coverage (%)', fontsize=10)
axes[1].set_ylabel('OOD Rejection Rate (%)', fontsize=10)
axes[1].set_title('(b) OOD Rejection at Coverage Levels', fontsize=11, fontweight='bold')
axes[1].legend(fontsize=7, loc='upper right')
axes[1].set_xlim(25, 105)
axes[1].set_ylim(-5, 100)
axes[1].grid(True, alpha=0.3)

# Highlight key operating point
axes[1].annotate('92.5% OOD rejection\nat 30% coverage',
                xy=(30, 92.5), xytext=(45, 85),
                fontsize=7, fontweight='bold', color='#e74c3c',
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5))

# (c) Cost-effectiveness
signal_names = ['Action Mass\n(T=1.0)', 'Action Mass\n(T=0.25)', 'Entropy\n(T=1.0)',
                'Aug Ensemble', 'MC Entropy', 'Combined']
passes = [1, 1, 1, 5, 10, 11]
aurocs = [0.877, 0.869, 0.742, 0.856, 0.824, 0.873]
auroc_per_pass = [a/p for a, p in zip(aurocs, passes)]

colors = ['#e74c3c', '#e74c3c', '#95a5a6', '#2ecc71', '#3498db', '#9b59b6']
bars = axes[2].bar(signal_names, auroc_per_pass, color=colors, edgecolor='black', linewidth=0.5)

axes[2].set_ylabel('AUROC / Forward Pass', fontsize=10)
axes[2].set_title('(c) Cost-Effectiveness', fontsize=11, fontweight='bold')
axes[2].tick_params(axis='x', rotation=30, labelsize=7)

for bar, val, n_pass in zip(bars, auroc_per_pass, passes):
    axes[2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.3f}\n({n_pass}p)', ha='center', va='bottom', fontsize=7, fontweight='bold')

axes[2].set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig('/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures/fig16_selective_action_mass.png',
            dpi=300, bbox_inches='tight')
print("Saved fig16_selective_action_mass.png")
