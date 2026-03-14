"""Generate Figure 20: Cosine Distance Robustness Analysis."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# (a) Layer-wise AUROC
layers = [0, 8, 16, 24, 31]
aurocs = [0.891, 0.889, 0.990, 0.988, 0.988]

axes[0].plot(layers, aurocs, 'o-', color='#3498db', linewidth=2.5, markersize=10, zorder=5)
axes[0].fill_between(layers, aurocs, alpha=0.15, color='#3498db')

# Mark best
best_idx = np.argmax(aurocs)
axes[0].scatter([layers[best_idx]], [aurocs[best_idx]], s=200, color='#f39c12',
               edgecolor='black', linewidth=1.5, zorder=10, marker='*')
axes[0].annotate(f'Best: {aurocs[best_idx]:.3f}', xy=(layers[best_idx], aurocs[best_idx]),
                xytext=(20, 0.970), fontsize=9, fontweight='bold', color='#f39c12',
                arrowprops=dict(arrowstyle='->', color='#f39c12', lw=1.5))

# Reference lines
axes[0].axhline(y=0.988, color='gray', linestyle=':', alpha=0.4)
axes[0].text(1, 0.985, 'Last layer', fontsize=7, color='gray')

axes[0].set_xlabel('Layer Index (of 32)', fontsize=10)
axes[0].set_ylabel('Cosine Distance AUROC', fontsize=10)
axes[0].set_title('(a) Layer-wise OOD Detection', fontsize=11, fontweight='bold')
axes[0].set_ylim(0.85, 1.01)
axes[0].set_xticks(layers)
axes[0].grid(True, alpha=0.3)

# Add annotation for early vs late layers
axes[0].annotate('Early layers:\nweak signal', xy=(4, 0.89), xytext=(6, 0.865),
                fontsize=7, color='#e74c3c',
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1))

# (b) Calibration set size sensitivity with error bars
n_cals = [5, 10, 15, 20, 25, 30]
cos_means = [0.968, 0.979, 0.978, 0.978, 0.980, 0.981]
cos_stds = [0.006, 0.006, 0.007, 0.010, 0.009, 0.006]
mass_means = [0.736, 0.738, 0.738, 0.738, 0.744, 0.744]
mass_stds = [0.010, 0.008, 0.008, 0.008, 0.025, 0.025]

axes[1].errorbar(n_cals, cos_means, yerr=[1.96*s for s in cos_stds],
                fmt='o-', color='#2ecc71', linewidth=2.5, markersize=8,
                capsize=4, capthick=1.5, label='Cosine Distance', zorder=5)
axes[1].errorbar(n_cals, mass_means, yerr=[1.96*s for s in mass_stds],
                fmt='s--', color='#e74c3c', linewidth=2, markersize=7,
                capsize=4, capthick=1.5, label='Action Mass', zorder=4)

# Shade the gap
axes[1].fill_between(n_cals, cos_means, mass_means, alpha=0.1, color='#2ecc71')
axes[1].annotate('', xy=(17, 0.97), xytext=(17, 0.74),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
axes[1].text(18, 0.855, 'Δ ≈ +0.24', fontsize=9, fontweight='bold', rotation=90, va='center')

axes[1].set_xlabel('Calibration Set Size', fontsize=10)
axes[1].set_ylabel('Overall AUROC', fontsize=10)
axes[1].set_title('(b) Calibration Size Sensitivity', fontsize=11, fontweight='bold')
axes[1].set_ylim(0.65, 1.05)
axes[1].set_xticks(n_cals)
axes[1].legend(fontsize=9, loc='center right')
axes[1].grid(True, alpha=0.3)

# Highlight that 5 samples is already great
axes[1].annotate('5 samples\nsuffice!', xy=(5, 0.968), xytext=(9, 0.94),
                fontsize=8, fontweight='bold', color='#2ecc71',
                arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=1.5))

# (c) Per-OOD-type AUROC at n_cal=5 vs n_cal=30
ood_types = ['Noise', 'Blank', 'Indoor', 'Inverted', 'Checker', 'Blackout']
auroc_5 = [0.984, 0.980, 0.948, 0.898, 0.999, 1.000]
auroc_30 = [0.991, 0.993, 0.967, 0.934, 1.000, 1.000]

x = np.arange(len(ood_types))
width = 0.35

b1 = axes[2].bar(x - width/2, auroc_5, width, label='n_cal=5',
                color='#f39c12', edgecolor='black', linewidth=0.5)
b2 = axes[2].bar(x + width/2, auroc_30, width, label='n_cal=30',
                color='#3498db', edgecolor='black', linewidth=0.5)

axes[2].set_ylabel('AUROC', fontsize=10)
axes[2].set_title('(c) Per-Type: 5 vs 30 Calibration Samples', fontsize=11, fontweight='bold')
axes[2].set_xticks(x)
axes[2].set_xticklabels(ood_types, fontsize=8)
axes[2].legend(fontsize=9)
axes[2].set_ylim(0.85, 1.05)
axes[2].grid(True, alpha=0.2, axis='y')

# Add value labels
for bar, val in zip(b1, auroc_5):
    axes[2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.003,
                f'{val:.3f}', ha='center', va='bottom', fontsize=6, fontweight='bold')
for bar, val in zip(b2, auroc_30):
    axes[2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.003,
                f'{val:.3f}', ha='center', va='bottom', fontsize=6, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures/fig20_cosine_robustness.png',
            dpi=300, bbox_inches='tight')
print("Saved fig20_cosine_robustness.png")
