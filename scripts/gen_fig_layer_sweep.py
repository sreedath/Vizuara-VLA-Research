"""Generate Figure 58: Hidden Layer Sweep for OOD Detection."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# Data
layers = [0, 4, 8, 12, 16, 20, 24, 28, 31, 32]
aurocs = [0.500, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000]
cohens_d = [0.00, 3.26, 2.47, 2.78, 3.81, 5.61, 10.54, 9.22, 6.26, 10.45]
id_means = [0.000, 0.0003, 0.0008, 0.0083, 0.0253, 0.0340, 0.0347, 0.0306, 0.0286, 0.0873]
ood_means = [0.000, 0.0031, 0.0055, 0.0454, 0.1541, 0.2018, 0.1874, 0.1597, 0.1538, 0.4295]

# Panel (a): AUROC by layer
ax = axes[0]
colors = ['#F44336' if a < 0.9 else '#4CAF50' for a in aurocs]
bars = ax.bar(range(len(layers)), aurocs, 0.6, color=colors,
              edgecolor='black', linewidth=0.5, alpha=0.85)
ax.set_xticks(range(len(layers)))
ax.set_xticklabels([f'L{l}' for l in layers], fontsize=9)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(a) AUROC by Hidden Layer', fontsize=12, fontweight='bold')
ax.set_ylim(0.3, 1.1)
ax.axhline(y=0.5, color='red', linestyle=':', alpha=0.3, label='Random')
ax.grid(True, alpha=0.3, axis='y')
ax.legend(fontsize=8)
ax.annotate('Embedding layer\nfails completely', xy=(0, 0.5), xytext=(2, 0.65),
            fontsize=8, color='red', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='red'))
ax.annotate('All layers 4-32\nachieve perfect', xy=(5, 1.0), xytext=(3, 0.85),
            fontsize=8, color='green', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='green'))

# Panel (b): Cohen's d by layer (shows which layers are BEST)
ax = axes[1]
colors_d = plt.cm.YlOrRd(np.array(cohens_d) / max(cohens_d))
bars = ax.bar(range(len(layers)), cohens_d, 0.6, color=colors_d,
              edgecolor='black', linewidth=0.5, alpha=0.85)
ax.set_xticks(range(len(layers)))
ax.set_xticklabels([f'L{l}' for l in layers], fontsize=9)
ax.set_ylabel("Cohen's d (effect size)", fontsize=11)
ax.set_title("(b) Separability by Layer", fontsize=12, fontweight='bold')
ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.3, label='Large effect')
ax.grid(True, alpha=0.3, axis='y')
ax.legend(fontsize=8)
for bar, v in zip(bars, cohens_d):
    if v > 0:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                f'{v:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.annotate('Layer 24: peak\nseparability!', xy=(6, 10.54), xytext=(3.5, 8),
            fontsize=9, color='darkred', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5))

# Panel (c): ID vs OOD cosine distance by layer
ax = axes[2]
ax.plot(range(len(layers)), id_means, 'bo-', linewidth=2, markersize=8, label='ID mean')
ax.plot(range(len(layers)), ood_means, 'rs-', linewidth=2, markersize=8, label='OOD mean')
ax.fill_between(range(len(layers)), id_means, ood_means, alpha=0.2, color='orange')
ax.set_xticks(range(len(layers)))
ax.set_xticklabels([f'L{l}' for l in layers], fontsize=9)
ax.set_ylabel('Mean Cosine Distance', fontsize=11)
ax.set_title('(c) ID vs OOD Distance by Layer', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.annotate('Gap grows\nthrough layers', xy=(7, 0.10), xytext=(4, 0.30),
            fontsize=9, color='orange', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='orange'))

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig58_layer_sweep.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig58_layer_sweep.pdf', dpi=200, bbox_inches='tight')
print("Saved fig58_layer_sweep.png/pdf")
