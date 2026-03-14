"""Generate Figure 67: Adversarial Perturbation Robustness."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

# Data from experiment 81
perturbations = [
    ('Gauss-10', 1.000, 0.131),
    ('Gauss-25', 1.000, 0.221),
    ('Gauss-50', 1.000, 0.307),
    ('Gauss-100', 0.667, 0.395),
    ('S&P-1%', 0.356, 0.433),
    ('S&P-5%', 0.208, 0.456),
    ('S&P-10%', 0.537, 0.422),
    ('S&P-20%', 0.597, 0.417),
    ('JPEG-50', 1.000, 0.092),
    ('JPEG-20', 1.000, 0.092),
    ('JPEG-5', 1.000, 0.101),
    ('Blur-1', 1.000, 0.126),
    ('Blur-3', 0.991, 0.310),
    ('Blur-5', 0.949, 0.335),
]

names = [p[0] for p in perturbations]
aurocs = [p[1] for p in perturbations]
dists = [p[2] for p in perturbations]

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Panel (a): AUROC by perturbation type
ax = axes[0]
colors = []
for a in aurocs:
    if a >= 0.95:
        colors.append('#4CAF50')
    elif a >= 0.8:
        colors.append('#FF9800')
    else:
        colors.append('#F44336')

bars = ax.barh(range(len(names)), aurocs, 0.6, color=colors,
               edgecolor='black', linewidth=0.5, alpha=0.85)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=9)
ax.set_xlabel('AUROC', fontsize=11)
ax.set_title('(a) Detection AUROC Under Perturbation', fontsize=12, fontweight='bold')
ax.axvline(x=0.95, color='green', linestyle='--', alpha=0.3, label='Robust (0.95)')
ax.axvline(x=0.80, color='orange', linestyle='--', alpha=0.3, label='Weak (0.80)')
ax.set_xlim(0, 1.1)
ax.legend(fontsize=8, loc='lower right')
ax.grid(True, alpha=0.3, axis='x')

for bar, v in zip(bars, aurocs):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2.,
            f'{v:.3f}', ha='left', va='center', fontsize=8, fontweight='bold')

# Add category labels
ax.annotate('JPEG: Fully Robust', xy=(1.0, 9.5), fontsize=9,
            color='darkgreen', fontweight='bold')
ax.annotate('S&P: Breaks\nDetection!', xy=(0.3, 5), fontsize=9,
            color='darkred', fontweight='bold')

# Panel (b): Perturbed ID distance vs OOD distance
ax = axes[1]
clean_id = 0.087
ood_mean = 0.408

ax.barh(range(len(names)), dists, 0.6, color=colors,
        edgecolor='black', linewidth=0.5, alpha=0.7, label='Perturbed ID dist')
ax.axvline(x=clean_id, color='blue', linestyle='-', linewidth=2, label=f'Clean ID ({clean_id:.3f})')
ax.axvline(x=ood_mean, color='red', linestyle='-', linewidth=2, label=f'OOD mean ({ood_mean:.3f})')
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=9)
ax.set_xlabel('Cosine Distance to Centroid', fontsize=11)
ax.set_title('(b) Perturbation Drift in Embedding Space', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='lower right')
ax.grid(True, alpha=0.3, axis='x')

# Shade danger zone
ax.axvspan(ood_mean - 0.05, ood_mean + 0.05, alpha=0.1, color='red')
ax.annotate('S&P crosses\ninto OOD zone', xy=(0.43, 5), xytext=(0.5, 2),
            fontsize=9, color='darkred', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='darkred'))

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig67_adversarial_robustness.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig67_adversarial_robustness.pdf', dpi=200, bbox_inches='tight')
print("Saved fig67_adversarial_robustness.png/pdf")
