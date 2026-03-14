"""Generate Figure 44: Mahalanobis vs Cosine Distance OOD Detection."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# Panel (a): Method comparison bar chart
ax = axes[0]
methods = ['Cosine\nDistance', 'Mahalanobis\n(PCA-20)', 'Feature\nNorm Diff', 'Cosine +\nNorm Diff']
aurocs = [0.933, 0.097, 0.589, 0.905]
colors = ['#2196F3', '#F44336', '#FF9800', '#4CAF50']

bars = ax.bar(range(len(methods)), aurocs, 0.6, color=colors,
              edgecolor='black', linewidth=0.5, alpha=0.85)
ax.axhline(y=0.5, color='red', linestyle=':', alpha=0.3, label='Random')
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods, fontsize=9)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(a) Detection Method Comparison', fontsize=12, fontweight='bold')
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, aurocs):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.annotate('Worse than\nrandom!', xy=(1, 0.097), xytext=(1.5, 0.35),
            fontsize=9, color='#F44336', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#F44336'))

# Panel (b): Per-scenario comparison
ax = axes[1]
ood_types = ['Noise', 'Indoor', 'Inverted', 'Blackout']
cos_aurocs = [0.994, 0.887, 0.850, 1.000]
mah_aurocs = [0.162, 0.112, 0.112, 0.000]

x = np.arange(len(ood_types))
width = 0.35
bars1 = ax.bar(x - width/2, cos_aurocs, width, label='Cosine',
               color='#2196F3', edgecolor='black', linewidth=0.5, alpha=0.85)
bars2 = ax.bar(x + width/2, mah_aurocs, width, label='Mahalanobis',
               color='#F44336', edgecolor='black', linewidth=0.5, alpha=0.85)
ax.axhline(y=0.5, color='red', linestyle=':', alpha=0.3)
ax.set_xticks(x)
ax.set_xticklabels(ood_types, fontsize=10)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(b) Per-OOD Type: Cosine vs Mahalanobis', fontsize=12, fontweight='bold')
ax.set_ylim(0, 1.15)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Panel (c): PCA dimension sweep
ax = axes[2]
pca_dims = [4, 8, 12, 16, 20, 25, 29]
pca_aurocs = [0.263, 0.144, 0.094, 0.102, 0.097, 0.089, 0.114]

ax.plot(pca_dims, pca_aurocs, 'r-o', linewidth=2, markersize=8, label='Mahalanobis')
ax.axhline(y=0.933, color='#2196F3', linestyle='--', linewidth=2, alpha=0.7,
           label='Cosine (0.933)')
ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3, label='Random')
ax.fill_between(pca_dims, pca_aurocs, 0.5, alpha=0.1, color='red',
                label='Below random')
ax.set_xlabel('PCA Dimensions', fontsize=11)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(c) Mahalanobis PCA Sweep', fontsize=12, fontweight='bold')
ax.set_ylim(0, 1.05)
ax.legend(fontsize=8, loc='center right')
ax.grid(True, alpha=0.3)
ax.annotate('All dims < random', xy=(16, 0.102), xytext=(20, 0.45),
            fontsize=9, color='#F44336',
            arrowprops=dict(arrowstyle='->', color='#F44336'))

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig44_mahalanobis.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig44_mahalanobis.pdf', dpi=200, bbox_inches='tight')
print("Saved fig44_mahalanobis.png/pdf")
