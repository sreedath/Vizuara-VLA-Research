"""Generate Figure 64: Embedding Dimensionality Analysis."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

dims = [2, 4, 8, 16, 4096]
aurocs = [1.000, 1.000, 1.000, 1.000, 1.000]
cohens_d = [8.84, 19.80, 14.95, 15.01, 9.74]

# Panel (a): AUROC vs dimensionality
ax = axes[0]
ax.plot(range(len(dims)), aurocs, 'go-', linewidth=2, markersize=12)
ax.set_xticks(range(len(dims)))
ax.set_xticklabels([str(d) for d in dims], fontsize=11)
ax.set_xlabel('Embedding Dimensions', fontsize=11)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(a) AUROC vs Dimensionality', fontsize=12, fontweight='bold')
ax.set_ylim(0.9, 1.02)
ax.grid(True, alpha=0.3)
ax.annotate('Perfect at ALL\ndimensions!', xy=(0, 1.0), xytext=(1, 0.94),
            fontsize=10, color='green', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='green'))
ax.annotate('Even 2D PCA\nsuffices', xy=(0, 1.0), xytext=(2, 0.96),
            fontsize=9, color='darkgreen', fontweight='bold')

# Panel (b): Cohen's d vs dimensionality
ax = axes[1]
colors = plt.cm.YlOrRd(np.array(cohens_d) / max(cohens_d))
bars = ax.bar(range(len(dims)), cohens_d, 0.6, color=colors,
              edgecolor='black', linewidth=0.5, alpha=0.85)
ax.set_xticks(range(len(dims)))
ax.set_xticklabels([str(d) for d in dims], fontsize=11)
ax.set_xlabel('Embedding Dimensions', fontsize=11)
ax.set_ylabel("Cohen's d", fontsize=11)
ax.set_title("(b) Separability vs Dimensionality", fontsize=12, fontweight='bold')
ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.3, label='Large effect')
ax.grid(True, alpha=0.3, axis='y')
ax.legend(fontsize=8)
for bar, v in zip(bars, cohens_d):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.4,
            f'{v:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.annotate('PCA-4: peak\nd=19.80!', xy=(1, 19.8), xytext=(3, 17),
            fontsize=10, color='darkred', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5))

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig64_embedding_dim.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig64_embedding_dim.pdf', dpi=200, bbox_inches='tight')
print("Saved fig64_embedding_dim.png/pdf")
