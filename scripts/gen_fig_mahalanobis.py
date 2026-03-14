"""Generate Figure 75: Mahalanobis Distance Comparison."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

dims = ['PCA-4', 'PCA-8', 'PCA-16', 'PCA-28', 'Full\n4096']
cosine = [0.906, 0.086, 0.995, 0.670, 1.000]
mahalanobis = [0.541, 0.978, 1.000, 1.000, 0]
euclidean = [0.000, 0.000, 0.000, 0.000, 1.000]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel (a): AUROC comparison
ax = axes[0]
x = np.arange(len(dims))
width = 0.25

bars1 = ax.bar(x - width, cosine, width, label='Cosine', color='#2196F3',
               edgecolor='black', linewidth=0.5, alpha=0.85)
bars2 = ax.bar(x, mahalanobis, width, label='Mahalanobis', color='#4CAF50',
               edgecolor='black', linewidth=0.5, alpha=0.85)
bars3 = ax.bar(x + width, euclidean, width, label='Euclidean', color='#FF9800',
               edgecolor='black', linewidth=0.5, alpha=0.85)

ax.text(4, 0.05, 'N/A', ha='center', va='bottom', fontsize=8, color='gray')

ax.set_xticks(x)
ax.set_xticklabels(dims, fontsize=10)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(a) Distance Metric Comparison', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 1.15)
ax.axhline(y=0.95, color='green', linestyle='--', alpha=0.3)

ax.annotate('Cosine fails\nin PCA-8!', xy=(1, 0.086), xytext=(2, 0.2),
            fontsize=9, color='darkred', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='darkred'))
ax.annotate('Mahalanobis\nperfect at PCA-16+', xy=(2, 1.0), xytext=(0.5, 0.7),
            fontsize=9, color='darkgreen', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='darkgreen'))

# Panel (b): Recommendation summary
ax = axes[1]
ax.axis('off')

summary_text = """
DISTANCE METRIC RECOMMENDATIONS

Full Dimensionality (4096):
  → Cosine: AUROC = 1.000 ✓
  → Euclidean: AUROC = 1.000 ✓
  → Both work perfectly

Reduced Dimensionality (PCA):
  → Mahalanobis: 1.000 at PCA-16+ ✓
  → Cosine: Unstable (0.086 at PCA-8!)
  → Euclidean: Always fails (0.000)

Key Insight:
  Cosine distance works only because full-dim
  embeddings are approximately isotropic.
  In PCA space, covariance structure matters
  → use Mahalanobis for reduced-dim deployment.
"""
ax.text(0.1, 0.95, summary_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax.set_title('(b) Recommendations', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig75_mahalanobis.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig75_mahalanobis.pdf', dpi=200, bbox_inches='tight')
print("Saved fig75_mahalanobis.png/pdf")
