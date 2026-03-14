"""Generate Figure 69: Feature Correlation Analysis."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

# Correlation matrix from experiment 83
feature_names = ['cosine\ndist', 'attn\nmax', 'attn\nentropy', 'output\nentropy',
                 'top1\nprob', 'top5\nprob', 'hidden\nnorm', 'attn\nmean']
corr = np.array([
    [1.000, 0.686, -0.643, 0.295, -0.304, -0.325, -0.116, 0.106],
    [0.686, 1.000, -0.676, 0.519, -0.450, -0.542, -0.435, 0.189],
    [-0.643, -0.676, 1.000, -0.827, 0.725, 0.849, 0.617, -0.353],
    [0.295, 0.519, -0.827, 1.000, -0.914, -0.961, -0.722, 0.386],
    [-0.304, -0.450, 0.725, -0.914, 1.000, 0.798, 0.437, -0.224],
    [-0.325, -0.542, 0.849, -0.961, 0.798, 1.000, 0.828, -0.447],
    [-0.116, -0.435, 0.617, -0.722, 0.437, 0.828, 1.000, -0.470],
    [0.106, 0.189, -0.353, 0.386, -0.224, -0.447, -0.470, 1.000],
])

aurocs = [1.000, 0.918, 0.965, 0.653, 0.724, 0.621, 0.841, 0.502]
auroc_names = ['cosine_dist', 'attn_max', 'attn_entropy', 'output_entropy',
               'top1_prob', 'top5_prob', 'hidden_norm', 'attn_mean']

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Panel (a): Correlation heatmap
ax = axes[0]
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(len(feature_names)))
ax.set_xticklabels(feature_names, fontsize=8, rotation=0, ha='center')
ax.set_yticks(range(len(feature_names)))
ax.set_yticklabels(feature_names, fontsize=8)
ax.set_title('(a) Feature Correlation Matrix', fontsize=12, fontweight='bold')

# Add correlation values
for i in range(len(feature_names)):
    for j in range(len(feature_names)):
        color = 'white' if abs(corr[i, j]) > 0.6 else 'black'
        ax.text(j, i, f'{corr[i,j]:.2f}', ha='center', va='center',
                fontsize=7, color=color, fontweight='bold')

plt.colorbar(im, ax=ax, shrink=0.8)

# Panel (b): Per-feature AUROC
ax = axes[1]
sorted_idx = np.argsort(aurocs)[::-1]
sorted_aurocs = [aurocs[i] for i in sorted_idx]
sorted_names = [auroc_names[i] for i in sorted_idx]

colors = []
for a in sorted_aurocs:
    if a >= 0.95:
        colors.append('#4CAF50')
    elif a >= 0.8:
        colors.append('#FF9800')
    elif a >= 0.7:
        colors.append('#FFC107')
    else:
        colors.append('#F44336')

bars = ax.barh(range(len(sorted_names)), sorted_aurocs, 0.6, color=colors,
               edgecolor='black', linewidth=0.5, alpha=0.85)
ax.set_yticks(range(len(sorted_names)))
ax.set_yticklabels(sorted_names, fontsize=10)
ax.set_xlabel('AUROC', fontsize=11)
ax.set_title('(b) Per-Feature OOD Detection AUROC', fontsize=12, fontweight='bold')
ax.set_xlim(0.4, 1.05)
ax.axvline(x=0.95, color='green', linestyle='--', alpha=0.3, label='Excellent')
ax.axvline(x=0.80, color='orange', linestyle='--', alpha=0.3, label='Good')
ax.grid(True, alpha=0.3, axis='x')
ax.legend(fontsize=8)

for bar, v in zip(bars, sorted_aurocs):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2.,
            f'{v:.3f}', ha='left', va='center', fontsize=9, fontweight='bold')

# Annotate key finding
ax.annotate('Cosine + Attn:\nr=0.686\n(complementary!)', xy=(0.95, 1), xytext=(0.6, 3),
            fontsize=9, color='darkblue', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='darkblue'))

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig69_feature_correlation.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig69_feature_correlation.pdf', dpi=200, bbox_inches='tight')
print("Saved fig69_feature_correlation.png/pdf")
