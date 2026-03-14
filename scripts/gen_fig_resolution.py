"""Generate Figure 61: Resolution Sensitivity Analysis."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Panel (a): AUROC by resolution
ax = axes[0]
resolutions = [64, 128, 224, 256, 512]
cos_aurocs = [1.000, 1.000, 1.000, 1.000, 1.000]
attn_aurocs = [1.000, 1.000, 1.000, 1.000, 1.000]

ax.plot(resolutions, cos_aurocs, 'ro-', linewidth=2, markersize=10, label='Cosine')
ax.plot(resolutions, attn_aurocs, 'gs-', linewidth=2, markersize=10, label='Attn Max')
ax.set_xlabel('Input Resolution (pixels)', fontsize=11)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(a) Detection AUROC vs Resolution', fontsize=12, fontweight='bold')
ax.set_ylim(0.9, 1.02)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.annotate('Perfect at ALL\nresolutions!', xy=(224, 1.0), xytext=(350, 0.95),
            fontsize=10, color='green', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='green'))

# Panel (b): Cross-resolution centroid distances
ax = axes[1]
labels_res = ['64', '128', '224', '256', '512']
dist_matrix = np.zeros((5, 5))
dists = {
    (0,1): 0.063, (0,2): 0.132, (0,3): 0.161, (0,4): 0.129,
    (1,2): 0.037, (1,3): 0.059, (1,4): 0.030,
    (2,3): 0.005, (2,4): 0.005,
    (3,4): 0.012,
}
for (i,j), d in dists.items():
    dist_matrix[i,j] = d
    dist_matrix[j,i] = d

im = ax.imshow(dist_matrix, cmap='YlOrRd', vmin=0, vmax=0.17)
ax.set_xticks(range(5))
ax.set_yticks(range(5))
ax.set_xticklabels(labels_res, fontsize=10)
ax.set_yticklabels(labels_res, fontsize=10)
ax.set_title('(b) Cross-Resolution\nCentroid Distance', fontsize=12, fontweight='bold')
ax.set_xlabel('Resolution', fontsize=11)
ax.set_ylabel('Resolution', fontsize=11)
for i in range(5):
    for j in range(5):
        color = 'white' if dist_matrix[i,j] > 0.1 else 'black'
        ax.text(j, i, f'{dist_matrix[i,j]:.3f}', ha='center', va='center',
                fontsize=9, color=color, fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig61_resolution.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig61_resolution.pdf', dpi=200, bbox_inches='tight')
print("Saved fig61_resolution.png/pdf")
