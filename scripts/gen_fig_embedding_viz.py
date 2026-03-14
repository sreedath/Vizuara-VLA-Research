"""Generate Figure 41: Embedding Space Visualization."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/embedding_viz_20260314_175156.json") as f:
    data = json.load(f)

pca_2d = np.array(data['pca_2d'])
scenarios = data['scenarios']
is_ood = data['is_ood']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel (a): PCA scatter colored by scenario
ax = axes[0]
scenario_colors = {
    'highway': '#2196F3',
    'urban': '#03A9F4',
    'noise': '#F44336',
    'indoor': '#FF5722',
    'inverted': '#FF9800',
    'blackout': '#9C27B0',
    'blank': '#E91E63',
}
scenario_markers = {
    'highway': 'o', 'urban': 's',
    'noise': '^', 'indoor': 'v', 'inverted': 'D', 'blackout': 'X', 'blank': 'P',
}

for scene in sorted(set(scenarios)):
    mask = [s == scene for s in scenarios]
    pts = pca_2d[mask]
    ood = is_ood[scenarios.index(scene)]
    label = f'{scene} ({"OOD" if ood else "ID"})'
    ax.scatter(pts[:, 0], pts[:, 1], c=scenario_colors.get(scene, 'gray'),
               marker=scenario_markers.get(scene, 'o'), s=60, alpha=0.8,
               edgecolors='black', linewidths=0.5, label=label, zorder=3)

ax.set_xlabel('PC1 (8.9%)', fontsize=11)
ax.set_ylabel('PC2 (7.1%)', fontsize=11)
ax.set_title('(a) PCA: Per-Scenario Clusters', fontsize=12, fontweight='bold')
ax.legend(fontsize=7, loc='upper left', ncol=2)
ax.grid(True, alpha=0.2)

# Panel (b): PCA scatter colored by ID/OOD
ax = axes[1]
id_mask = [not ood for ood in is_ood]
ood_mask = is_ood

ax.scatter(pca_2d[id_mask, 0], pca_2d[id_mask, 1], c='#2196F3', s=70, alpha=0.8,
           edgecolors='black', linewidths=0.5, label='In-Distribution', zorder=3, marker='o')
ax.scatter(pca_2d[ood_mask, 0], pca_2d[ood_mask, 1], c='#F44336', s=70, alpha=0.8,
           edgecolors='black', linewidths=0.5, label='Out-of-Distribution', zorder=3, marker='^')

# Draw centroids
centroids = data['scenario_centroids_2d']
id_centroid = np.mean(pca_2d[id_mask], axis=0)
ood_centroid = np.mean(pca_2d[ood_mask], axis=0)
ax.plot(*id_centroid, 'b*', markersize=20, markeredgecolor='black', markeredgewidth=1.5,
        zorder=5, label='ID centroid')
ax.plot(*ood_centroid, 'r*', markersize=20, markeredgecolor='black', markeredgewidth=1.5,
        zorder=5, label='OOD centroid')

# Draw line between centroids
ax.plot([id_centroid[0], ood_centroid[0]], [id_centroid[1], ood_centroid[1]],
        'k--', linewidth=1.5, alpha=0.5)

ax.set_xlabel('PC1 (8.9%)', fontsize=11)
ax.set_ylabel('PC2 (7.1%)', fontsize=11)
ax.set_title('(b) PCA: ID vs OOD Separation', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

# Panel (c): PCA explained variance
ax = axes[2]
explained = data['pca_explained_variance']
cumulative = np.cumsum(explained)

ax.bar(range(1, len(explained)+1), [e*100 for e in explained],
       color='#2196F3', edgecolor='black', linewidth=0.5, alpha=0.85, label='Individual')
ax2 = ax.twinx()
ax2.plot(range(1, len(explained)+1), [c*100 for c in cumulative],
         'r-o', linewidth=2, markersize=6, label='Cumulative')
ax2.axhline(y=47.8, color='green', linestyle='--', alpha=0.5, label='10 PCs: 47.8%')

ax.set_xlabel('Principal Component', fontsize=11)
ax.set_ylabel('Explained Variance (%)', fontsize=11, color='#2196F3')
ax2.set_ylabel('Cumulative (%)', fontsize=11, color='red')
ax.set_title('(c) PCA Variance Decomposition', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.2)

# Combine legends
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='center right')

# Add key stats as text
stats_text = (f'AUROC = {data["auroc"]:.3f}\n'
              f'Silhouette = {data["silhouette_score"]:.3f}\n'
              f'Inter/Intra = {data["inter_dist"]/max(data["id_intra_dist"], data["ood_intra_dist"]):.2f}')
axes[1].text(0.02, 0.02, stats_text, transform=axes[1].transAxes, fontsize=9,
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig41_embedding_viz.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig41_embedding_viz.pdf', dpi=200, bbox_inches='tight')
print("Saved fig41_embedding_viz.png/pdf")
