"""Generate Figure 80: Embedding Space Geometry."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/embedding_geometry_20260314_205339.json") as f:
    data = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel (a): 2D PCA scatter plot
ax = axes[0]
colors_map = {'highway': '#4CAF50', 'urban': '#2196F3', 'noise': '#F44336',
              'indoor': '#FF9800', 'twilight': '#9C27B0', 'snow': '#00BCD4'}
markers_map = {'highway': 'o', 'urban': 's', 'noise': '^', 'indoor': 'D',
               'twilight': 'v', 'snow': 'P'}

for name in ['highway', 'urban', 'noise', 'indoor', 'twilight', 'snow']:
    coords = np.array(data['coords_2d'][name])
    ax.scatter(coords[:, 0], coords[:, 1], c=colors_map[name], marker=markers_map[name],
               s=50, label=name, alpha=0.8, edgecolors='black', linewidth=0.3)

ax.set_xlabel('PC1', fontsize=11)
ax.set_ylabel('PC2', fontsize=11)
ax.set_title(f'(a) PCA-2 Embedding Space ({data["pca_2d_explained_var"]:.1%} var)', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

# Panel (b): Intrinsic dimensionality
ax = axes[1]
thresholds = ['dim_90', 'dim_95', 'dim_99']
labels = ['90%', '95%', '99%']
id_dims = [data['intrinsic_dimensionality']['id'][t] for t in thresholds]
ood_dims = [data['intrinsic_dimensionality']['ood'][t] for t in thresholds]
all_dims = [data['intrinsic_dimensionality']['all'][t] for t in thresholds]

x = np.arange(len(thresholds))
width = 0.25
ax.bar(x - width, id_dims, width, label='ID', color='#4CAF50', alpha=0.8, edgecolor='black', linewidth=0.5)
ax.bar(x, ood_dims, width, label='OOD', color='#F44336', alpha=0.8, edgecolor='black', linewidth=0.5)
ax.bar(x + width, all_dims, width, label='All', color='#2196F3', alpha=0.8, edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_xlabel('Variance Threshold', fontsize=11)
ax.set_ylabel('Intrinsic Dimensions', fontsize=11)
ax.set_title('(b) Intrinsic Dimensionality', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

for i, (id_d, ood_d, all_d) in enumerate(zip(id_dims, ood_dims, all_dims)):
    ax.text(i - width, id_d + 0.5, str(id_d), ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.text(i, ood_d + 0.5, str(ood_d), ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.text(i + width, all_d + 0.5, str(all_d), ha='center', va='bottom', fontsize=8, fontweight='bold')

# Panel (c): Inter-category distance heatmap
ax = axes[2]
cats = ['highway', 'urban', 'snow', 'twilight', 'indoor', 'noise']
n = len(cats)
dist_matrix = np.zeros((n, n))
for i, c1 in enumerate(cats):
    for j, c2 in enumerate(cats):
        if i == j:
            dist_matrix[i, j] = 0
        else:
            key = f"{min(c1,c2)}_vs_{max(c1,c2)}"
            dist_matrix[i, j] = data['inter_category_distances'].get(key, 0)

im = ax.imshow(dist_matrix, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(n))
ax.set_xticklabels(cats, fontsize=9, rotation=45, ha='right')
ax.set_yticks(range(n))
ax.set_yticklabels(cats, fontsize=9)
ax.set_title('(c) Inter-Category Cosine Distance', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.8)

# Add values
for i in range(n):
    for j in range(n):
        if i != j:
            ax.text(j, i, f'{dist_matrix[i,j]:.2f}', ha='center', va='center', fontsize=7,
                   color='white' if dist_matrix[i,j] > 0.4 else 'black')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig80_embedding_geometry.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig80_embedding_geometry.pdf', dpi=200, bbox_inches='tight')
print("Saved fig80_embedding_geometry.png/pdf")
