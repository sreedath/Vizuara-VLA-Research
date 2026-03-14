"""Generate Figure 105: Embedding Geometry Analysis."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/embedding_geometry_20260314_225144.json") as f:
    data = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel (a): PCA cumulative variance
ax = axes[0]
n_comp = len(data['pca_cumvar'])
ax.plot(range(1, n_comp+1), data['pca_cumvar'], 'o-', color='#9C27B0', linewidth=2, markersize=4, label='All data')
n_id = len(data['id_pca_cumvar'])
ax.plot(range(1, n_id+1), data['id_pca_cumvar'], 's-', color='#4CAF50', linewidth=2, markersize=4, label='ID only')
n_ood = len(data['ood_pca_cumvar'])
ax.plot(range(1, n_ood+1), data['ood_pca_cumvar'], 'D-', color='#F44336', linewidth=2, markersize=4, label='OOD only')
ax.axhline(y=0.9, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.axhline(y=0.95, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('Number of PCs', fontsize=11)
ax.set_ylabel('Cumulative Variance Explained', fontsize=11)
ax.set_title('(a) Intrinsic Dimensionality', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel (b): Angular distribution
ax = axes[1]
cats = ['highway', 'urban', 'snow', 'indoor', 'twilight', 'noise']
angles = [data['angular_stats'][c]['mean_angle'] for c in cats]
angle_stds = [data['angular_stats'][c]['std_angle'] for c in cats]
groups = [data['angular_stats'][c]['group'] for c in cats]
colors = ['#4CAF50' if g == 'ID' else '#F44336' for g in groups]

bars = ax.bar(range(len(cats)), angles, yerr=angle_stds,
              color=colors, alpha=0.8, edgecolor='black', linewidth=0.5,
              capsize=3)
ax.set_xticks(range(len(cats)))
ax.set_xticklabels(cats, fontsize=9)
ax.set_ylabel('Angular Distance to ID Centroid (°)', fontsize=11)
ax.set_title('(b) Angular Distribution', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# ID/OOD boundary
ax.axhline(y=30, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Decision boundary')
ax.legend(fontsize=9)

# Panel (c): Intra vs inter distances
ax = axes[2]
intra = data['intra_distances']
cats_all = list(intra.keys())
intra_means = [intra[c]['mean'] for c in cats_all]
intra_groups = [intra[c]['group'] for c in cats_all]
colors_intra = ['#4CAF50' if g == 'ID' else '#F44336' for g in intra_groups]

x = np.arange(len(cats_all))
bars = ax.bar(x, intra_means, color=colors_intra, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.axhline(y=data['compactness']['id_ood_inter'], color='blue', linestyle='--',
           linewidth=2, label=f'ID-OOD inter={data["compactness"]["id_ood_inter"]:.3f}')
ax.set_xticks(x)
ax.set_xticklabels(cats_all, fontsize=8, rotation=30, ha='right')
ax.set_ylabel('Mean Cosine Distance', fontsize=11)
ax.set_title(f'(c) Compactness Ratio: {data["compactness"]["ratio"]:.1f}×', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig105_embedding_geometry.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig105_embedding_geometry.pdf', dpi=200, bbox_inches='tight')
print("Saved fig105_embedding_geometry.png/pdf")
