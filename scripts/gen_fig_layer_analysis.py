"""Generate Figure 36: Layer-wise Hidden State Analysis."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/layer_analysis_20260314_172307.json") as f:
    data = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Panel (a): Per-layer AUROC
ax = axes[0]
layers = sorted([int(k) for k in data['layer_aurocs'].keys()])
aurocs = [data['layer_aurocs'][str(l)] for l in layers]

colors = ['#F44336' if a < 0.7 else '#FF9800' if a < 0.85 else '#4CAF50' if a < 0.9 else '#2196F3'
          for a in aurocs]

bars = ax.bar(range(len(layers)), aurocs, color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)
ax.set_xticks(range(len(layers)))
ax.set_xticklabels([f'L{l}' for l in layers], fontsize=8, rotation=45)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(a) Per-Layer OOD Detection AUROC', fontsize=12, fontweight='bold')
ax.set_ylim(0.4, 1.0)
ax.axhline(y=0.5, color='red', linestyle=':', alpha=0.3, label='Random')
ax.grid(True, alpha=0.3, axis='y')

# Annotate best
best_idx = np.argmax(aurocs)
ax.annotate(f'Best: {aurocs[best_idx]:.3f}',
            xy=(best_idx, aurocs[best_idx]),
            xytext=(best_idx - 2, aurocs[best_idx] + 0.03),
            fontsize=9, fontweight='bold', color='#2196F3',
            arrowprops=dict(arrowstyle='->', color='#2196F3'))

# Panel (b): Hidden state norm growth across layers
ax = axes[1]
geom = data['geometry']
geom_layers = sorted([int(k) for k in geom.keys()])
id_norms = [geom[str(l)]['id_norm'] for l in geom_layers]
ood_norms = [geom[str(l)]['ood_norm'] for l in geom_layers]

ax.plot(range(len(geom_layers)), id_norms, 'o-', color='#2196F3', linewidth=2,
        markersize=6, label='ID', zorder=3)
ax.plot(range(len(geom_layers)), ood_norms, 's-', color='#F44336', linewidth=2,
        markersize=6, label='OOD', zorder=3)
ax.set_xticks(range(len(geom_layers)))
ax.set_xticklabels([f'L{l}' for l in geom_layers], fontsize=8, rotation=45)
ax.set_ylabel('Hidden State Norm', fontsize=11)
ax.set_title('(b) Representation Norm by Layer', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel (c): Per-scenario per-layer heatmap
ax = axes[2]
ood_types = ['noise', 'indoor', 'inverted', 'blackout']
# Build heatmap from results
results = data['results']
id_data = [r for r in results if not r['is_ood']]
ood_data = {t: [r for r in results if r['scenario'] == t] for t in ood_types}

heatmap = np.zeros((len(layers), len(ood_types)))
for i, l in enumerate(layers):
    key = f'cos_L{l}'
    for j, ood_type in enumerate(ood_types):
        type_ood = ood_data[ood_type]
        type_all = id_data + type_ood
        labels = [0]*len(id_data) + [1]*len(type_ood)
        scores = [r[key] for r in type_all]
        from sklearn.metrics import roc_auc_score
        try:
            heatmap[i, j] = roc_auc_score(labels, scores)
        except:
            heatmap[i, j] = 0.5

im = ax.imshow(heatmap.T, cmap='RdYlGn', aspect='auto', vmin=0.3, vmax=1.0)
ax.set_xticks(range(len(layers)))
ax.set_xticklabels([f'L{l}' for l in layers], fontsize=8, rotation=45)
ax.set_yticks(range(len(ood_types)))
ax.set_yticklabels(ood_types, fontsize=10)
ax.set_title('(c) Per-Layer Per-OOD AUROC', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, label='AUROC', shrink=0.8)

# Add text annotations
for i in range(len(layers)):
    for j in range(len(ood_types)):
        val = heatmap[i, j]
        color = 'white' if val < 0.6 else 'black'
        ax.text(i, j, f'{val:.2f}', ha='center', va='center', fontsize=7,
                fontweight='bold', color=color)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig36_layer_analysis.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig36_layer_analysis.pdf', dpi=200, bbox_inches='tight')
print("Saved fig36_layer_analysis.png/pdf")
