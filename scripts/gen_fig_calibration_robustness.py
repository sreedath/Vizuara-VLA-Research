"""Generate Figure 32: Calibration Robustness Study."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/calibration_robustness_20260314_165245.json"
OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

with open(RESULTS) as f:
    data = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# Panel (a): Size sensitivity
ax = axes[0]
sizes = [r['size'] for r in data['size_sensitivity']]
combo_means = [r['combo_mean'] for r in data['size_sensitivity']]
combo_stds = [r['combo_std'] for r in data['size_sensitivity']]
cos_means = [r['cos_mean'] for r in data['size_sensitivity']]
cos_stds = [r['cos_std'] for r in data['size_sensitivity']]

ax.errorbar(sizes, combo_means, yerr=combo_stds, fmt='s-', color='#2196F3',
            linewidth=2, markersize=8, capsize=4, label='Cos+Mass combo')
ax.errorbar(sizes, cos_means, yerr=cos_stds, fmt='o-', color='#FF9800',
            linewidth=2, markersize=8, capsize=4, label='Cosine only')
ax.set_xlabel('Calibration Set Size', fontsize=11)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(a) Size Sensitivity', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.5, 0.85)

# Panel (b): Composition
ax = axes[1]
comp = data['composition']
names = [c['name'].split('(')[0].strip() for c in comp]
global_aurocs = [c['combo_global'] for c in comp]
perscene_aurocs = [c['combo_perscene'] for c in comp]

x = np.arange(len(names))
width = 0.35
bars1 = ax.barh(x - width/2, global_aurocs, width, label='Global centroid',
                color='#FF9800', edgecolor='black', linewidth=0.5)
bars2 = ax.barh(x + width/2, perscene_aurocs, width, label='Per-scene',
                color='#2196F3', edgecolor='black', linewidth=0.5)

ax.set_yticks(x)
ax.set_yticklabels(names, fontsize=7)
ax.set_xlabel('AUROC', fontsize=11)
ax.set_title('(b) Calibration Composition', fontsize=12, fontweight='bold')
ax.set_xlim(0.45, 0.85)
ax.legend(fontsize=8, loc='lower right')
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

for bar, val in zip(list(bars1) + list(bars2), global_aurocs + perscene_aurocs):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
            ha='left', va='center', fontsize=7, fontweight='bold')

# Panel (c): Transfer heatmap
ax = axes[2]
transfer = data['transfer']
cal_scenes = [t['cal_scene'] for t in transfer]
ood_types = ['snow', 'flooded', 'offroad', 'tunnel']

heatmap = np.array([[t[ood] for ood in ood_types] for t in transfer])

im = ax.imshow(heatmap, cmap='RdYlGn', vmin=0.3, vmax=1.0, aspect='auto')
ax.set_xticks(range(4))
ax.set_xticklabels(ood_types, fontsize=9, rotation=20)
ax.set_yticks(range(4))
ax.set_yticklabels(cal_scenes, fontsize=9)
ax.set_title('(c) Transfer: Cal → OOD Detection', fontsize=12, fontweight='bold')

for i in range(4):
    for j in range(4):
        color = 'white' if heatmap[i, j] < 0.6 else 'black'
        ax.text(j, i, f'{heatmap[i, j]:.2f}', ha='center', va='center',
                fontsize=10, fontweight='bold', color=color)

plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig32_calibration_robustness.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig32_calibration_robustness.pdf', dpi=200, bbox_inches='tight')
print("Saved fig32_calibration_robustness.png/pdf")
