"""Generate Figure 86: Comprehensive Ablation Study."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/comprehensive_ablation_20260314_211759.json") as f:
    data = json.load(f)

results = data['results']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel (a): Feature source ablation
ax = axes[0]
feats = results['feature_source']
feat_names = ['Multi-layer\nPCA-8', 'Last\nLayer', 'Layer\n24', 'Norm\nOnly']
feat_keys = ['multi_layer_pca8', 'last_layer', 'layer_24', 'norm_only']
aurocs = [feats[k]['auroc'] for k in feat_keys]
ds = [feats[k]['d'] if feats[k]['d'] else 0 for k in feat_keys]

colors = ['#4CAF50' if a >= 1.0 else '#FF9800' for a in aurocs]
bars = ax.bar(range(len(feat_names)), aurocs, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(feat_names)))
ax.set_xticklabels(feat_names, fontsize=9)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(a) Feature Source', fontsize=12, fontweight='bold')
ax.set_ylim(0.9, 1.02)
ax.grid(True, alpha=0.3, axis='y')

for i, (a, d) in enumerate(zip(aurocs, ds)):
    label = f'{a:.3f}\nd={d:.1f}' if d else f'{a:.3f}'
    ax.text(i, a + 0.003, label, ha='center', va='bottom', fontsize=8, fontweight='bold')

# Panel (b): Calibration size ablation
ax = axes[1]
cal = results['calibration_size']
cal_names = sorted(cal.keys(), key=lambda k: cal[k]['n'])
ns = [cal[k]['n'] for k in cal_names]
cal_aurocs = [cal[k]['auroc'] for k in cal_names]
cal_ds = [cal[k]['d'] for k in cal_names]

ax.plot(ns, cal_aurocs, 'o-', color='#2196F3', linewidth=2, markersize=8, label='AUROC')
ax2 = ax.twinx()
ax2.plot(ns, cal_ds, 's--', color='#F44336', linewidth=2, markersize=8, label="Cohen's d")
ax.set_xlabel('Calibration Samples (N)', fontsize=11)
ax.set_ylabel('AUROC', fontsize=11, color='#2196F3')
ax2.set_ylabel("Cohen's d", fontsize=11, color='#F44336')
ax.set_title('(b) Calibration Size', fontsize=12, fontweight='bold')
ax.set_ylim(0.98, 1.005)
ax2.set_ylim(0, 7)
ax.grid(True, alpha=0.3)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='center right')

# Panel (c): Summary recommendation table
ax = axes[2]
ax.axis('off')

# Create table
cell_text = [
    ['Maximum d', 'Multi-layer PCA-8', 'd=15.78', '8', '≥20'],
    ['Simplest perfect', 'Last-layer cosine', 'd=5.75', '4096', '≥20'],
    ['Minimum data', 'Last-layer cosine', 'AUROC=0.998', '4096', '1'],
    ['No calibration', 'L2 norm', 'AUROC=0.975', '1', '0'],
]
col_labels = ['Goal', 'Method', 'Performance', 'Dims', 'N_cal']

table = ax.table(cellText=cell_text, colLabels=col_labels,
                cellLoc='center', loc='center', colWidths=[0.2, 0.25, 0.2, 0.1, 0.1])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.6)

# Color header
for j in range(len(col_labels)):
    table[0, j].set_facecolor('#2196F3')
    table[0, j].set_text_props(color='white', fontweight='bold')

# Color rows
row_colors = ['#E8F5E9', '#E3F2FD', '#FFF3E0', '#FFEBEE']
for i in range(4):
    for j in range(5):
        table[i+1, j].set_facecolor(row_colors[i])

ax.set_title('(c) Deployment Recommendations', fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig86_comprehensive_ablation.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig86_comprehensive_ablation.pdf', dpi=200, bbox_inches='tight')
print("Saved fig86_comprehensive_ablation.png/pdf")
