"""Generate Figure 100: Layer-wise Hidden State Analysis."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/layer_analysis_20260314_221354.json") as f:
    data = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

lr = data['layer_results']
layers = sorted([int(k) for k in lr.keys()])
aurocs = [lr[str(l)]['auroc'] for l in layers]
ds = [lr[str(l)]['d'] for l in layers]
norms = [lr[str(l)]['mean_norm'] for l in layers]

# Panel (a): AUROC and d by layer
ax = axes[0]
ax2 = ax.twinx()
ln1 = ax.plot(layers, aurocs, 'o-', color='#2196F3', linewidth=2, markersize=5, label='AUROC')
ln2 = ax2.plot(layers, ds, 's-', color='#F44336', linewidth=2, markersize=5, label="Cohen's d")
ax.set_xlabel('Layer Index', fontsize=11)
ax.set_ylabel('AUROC', fontsize=11, color='#2196F3')
ax2.set_ylabel("Cohen's d", fontsize=11, color='#F44336')
ax.set_title('(a) OOD Detection by Layer', fontsize=12, fontweight='bold')
ax.set_ylim(0.4, 1.1)
ax.grid(True, alpha=0.3)

# Combined legend
lns = ln1 + ln2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, fontsize=9, loc='center right')

# Panel (b): Hidden state norms by layer
ax = axes[1]
ax.plot(layers, norms, 'D-', color='#9C27B0', linewidth=2, markersize=5)
ax.set_xlabel('Layer Index', fontsize=11)
ax.set_ylabel('Mean Hidden State Norm', fontsize=11)
ax.set_title('(b) Representation Norm by Layer', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Mark peak d layer
peak_layer = layers[np.argmax(ds)]
ax.axvline(x=peak_layer, color='red', linestyle='--', linewidth=1, alpha=0.7,
           label=f'Peak d at layer {peak_layer}')
ax.legend(fontsize=9)

# Panel (c): Multi-layer fusion comparison
ax = axes[2]
fusion = data['fusion_results']
methods = ['last_only', 'last_two', 'last_four', 'every_4th', 'all_layers']
labels = ['Last\n(4K)', 'Last 2\n(8K)', 'Last 4\n(16K)', 'Every 4th\n(37K)', 'All 33\n(135K)']
fusion_ds = [fusion[m]['d'] for m in methods]
fusion_aurocs = [fusion[m]['auroc'] for m in methods]

colors = ['#4CAF50' if a >= 1.0 else '#FF9800' for a in fusion_aurocs]
bars = ax.bar(range(len(methods)), fusion_ds, color=colors, alpha=0.8,
              edgecolor='black', linewidth=0.5)
for i, (bar, auroc) in enumerate(zip(bars, fusion_aurocs)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'AUROC={auroc:.3f}', ha='center', va='bottom', fontsize=7)
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Cohen's d", fontsize=11)
ax.set_title('(c) Multi-Layer Fusion', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig100_layer_analysis.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig100_layer_analysis.pdf', dpi=200, bbox_inches='tight')
print("Saved fig100_layer_analysis.png/pdf")
