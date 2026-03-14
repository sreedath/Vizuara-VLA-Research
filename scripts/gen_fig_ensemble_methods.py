"""Generate Figure 98: Ensemble Detection Methods."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/ensemble_methods_20260314_220250.json") as f:
    data = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel (a): Individual detector comparison
ax = axes[0]
detectors = ['cosine', 'norm', 'top1_inv', 'entropy']
labels = ['Cosine\nDistance', 'Norm\nDeviation', '1 - Top1\nProb', 'Logit\nEntropy']
aurocs = [data['individual'][d]['auroc'] for d in detectors]
ds = [data['individual'][d]['d'] for d in detectors]
colors = ['#4CAF50' if a >= 1.0 else '#FF9800' if a >= 0.9 else '#F44336' for a in aurocs]

x = np.arange(len(detectors))
bars = ax.bar(x, aurocs, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
for i, (bar, d_val) in enumerate(zip(bars, ds)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'd={d_val:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(a) Individual Detector Performance', fontsize=12, fontweight='bold')
ax.set_ylim(0.5, 1.1)
ax.axhline(y=1.0, color='green', linestyle='--', linewidth=0.8, alpha=0.5)
ax.grid(True, alpha=0.3, axis='y')

# Panel (b): Ensemble vs individual comparison
ax = axes[1]
methods = ['cosine', 'avg_cos_norm', 'avg_cos_ent', 'avg_cos_top1', 'avg_all',
           'max_cos_ent', 'prod_cos_ent']
method_labels = ['Cosine\n(alone)', 'Avg\ncos+norm', 'Avg\ncos+ent', 'Avg\ncos+top1',
                 'Avg\nall 4', 'Max\ncos+ent', 'Prod\ncos+ent']
method_aurocs = []
method_ds = []
for m in methods:
    if m in data['individual']:
        method_aurocs.append(data['individual'][m]['auroc'])
        method_ds.append(data['individual'][m]['d'])
    else:
        method_aurocs.append(data['ensemble'][m]['auroc'])
        method_ds.append(data['ensemble'][m]['d'])

colors2 = ['#4CAF50' if a >= 1.0 else '#FF9800' if a >= 0.9 else '#F44336' for a in method_aurocs]
x2 = np.arange(len(methods))
bars2 = ax.bar(x2, method_ds, color=colors2, alpha=0.8, edgecolor='black', linewidth=0.5)
for i, (bar, auroc_val) in enumerate(zip(bars2, method_aurocs)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{auroc_val:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)
ax.set_xticks(x2)
ax.set_xticklabels(method_labels, fontsize=7)
ax.set_ylabel("Cohen's d", fontsize=11)
ax.set_title('(b) Ensemble vs Individual (d)', fontsize=12, fontweight='bold')
ax.axhline(y=data['individual']['cosine']['d'], color='green', linestyle='--',
           linewidth=1.5, alpha=0.7, label=f'Cosine baseline d={data["individual"]["cosine"]["d"]:.1f}')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# Panel (c): Signal correlation heatmap
ax = axes[2]
signal_names = ['Cosine', 'Entropy', 'Norm', 'Top1-inv']
corr = np.ones((4, 4))
name_map = {'cosine': 0, 'entropy': 1, 'norm': 2, 'top1_inv': 3}
for key, val in data['correlations'].items():
    parts = key.split('_vs_')
    i = name_map[parts[0]]
    j = name_map[parts[1]]
    corr[i, j] = val
    corr[j, i] = val

im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(4))
ax.set_yticks(range(4))
ax.set_xticklabels(signal_names, fontsize=9, rotation=30, ha='right')
ax.set_yticklabels(signal_names, fontsize=9)
for i in range(4):
    for j in range(4):
        color = 'white' if abs(corr[i, j]) > 0.5 else 'black'
        ax.text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center',
                fontsize=10, fontweight='bold', color=color)
ax.set_title('(c) Signal Correlations', fontsize=12, fontweight='bold')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig98_ensemble_methods.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig98_ensemble_methods.pdf', dpi=200, bbox_inches='tight')
print("Saved fig98_ensemble_methods.png/pdf")
