"""Generate Figure 88: Activation Statistics Analysis."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/activation_stats_20260314_212554.json") as f:
    data = json.load(f)

feat_aurocs = data['feature_aurocs']
cats = data['categories']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel (a): Feature AUROC ranking
ax = axes[0]
features = sorted(feat_aurocs.keys(), key=lambda k: feat_aurocs[k]['auroc'], reverse=True)
aurocs = [feat_aurocs[f]['auroc'] for f in features]
colors = ['#4CAF50' if a > 0.97 else '#2196F3' if a > 0.9 else '#FF9800' if a > 0.7 else '#F44336' for a in aurocs]

ax.barh(range(len(features)), aurocs, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(features)))
ax.set_yticklabels([f.replace('_', '\n') for f in features], fontsize=8)
ax.set_xlabel('AUROC', fontsize=11)
ax.set_title('(a) Activation Feature Ranking', fontsize=12, fontweight='bold')
ax.set_xlim(0.5, 1.02)
ax.axvline(x=0.95, color='gray', linestyle='--', alpha=0.5)
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

# Panel (b): ID vs OOD key statistics
ax = axes[1]
key_stats = ['mean', 'std', 'kurtosis', 'sparsity_1']
stat_labels = ['Mean', 'Std Dev', 'Kurtosis', 'Sparsity\n(|x|<1)']
cat_order = ['highway', 'urban', 'noise', 'indoor', 'twilight', 'snow']
cat_colors = {'highway': '#4CAF50', 'urban': '#2196F3', 'noise': '#F44336',
              'indoor': '#FF9800', 'twilight': '#9C27B0', 'snow': '#00BCD4'}

x = np.arange(len(key_stats))
width = 0.12
for i, cat in enumerate(cat_order):
    vals = []
    for stat in key_stats:
        vals.append(cats[cat]['stats_summary'][stat]['mean'])
    # Normalize for display
    vals_norm = []
    for j, stat in enumerate(key_stats):
        all_vals = [cats[c]['stats_summary'][stat]['mean'] for c in cat_order]
        vmin, vmax = min(all_vals), max(all_vals)
        vals_norm.append((vals[j] - vmin) / (vmax - vmin + 1e-10))
    ax.bar(x + i * width - 2.5*width, vals_norm, width, color=cat_colors[cat],
           alpha=0.7, label=cat, edgecolor='black', linewidth=0.3)

ax.set_xticks(x)
ax.set_xticklabels(stat_labels, fontsize=9)
ax.set_ylabel('Normalized Value', fontsize=11)
ax.set_title('(b) Key Statistics by Category', fontsize=12, fontweight='bold')
ax.legend(fontsize=7, ncol=3, loc='upper right')
ax.grid(True, alpha=0.3, axis='y')

# Panel (c): OOD activation profile
ax = axes[2]
# Show sparsity at different thresholds for ID vs OOD
thresholds_labels = ['|x|<0.01', '|x|<0.1', '|x|<1.0']
id_sparsity = []
ood_sparsity = []
for key in ['sparsity_0', 'sparsity_01', 'sparsity_1']:
    id_vals = []
    ood_vals = []
    for cat in cat_order:
        val = cats[cat]['stats_summary'][key]['mean']
        if cats[cat]['group'] == 'ID':
            id_vals.append(val)
        else:
            ood_vals.append(val)
    id_sparsity.append(np.mean(id_vals))
    ood_sparsity.append(np.mean(ood_vals))

x = np.arange(len(thresholds_labels))
width = 0.3
ax.bar(x - width/2, id_sparsity, width, label='ID', color='#4CAF50', alpha=0.7, edgecolor='black', linewidth=0.5)
ax.bar(x + width/2, ood_sparsity, width, label='OOD', color='#F44336', alpha=0.7, edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(thresholds_labels, fontsize=10)
ax.set_ylabel('Fraction of Activations', fontsize=11)
ax.set_title('(c) Sparsity Profile', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig88_activation_stats.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig88_activation_stats.pdf', dpi=200, bbox_inches='tight')
print("Saved fig88_activation_stats.png/pdf")
