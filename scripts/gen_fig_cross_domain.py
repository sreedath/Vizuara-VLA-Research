"""Generate Figure 92: Cross-Domain Generalization."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/cross_domain_20260314_214017.json") as f:
    data = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel (a): AUROC and d by calibration strategy
ax = axes[0]
scenarios = ['highway_calibrated', 'urban_calibrated', 'mixed_calibrated',
             'highway_to_urban', 'urban_to_highway']
labels = ['Highway\nCal', 'Urban\nCal', 'Mixed\nCal', 'Hwy->Urb\nTransfer', 'Urb->Hwy\nTransfer']
aurocs = [data['scenarios'][s]['auroc'] for s in scenarios]
ds = [data['scenarios'][s]['d'] for s in scenarios]
colors = ['#2196F3', '#FF9800', '#4CAF50', '#F44336', '#9C27B0']

bars = ax.bar(range(len(scenarios)), aurocs, color=colors, alpha=0.7,
              edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(scenarios)))
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(a) Detection AUROC by Calibration', fontsize=12, fontweight='bold')
ax.set_ylim(0.98, 1.005)
ax.grid(True, alpha=0.3, axis='y')

for i, (a, d_val) in enumerate(zip(aurocs, ds)):
    ax.text(i, a + 0.001, f'd={d_val:.1f}', ha='center', fontsize=7, fontweight='bold')

# Panel (b): Per-category scores under highway calibration
ax = axes[1]
hw_data = data['scenarios']['highway_calibrated']['per_category']
cats = ['highway', 'urban', 'noise', 'indoor', 'twilight', 'snow']
cat_labels = ['Highway\n(ID)', 'Urban\n(ID)', 'Noise\n(OOD)', 'Indoor\n(OOD)', 'Twilight\n(OOD)', 'Snow\n(OOD)']
means = [hw_data[c]['mean'] for c in cats]
stds = [hw_data[c]['std'] for c in cats]
cat_colors = ['#4CAF50', '#FF9800', '#F44336', '#F44336', '#F44336', '#F44336']

bars = ax.bar(range(len(cats)), means, yerr=stds, color=cat_colors, alpha=0.7,
              edgecolor='black', linewidth=0.5, capsize=3)
ax.set_xticks(range(len(cats)))
ax.set_xticklabels(cat_labels, fontsize=8)
ax.set_ylabel('Cosine Distance', fontsize=11)
ax.set_title('(b) Highway-Calibrated Scores', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Annotate the urban/snow overlap
ax.annotate('Urban~Snow!', xy=(1.5, 0.29), fontsize=9, fontweight='bold',
            color='red', ha='center')

# Panel (c): Centroid distances
ax = axes[2]
inter_dist = data['inter_domain_distance']
ood_cats = ['noise', 'indoor', 'twilight', 'snow']
hw_ood_dists = [hw_data[c]['mean'] for c in ood_cats]
urb_data = data['scenarios']['urban_calibrated']['per_category']
urb_ood_dists = [urb_data[c]['mean'] for c in ood_cats]

bar_labels = ['Hwy<->Urb\nCentroid', 'Avg OOD\n(Hwy cal)', 'Avg OOD\n(Urb cal)', 'Avg OOD\n(Mixed cal)']
mix_data = data['scenarios']['mixed_calibrated']['per_category']
mix_ood_dists = [mix_data[c]['mean'] for c in ood_cats]
bar_vals = [inter_dist, np.mean(hw_ood_dists), np.mean(urb_ood_dists), np.mean(mix_ood_dists)]
bar_colors = ['#9C27B0', '#2196F3', '#FF9800', '#4CAF50']

ax.bar(range(len(bar_labels)), bar_vals, color=bar_colors, alpha=0.7,
       edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(bar_labels)))
ax.set_xticklabels(bar_labels, fontsize=8)
ax.set_ylabel('Cosine Distance', fontsize=11)
ax.set_title('(c) Distance Comparison', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

for i, v in enumerate(bar_vals):
    ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig92_cross_domain.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig92_cross_domain.pdf', dpi=200, bbox_inches='tight')
print("Saved fig92_cross_domain.png/pdf")
