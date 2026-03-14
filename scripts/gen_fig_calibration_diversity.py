"""Generate Figure 77: Calibration Set Diversity Analysis."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/calibration_diversity_20260314_204328.json") as f:
    data = json.load(f)

results = data['results']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel (a): Cohen's d comparison across calibration configs
ax = axes[0]
configs = ['highway_5', 'urban_5', 'mixed_5', 'highway_only', 'urban_only', 'mixed_15', 'mixed_30']
labels = ['Hwy\n(5)', 'Urb\n(5)', 'Mix\n(5)', 'Hwy\n(15)', 'Urb\n(15)', 'Mix\n(15)', 'Mix\n(30)']
ds = [results[c]['cohens_d'] for c in configs]
colors = ['#FF9800', '#2196F3', '#4CAF50', '#FF9800', '#2196F3', '#4CAF50', '#4CAF50']
alphas = [0.6, 0.6, 0.6, 0.9, 0.9, 0.9, 0.9]

bars = ax.bar(range(len(configs)), ds, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
for i, (bar, a) in enumerate(zip(bars, alphas)):
    bar.set_alpha(a)
ax.set_xticks(range(len(configs)))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Cohen's d", fontsize=11)
ax.set_title("(a) Effect Size by Calibration Strategy", fontsize=12, fontweight='bold')
ax.axhline(y=3.0, color='gray', linestyle='--', alpha=0.5, label='d=3.0 threshold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, d in enumerate(ds):
    ax.text(i, d + 0.1, f'{d:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# Panel (b): ID mean distance — mixed calibration reduces ID distance
ax = axes[1]
id_means = [results[c]['id_mean'] for c in configs]
ood_means = [results[c]['ood_mean'] for c in configs]

x = np.arange(len(configs))
width = 0.35
bars1 = ax.bar(x - width/2, id_means, width, label='ID mean', color='#4CAF50', alpha=0.7, edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, ood_means, width, label='OOD mean', color='#F44336', alpha=0.7, edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('Cosine Distance to Centroid', fontsize=11)
ax.set_title('(b) ID vs OOD Score Distributions', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Panel (c): Centroid distances between calibration strategies
ax = axes[2]
cdists = data['centroid_distances']
pairs = list(cdists.keys())
pair_labels = ['Hwy vs\nUrb', 'Hwy vs\nMixed', 'Mixed vs\nUrb']
vals = [cdists[p] for p in pairs]
colors_c = ['#F44336', '#FF9800', '#FF9800']
ax.bar(range(len(pairs)), vals, color=colors_c, alpha=0.7, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(pairs)))
ax.set_xticklabels(pair_labels, fontsize=10)
ax.set_ylabel('Cosine Distance', fontsize=11)
ax.set_title('(c) Centroid Divergence', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

for i, v in enumerate(vals):
    ax.text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig77_calibration_diversity.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig77_calibration_diversity.pdf', dpi=200, bbox_inches='tight')
print("Saved fig77_calibration_diversity.png/pdf")
