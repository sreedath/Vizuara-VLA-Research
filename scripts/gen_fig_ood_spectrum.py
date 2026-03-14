"""Generate Figure 106: OOD Hardness Spectrum."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/ood_spectrum_20260314_230037.json") as f:
    data = json.load(f)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

pc = data['per_category']

# Sort by mean score
cats = sorted(pc.keys(), key=lambda c: pc[c]['mean_score'])
scores = [pc[c]['mean_score'] for c in cats]
stds = [pc[c]['std_score'] for c in cats]
groups = [pc[c]['group'] for c in cats]

# Color coding
colors = []
for c, g in zip(cats, groups):
    if g == 'ID':
        colors.append('#4CAF50')
    elif c in ['fog_highway', 'snow', 'construction']:
        colors.append('#FF9800')  # near-ID OOD
    else:
        colors.append('#F44336')  # clear OOD

# Panel (a): Horizontal bar chart sorted by score
ax = axes[0]
y_pos = range(len(cats))
bars = ax.barh(y_pos, scores, xerr=stds, color=colors, alpha=0.8,
               edgecolor='black', linewidth=0.5, capsize=3)
ax.set_yticks(y_pos)
ax.set_yticklabels([c.replace('_', '\n') for c in cats], fontsize=8)
ax.set_xlabel('Cosine Distance from ID Centroid', fontsize=11)
ax.set_title('(a) OOD Hardness Spectrum', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# Add ID/OOD labels
for i, (c, g) in enumerate(zip(cats, groups)):
    label = 'ID' if g == 'ID' else 'OOD'
    ax.text(scores[i] + stds[i] + 0.01, i, label, va='center', fontsize=7,
            fontweight='bold', color=colors[i])

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#4CAF50', alpha=0.8, label='ID'),
    Patch(facecolor='#FF9800', alpha=0.8, label='Near-ID OOD'),
    Patch(facecolor='#F44336', alpha=0.8, label='Clear OOD'),
]
ax.legend(handles=legend_elements, fontsize=8, loc='lower right')

# Panel (b): Score distribution with ID zone
ax = axes[1]
id_max = max(pc[c]['max_score'] for c in pc if pc[c]['group'] == 'ID')
id_mean = np.mean([pc[c]['mean_score'] for c in pc if pc[c]['group'] == 'ID'])

ood_cats = [c for c in cats if pc[c]['group'] == 'OOD']
ood_scores = [pc[c]['mean_score'] for c in ood_cats]
ood_stds = [pc[c]['std_score'] for c in ood_cats]

x = np.arange(len(ood_cats))
ood_colors = ['#FF9800' if c in ['fog_highway', 'snow', 'construction'] else '#F44336' for c in ood_cats]
bars2 = ax.bar(x, ood_scores, yerr=ood_stds, color=ood_colors, alpha=0.8,
               edgecolor='black', linewidth=0.5, capsize=3)
ax.axhspan(0, id_max + 0.02, alpha=0.15, color='green', label=f'ID zone (max={id_max:.3f})')
ax.axhline(y=id_mean, color='green', linestyle='--', linewidth=2,
           label=f'ID mean={id_mean:.3f}')
ax.set_xticks(x)
ax.set_xticklabels([c.replace('_', '\n') for c in ood_cats], fontsize=7, rotation=45, ha='right')
ax.set_ylabel('Cosine Distance', fontsize=11)
ax.set_title('(b) OOD Categories vs ID Zone', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig106_ood_spectrum.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig106_ood_spectrum.pdf', dpi=200, bbox_inches='tight')
print("Saved fig106_ood_spectrum.png/pdf")
