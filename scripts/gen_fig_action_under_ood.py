"""Generate Figure 112: Action Output Under OOD."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/action_under_ood_20260314_233732.json") as f:
    data = json.load(f)

dim_names = ['x_t', 'y_t', 'z_t', 'x_r', 'y_r', 'z_r', 'grip']
cats = list(data['categories'].keys())
id_center = np.array(data['id_center'])

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel A: Per-dimension mean actions by category
ax = axes[0]
x = np.arange(7)
width = 0.12
colors = {'highway': '#4CAF50', 'urban': '#8BC34A', 'noise': '#F44336',
          'indoor': '#FF9800', 'twilight': '#9C27B0', 'snow': '#2196F3'}

for i, cat in enumerate(cats):
    means = data['categories'][cat]['per_dim']['mean']
    stds = data['categories'][cat]['per_dim']['std']
    if means:
        ax.bar(x + i*width, means, width, color=colors[cat], alpha=0.8, label=cat)
        ax.errorbar(x + i*width, means, yerr=stds, fmt='none', color='black', capsize=1, linewidth=0.5)

ax.set_xticks(x + width*2.5)
ax.set_xticklabels(dim_names, fontsize=8)
ax.set_ylabel("Action Bin (0-255)")
ax.set_title("(A) Mean Action by Dimension")
ax.legend(fontsize=6, ncol=2)
ax.axhline(y=128, color='gray', linestyle='--', alpha=0.3, label='Neutral (128)')
ax.grid(True, alpha=0.3, axis='y')

# Panel B: Action divergence from ID center
ax = axes[1]
ood_cats = [c for c in cats if data['categories'][c]['group'] == 'OOD']
ood_colors = [colors[c] for c in ood_cats]

for i, cat in enumerate(ood_cats):
    means = np.array(data['categories'][cat]['per_dim']['mean'])
    diff = means - id_center
    ax.barh(np.arange(7) + i*0.18, diff, 0.18, color=colors[cat], alpha=0.8, label=cat)

ax.set_yticks(np.arange(7) + 0.27)
ax.set_yticklabels(dim_names, fontsize=8)
ax.set_xlabel("Shift from ID Center (bins)")
ax.set_title("(B) Per-Dim Divergence from ID")
ax.axvline(x=0, color='black', linewidth=0.5)
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3, axis='x')

# Panel C: Spread comparison
ax = axes[2]
spreads = [data['categories'][c]['spread'] for c in cats]
bar_colors = [colors[c] for c in cats]
bars = ax.bar(range(len(cats)), spreads, color=bar_colors, alpha=0.8)
ax.set_xticks(range(len(cats)))
ax.set_xticklabels(cats, rotation=45, ha='right', fontsize=8)
ax.set_ylabel("Intra-Category Spread (L2)")
ax.set_title("(C) Action Variability")
ax.grid(True, alpha=0.3, axis='y')

# Annotate
for bar, s in zip(bars, spreads):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{s:.0f}", ha='center', va='bottom', fontsize=8)

# Add ID/OOD labels
ax.axhline(y=np.mean([data['categories'][c]['spread'] for c in cats if data['categories'][c]['group'] == 'ID']),
           color='green', linestyle='--', alpha=0.5, label='ID mean')
ax.axhline(y=np.mean([data['categories'][c]['spread'] for c in cats if data['categories'][c]['group'] == 'OOD']),
           color='red', linestyle='--', alpha=0.5, label='OOD mean')
ax.legend(fontsize=8)

plt.suptitle("Action Output Analysis Under OOD (Exp 126)\nOOD shifts x_translation by -36 to -48 bins; OOD actions are less consistent",
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/latex/figures/fig112_action_under_ood.png",
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig112")
