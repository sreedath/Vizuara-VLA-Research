"""Generate Figure 110: Adversarial Evasion of OOD Detection."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/adversarial_evasion_20260314_232406.json") as f:
    data = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel A: Blend curves
ax = axes[0]
epsilons = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
colors = {'noise': '#F44336', 'indoor': '#FF9800', 'snow': '#2196F3'}

for cat_name, color in colors.items():
    means = [data['blend_results'][cat_name][str(e)]['mean'] for e in epsilons]
    stds = [data['blend_results'][cat_name][str(e)]['std'] for e in epsilons]
    ax.errorbar(epsilons, means, yerr=stds, fmt='o-', color=color, label=cat_name,
                linewidth=2, markersize=5, capsize=2)

threshold = data['detection_threshold_3sigma']
ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, label=f'Threshold ({threshold:.3f})')
ax.axhline(y=data['id_baseline']['mean'], color='green', linestyle=':', alpha=0.7, label=f'ID mean ({data["id_baseline"]["mean"]:.3f})')
ax.fill_between(epsilons, 0, threshold, alpha=0.1, color='green')
ax.set_xlabel("Blend Factor (eps)\n0=pure OOD, 1=pure ID")
ax.set_ylabel("Cosine Distance Score")
ax.set_title("(A) Pixel Blending Toward ID")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# Panel B: Natural transforms comparison
ax = axes[1]
transforms = ['original', 'blur_r2', 'blur_r5', 'blur_r10',
              'brightness_0.5', 'brightness_1.5', 'brightness_2.0',
              'contrast_0.3', 'contrast_0.1']
short_names = ['orig', 'blur2', 'blur5', 'blur10', 'dark', 'bright', 'bright2', 'low_c', 'very_low_c']

x = np.arange(len(transforms))
width = 0.25
for i, (cat_name, color) in enumerate(colors.items()):
    means = [data['transform_results'][cat_name].get(t, {}).get('mean', 0) for t in transforms]
    ax.bar(x + i*width, means, width, color=color, alpha=0.7, label=cat_name)

ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, label='Threshold')
ax.set_xticks(x + width)
ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=7)
ax.set_ylabel("Cosine Distance Score")
ax.set_title("(B) Natural Transforms")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3, axis='y')

# Panel C: Evasion difficulty summary
ax = axes[2]
# For each OOD category, find the epsilon needed to cross threshold
evasion_eps = {}
for cat_name in colors:
    for eps in epsilons:
        if data['blend_results'][cat_name][str(eps)]['mean'] < threshold:
            evasion_eps[cat_name] = eps
            break
    else:
        evasion_eps[cat_name] = 1.1  # never crosses

# Show min natural transform score / threshold ratio
nat_ratios = {}
for cat_name in colors:
    min_score = min(v['mean'] for v in data['transform_results'][cat_name].values())
    nat_ratios[cat_name] = min_score / threshold

cat_names = list(colors.keys())
x_pos = range(len(cat_names))

ax2 = ax.twinx()
bars1 = ax.bar([p - 0.15 for p in x_pos], [evasion_eps[c] for c in cat_names], 0.3,
               color=[colors[c] for c in cat_names], alpha=0.7, label='Blend eps to evade')
bars2 = ax2.bar([p + 0.15 for p in x_pos], [nat_ratios[c] for c in cat_names], 0.3,
                color=[colors[c] for c in cat_names], alpha=0.3, hatch='//', label='Min transform / threshold')

ax.set_xticks(x_pos)
ax.set_xticklabels(cat_names)
ax.set_ylabel("Blend eps needed")
ax.set_ylim(0, 1.2)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
ax2.set_ylabel("Score / Threshold ratio")
ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.3)

ax.set_title("(C) Evasion Difficulty")

# Combined legend
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='upper left')

plt.suptitle("Adversarial Evasion of OOD Detection (Exp 124)\nNo natural transform evades; blend requires 70-100% ID pixels",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/latex/figures/fig110_adversarial_evasion.png",
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig110")
