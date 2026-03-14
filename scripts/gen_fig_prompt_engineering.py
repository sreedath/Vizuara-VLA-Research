"""Generate Figure 81: Prompt Engineering for OOD Detection."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/prompt_engineering_20260314_205824.json") as f:
    data = json.load(f)

results = data['results']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel (a): Cohen's d per prompt (sorted)
ax = axes[0]
sorted_prompts = sorted(results.keys(), key=lambda k: results[k]['cohens_d'], reverse=True)
labels = {
    'driving_standard': 'Standard\nDriving',
    'driving_detailed': 'Detailed\nDriving',
    'safety_focused': 'Safety\nFocused',
    'scene_description': 'Scene\nDescribe',
    'minimal': 'Minimal',
    'robot_generic': 'Generic\nRobot',
    'empty_action': 'Empty\nAction',
    'adversarial': 'Adversarial',
}
ds = [results[p]['cohens_d'] for p in sorted_prompts]
prompt_labels = [labels[p] for p in sorted_prompts]
colors = ['#4CAF50' if d > 7 else '#2196F3' if d > 6 else '#FF9800' for d in ds]

bars = ax.barh(range(len(sorted_prompts)), ds, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(sorted_prompts)))
ax.set_yticklabels(prompt_labels, fontsize=9)
ax.set_xlabel("Cohen's d", fontsize=11)
ax.set_title("(a) Effect Size by Prompt", fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

for i, d in enumerate(ds):
    ax.text(d + 0.1, i, f'{d:.2f}', va='center', fontsize=9, fontweight='bold')

# Panel (b): ID/OOD means per prompt
ax = axes[1]
id_means = [results[p]['id_mean'] for p in sorted_prompts]
ood_means = [results[p]['ood_mean'] for p in sorted_prompts]

x = np.arange(len(sorted_prompts))
width = 0.35
ax.barh(x - width/2, id_means, width, label='ID mean', color='#4CAF50', alpha=0.7, edgecolor='black', linewidth=0.5)
ax.barh(x + width/2, ood_means, width, label='OOD mean', color='#F44336', alpha=0.7, edgecolor='black', linewidth=0.5)
ax.set_yticks(x)
ax.set_yticklabels(prompt_labels, fontsize=9)
ax.set_xlabel('Cosine Distance to Centroid', fontsize=11)
ax.set_title('(b) Score Distributions by Prompt', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='x')

# Panel (c): Centroid distances from standard prompt
ax = axes[2]
cdists = data['centroid_distances']
other_prompts = [k.replace('driving_standard_vs_', '') for k in sorted(cdists.keys(), key=lambda k: cdists[k])]
other_labels = [labels.get(p, p).replace('\n', ' ') for p in other_prompts]
dists = [cdists[f'driving_standard_vs_{p}'] for p in other_prompts]

colors_c = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(dists)))
ax.barh(range(len(other_prompts)), dists, color=colors_c, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(other_prompts)))
ax.set_yticklabels(other_labels, fontsize=9)
ax.set_xlabel('Cosine Distance from Standard', fontsize=11)
ax.set_title('(c) Centroid Divergence', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

for i, d in enumerate(dists):
    ax.text(d + 0.01, i, f'{d:.3f}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig81_prompt_engineering.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig81_prompt_engineering.pdf', dpi=200, bbox_inches='tight')
print("Saved fig81_prompt_engineering.png/pdf")
