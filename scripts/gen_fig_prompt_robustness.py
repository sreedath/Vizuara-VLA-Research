"""Generate Figure 101: Prompt Robustness Analysis."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/prompt_robustness_20260314_222104.json") as f:
    data = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

results = data['results']
names = list(results.keys())
short_names = ['Standard\nDriving', 'Simple\nDriving', 'Speed\nDriving', 'Stop\nDriving',
               'Generic\nRobot', 'Navigate\nRobot', 'Minimal', 'Empty\nTask',
               'Adversarial\nLong', 'Adversarial\nUnrelated']

aurocs = [results[n]['auroc'] for n in names]
ds = [results[n]['d'] for n in names]

# Panel (a): Cohen's d by prompt
ax = axes[0]
colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
bars = ax.bar(range(len(names)), ds, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(short_names, fontsize=6, rotation=45, ha='right')
ax.set_ylabel("Cohen's d", fontsize=11)
ax.set_title("(a) Detection Separation by Prompt", fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
# All AUROC=1.000, note it
ax.text(0.5, 0.95, 'All AUROC = 1.000', transform=ax.transAxes,
        ha='center', fontsize=10, fontweight='bold', color='green',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Panel (b): Per-category scores for selected prompts
ax = axes[1]
cats_order = ['highway', 'urban', 'snow', 'indoor', 'twilight', 'noise']
prompt_subset = ['driving_standard', 'minimal', 'robot_generic', 'adversarial_unrelated']
prompt_labels = ['Standard', 'Minimal', 'Robot Generic', 'Adversarial']
x = np.arange(len(cats_order))
width = 0.2
for i, (pname, plabel) in enumerate(zip(prompt_subset, prompt_labels)):
    scores = [results[pname]['per_category'][c]['mean'] for c in cats_order]
    ax.bar(x + i*width - 1.5*width, scores, width, label=plabel, alpha=0.8, edgecolor='black', linewidth=0.3)
ax.set_xticks(x)
ax.set_xticklabels(cats_order, fontsize=9)
ax.set_ylabel('Cosine Distance', fontsize=11)
ax.set_title('(b) Per-Category by Prompt', fontsize=12, fontweight='bold')
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3, axis='y')

# Panel (c): d range and centroid similarity
ax = axes[2]
# Sort by d
sorted_idx = np.argsort(ds)[::-1]
sorted_names = [short_names[i] for i in sorted_idx]
sorted_ds = [ds[i] for i in sorted_idx]

bars = ax.barh(range(len(sorted_ds)), sorted_ds,
               color=plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(sorted_ds))),
               alpha=0.8, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(sorted_ds)))
ax.set_yticklabels(sorted_names, fontsize=7)
ax.set_xlabel("Cohen's d", fontsize=11)
ax.set_title("(c) d Ranking Across Prompts", fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# Add centroid sim annotation
stats = data['centroid_sim_stats']
ax.text(0.98, 0.05, f'Centroid sim: {stats["mean"]:.3f}\n(range: {stats["min"]:.3f}–{stats["max"]:.3f})',
        transform=ax.transAxes, ha='right', fontsize=8,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig101_prompt_robustness.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig101_prompt_robustness.pdf', dpi=200, bbox_inches='tight')
print("Saved fig101_prompt_robustness.png/pdf")
