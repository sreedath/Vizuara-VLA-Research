"""Generate Figure 124: OOD Category Difficulty Ranking."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/ood_category_difficulty_20260315_002440.json") as f:
    data = json.load(f)

results = data['results']
id_stats = data['id_stats']
# Sort by gap (hardest first)
sorted_cats = sorted(results.items(), key=lambda x: x[1]['gap'])
cat_names = [c[0] for c in sorted_cats]
gaps = [c[1]['gap'] for c in sorted_cats]
mean_dists = [c[1]['mean_dist'] for c in sorted_cats]
std_dists = [c[1]['std_dist'] for c in sorted_cats]
aurocs = [c[1]['auroc'] for c in sorted_cats]
ds = [c[1]['d_prime'] for c in sorted_cats]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel A: Distance distributions
ax = axes[0]
colors = {'fog': '#FF5722', 'desert': '#FFC107', 'snow': '#03A9F4', 'construction': '#FF9800',
          'indoor': '#9C27B0', 'underwater': '#00BCD4', 'rain': '#607D8B', 'noise': '#F44336',
          'twilight': '#3F51B5'}
for i, (cat_name, res) in enumerate(sorted_cats):
    color = colors.get(cat_name, '#999999')
    ax.barh(i, res['mean_dist'], xerr=res['std_dist'], color=color, alpha=0.7, capsize=3)
    ax.plot(res['min_dist'], i, 'k<', markersize=6)
ax.axvline(x=id_stats['max_dist'], color='green', linestyle='--', alpha=0.7,
           label=f'ID max ({id_stats["max_dist"]:.3f})')
ax.axvline(x=id_stats['mean_dist'], color='green', linestyle=':', alpha=0.5,
           label=f'ID mean ({id_stats["mean_dist"]:.3f})')
ax.set_yticks(range(len(cat_names)))
ax.set_yticklabels([c.title() for c in cat_names], fontsize=9)
ax.set_xlabel("Cosine Distance from ID Centroid")
ax.set_title("(A) Distance Distributions (← harder)")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3, axis='x')

# Panel B: Gap and AUROC
ax = axes[1]
bar_colors = ['#F44336' if g < 0 else '#FF9800' if g < 0.15 else '#4CAF50' for g in gaps]
bars = ax.barh(range(len(cat_names)), gaps, color=bar_colors, alpha=0.8)
ax.axvline(x=0, color='red', linewidth=1.5)
ax.set_yticks(range(len(cat_names)))
ax.set_yticklabels([c.title() for c in cat_names], fontsize=9)
ax.set_xlabel("Gap (min OOD - max ID)")
ax.set_title("(B) Detection Gap (negative = overlap)")
ax.grid(True, alpha=0.3, axis='x')
for bar, g, a in zip(bars, gaps, aurocs):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
            f"AUROC={a:.3f}", va='center', fontsize=7)

# Panel C: D-prime ranking
ax = axes[2]
bar_colors2 = ['#F44336' if d < 5 else '#FF9800' if d < 20 else '#4CAF50' for d in ds]
bars = ax.barh(range(len(cat_names)), ds, color=bar_colors2, alpha=0.8)
ax.set_yticks(range(len(cat_names)))
ax.set_yticklabels([c.title() for c in cat_names], fontsize=9)
ax.set_xlabel("D-prime (σ)")
ax.set_title("(C) Separation Strength")
ax.grid(True, alpha=0.3, axis='x')
for bar, d in zip(bars, ds):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f"d={d:.1f}", va='center', fontsize=8)

plt.suptitle("OOD Category Difficulty Ranking (Exp 138)\nFog is the ONLY category with ID overlap (AUROC=0.976); all others perfectly detected",
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/latex/figures/fig124_ood_category_difficulty.png",
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig124")
