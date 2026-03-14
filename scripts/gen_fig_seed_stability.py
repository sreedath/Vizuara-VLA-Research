"""Generate Figure 115: Seed and Replication Stability."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/seed_stability_20260314_235229.json") as f:
    data = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel A: Deterministic inference
ax = axes[0]
imgs = list(data['repeat_results'].keys())
max_dists = [data['repeat_results'][i]['max_cosine_dist'] for i in imgs]
identical = [data['repeat_results'][i]['all_identical'] for i in imgs]
colors = ['#4CAF50' if i else '#F44336' for i in identical]
bars = ax.bar(range(len(imgs)), max_dists, color=colors, alpha=0.8)
ax.set_xticks(range(len(imgs)))
ax.set_xticklabels(imgs, fontsize=8)
ax.set_ylabel("Max Cosine Distance\n(across 10 repeats)")
ax.set_title("(A) Inference Determinism")
ax.set_ylim(0, max(max_dists) * 2 + 1e-8)
for bar, ident in zip(bars, identical):
    label = "Identical" if ident else "Differs"
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(max_dists)*0.1 + 1e-9,
            label, ha='center', fontsize=9, fontweight='bold',
            color='green' if ident else 'red')

# Panel B: Score stability
ax = axes[1]
scores = [data['score_stability'][i]['mean_score'] for i in imgs]
ranges = [data['score_stability'][i]['range_score'] for i in imgs]
cat_colors = ['#4CAF50', '#F44336', '#FF9800']
bars = ax.bar(range(len(imgs)), scores, color=cat_colors, alpha=0.8)
ax.errorbar(range(len(imgs)), scores, yerr=ranges, fmt='none', color='black', capsize=5, linewidth=2)
ax.set_xticks(range(len(imgs)))
ax.set_xticklabels(imgs, fontsize=8)
ax.set_ylabel("Detection Score")
ax.set_title("(B) Score Stability (10 repeats)")
ax.grid(True, alpha=0.3, axis='y')
for bar, r in zip(bars, ranges):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"range={r:.0e}", ha='center', fontsize=8)

# Panel C: Cross-seed variation
ax = axes[2]
seed_cats = list(data['seed_variation'].keys())
means = [data['seed_variation'][c]['mean'] for c in seed_cats]
stds = [data['seed_variation'][c]['std'] for c in seed_cats]
cvs = [data['seed_variation'][c]['cv'] for c in seed_cats]
ax.bar(range(len(seed_cats)), means, color=['#4CAF50', '#F44336'], alpha=0.7)
ax.errorbar(range(len(seed_cats)), means, yerr=stds, fmt='none', color='black', capsize=5)
ax.set_xticks(range(len(seed_cats)))
ax.set_xticklabels([f"{c}\n(20 seeds)" for c in seed_cats], fontsize=8)
ax.set_ylabel("Detection Score")
ax.set_title("(C) Cross-Seed Variation")
ax.grid(True, alpha=0.3, axis='y')
for i, (m, s, cv) in enumerate(zip(means, stds, cvs)):
    ax.text(i, m + s + 0.02, f"CV={cv:.3f}", ha='center', fontsize=9, fontweight='bold')

plt.suptitle("Seed & Replication Stability (Exp 129)\nInference is bitwise deterministic; cross-seed variation is minimal",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/latex/figures/fig115_seed_stability.png",
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig115")
