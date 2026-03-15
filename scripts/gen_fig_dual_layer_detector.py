"""Generate Figure 126: Dual-Layer OOD Detector."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/dual_layer_detector_20260315_003047.json") as f:
    data = json.load(f)

detection = data['detection_results']
per_cat = data['per_category_auroc']
cat_scores = data['per_category_scores']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel A: Method comparison
ax = axes[0]
methods = list(detection.keys())
aurocs = [detection[m]['auroc'] for m in methods]
ds = [detection[m]['d_prime'] for m in methods]
colors = ['#4CAF50' if a == 1.0 else '#FF9800' if a > 0.98 else '#F44336' for a in aurocs]
bars = ax.bar(range(len(methods)), aurocs, color=colors, alpha=0.8)
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(['L3 Only', 'L32 Only', 'Max(L3,L32)', 'Mean(L3,L32)'], fontsize=9)
ax.set_ylabel("AUROC")
ax.set_title("(A) Detection Method Comparison")
ax.set_ylim(0.97, 1.005)
ax.grid(True, alpha=0.3, axis='y')
for bar, a, d in zip(bars, aurocs, ds):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f"AUROC={a:.4f}\nd={d:.1f}", ha='center', fontsize=8)

# Panel B: L3 vs L32 scores per category
ax = axes[1]
cats = list(cat_scores.keys())
l3_scores = [cat_scores[c]['L3_mean'] for c in cats]
l32_scores = [cat_scores[c]['L32_mean'] for c in cats]
groups = [cat_scores[c]['group'] for c in cats]

x = np.arange(len(cats))
width = 0.35
colors_id = ['#4CAF50' if g == 'ID' else '#F44336' for g in groups]
bars1 = ax.bar(x - width/2, l3_scores, width, color='#2196F3', alpha=0.7, label='L3 score')
bars2 = ax.bar(x + width/2, l32_scores, width, color='#FF9800', alpha=0.7, label='L32 score')

ax.set_xticks(x)
ax.set_xticklabels([c.replace('_', '\n') for c in cats], fontsize=7)
ax.set_ylabel("Cosine Distance")
ax.set_title("(B) Per-Category L3 vs L32 Scores")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# Panel C: Per-category AUROC for max detector vs L3
ax = axes[2]
ood_cats = list(per_cat.keys())
max_aurocs = [per_cat[c]['auroc'] for c in ood_cats]
# Colors by difficulty
bar_colors = ['#4CAF50' if a == 1.0 else '#FF9800' if a > 0.95 else '#F44336' for a in max_aurocs]
bars = ax.barh(range(len(ood_cats)), max_aurocs, color=bar_colors, alpha=0.8)
ax.axvline(x=1.0, color='green', linestyle='--', alpha=0.5)
ax.set_yticks(range(len(ood_cats)))
ax.set_yticklabels([c.replace('_', ' ').title() for c in ood_cats], fontsize=9)
ax.set_xlabel("AUROC")
ax.set_title("(C) Per-Category AUROC (L3 Detector)")
ax.set_xlim(0.9, 1.01)
ax.grid(True, alpha=0.3, axis='x')
for bar, a in zip(bars, max_aurocs):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
            f"{a:.3f}", va='center', fontsize=9)
ax.invert_yaxis()

plt.suptitle("Dual-Layer OOD Detector (Exp 140)\nL3 alone achieves AUROC=1.000 (d=128.2) across ALL categories including fog",
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/latex/figures/fig126_dual_layer_detector.png",
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig126")
