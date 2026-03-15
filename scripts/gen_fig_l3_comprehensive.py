"""Generate Figure 128: L3 Comprehensive OOD Detection."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/l3_comprehensive_20260315_003839.json") as f:
    data = json.load(f)

results = data['per_category']
# Sort by d-prime
sorted_cats = sorted(results.items(), key=lambda x: x[1]['d_prime'], reverse=True)
cat_names = [c[0] for c in sorted_cats]
ds = [c[1]['d_prime'] for c in sorted_cats]
aurocs = [c[1]['auroc'] for c in sorted_cats]
gaps = [c[1]['gap'] for c in sorted_cats]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel A: D-prime ranking
ax = axes[0]
colors = ['#4CAF50' if d > 100 else '#8BC34A' if d > 50 else '#FF9800' if d > 15 else '#F44336' for d in ds]
bars = ax.barh(range(len(cat_names)), ds, color=colors, alpha=0.8)
ax.set_yticks(range(len(cat_names)))
ax.set_yticklabels([c.replace('_', ' ').title() for c in cat_names], fontsize=8)
ax.set_xlabel("D-prime (σ)")
ax.set_title("(A) L3 Detection Strength")
ax.grid(True, alpha=0.3, axis='x')
for bar, d in zip(bars, ds):
    ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
            f"d={d:.0f}", va='center', fontsize=7)
ax.invert_yaxis()

# Panel B: AUROC by category
ax = axes[1]
auroc_colors = ['#4CAF50' if a == 1.0 else '#FF9800' if a > 0.95 else '#F44336' for a in aurocs]
bars = ax.barh(range(len(cat_names)), aurocs, color=auroc_colors, alpha=0.8)
ax.axvline(x=1.0, color='green', linestyle='--', alpha=0.5)
ax.axvline(x=0.95, color='orange', linestyle=':', alpha=0.3)
ax.set_yticks(range(len(cat_names)))
ax.set_yticklabels([c.replace('_', ' ').title() for c in cat_names], fontsize=8)
ax.set_xlabel("AUROC")
ax.set_title("(B) Per-Category AUROC")
ax.set_xlim(0.95, 1.005)
ax.grid(True, alpha=0.3, axis='x')
for bar, a in zip(bars, aurocs):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
            f"{a:.4f}", va='center', fontsize=7)
ax.invert_yaxis()

# Panel C: Category families
ax = axes[2]
families = {
    'Weather': ['fog_30', 'fog_50', 'fog_70', 'rain', 'snow'],
    'Lighting': ['twilight', 'night'],
    'Domain': ['indoor', 'underwater', 'forest', 'desert', 'construction'],
    'Extreme': ['noise'],
}
family_ds = {}
for family, members in families.items():
    member_ds = [results[m]['d_prime'] for m in members if m in results]
    family_ds[family] = (np.mean(member_ds), np.std(member_ds), member_ds)

family_names = list(family_ds.keys())
family_means = [family_ds[f][0] for f in family_names]
family_stds = [family_ds[f][1] for f in family_names]
fam_colors = ['#2196F3', '#FF9800', '#4CAF50', '#F44336']

bars = ax.bar(range(len(family_names)), family_means, yerr=family_stds,
              color=fam_colors, alpha=0.8, capsize=5)
ax.set_xticks(range(len(family_names)))
ax.set_xticklabels(family_names, fontsize=10)
ax.set_ylabel("Mean D-prime (σ)")
ax.set_title("(C) Detection by Category Family")
ax.grid(True, alpha=0.3, axis='y')
for bar, m, s in zip(bars, family_means, family_stds):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 3,
            f"{m:.0f}", ha='center', fontsize=10)

plt.suptitle(f"Layer 3 Comprehensive OOD Detection (Exp 142)\n13/13 categories detected; overall AUROC={data['overall']['auroc']:.4f}, d={data['overall']['d_prime']:.1f}; fog_30% only gap < 0",
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/latex/figures/fig128_l3_comprehensive.png",
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig128")
