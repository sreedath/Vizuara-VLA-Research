import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/cal_diversity_20260315_013659.json") as f:
    data = json.load(f)

results = data["results"]
configs = data["cal_configs"]
cats = data["ood_categories"]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel A: Overall AUROC by config
ax = axes[0]
config_labels = ["Highway\nOnly", "Urban\nOnly", "Rural\nOnly", "Diverse\n(3 each)", "Partial\n(6hw+3ur)"]
colors = ["#90CAF9", "#FFCC80", "#A5D6A7", "#4CAF50", "#FF9800"]

x = np.arange(len(configs))
w = 0.35
for i, (ln, color_base) in enumerate([("L3", "#2196F3"), ("L32", "#FF9800")]):
    aurocs = [results[c][ln]["overall_auroc"] for c in configs]
    bars = ax.bar(x + (i-0.5)*w, aurocs, w, label=ln, color=color_base, alpha=0.7 + 0.15*i)
    for bar, val in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha='center', fontsize=7, rotation=45)

ax.set_xticks(x)
ax.set_xticklabels(config_labels, fontsize=8)
ax.set_ylabel("Overall AUROC")
ax.set_ylim(0.7, 1.05)
ax.legend()
ax.set_title("(A) Calibration Diversity Impact on AUROC")
ax.axhline(y=0.95, color='gray', linestyle='--', alpha=0.3)

# Panel B: Per-category AUROC for diverse vs best homogeneous (L3)
ax = axes[1]
x = np.arange(len(cats))
w = 0.25
for i, (cfg, label, color) in enumerate([("highway_only", "Highway Only", "#90CAF9"),
                                           ("diverse_3each", "Diverse (3 each)", "#4CAF50")]):
    aurocs = [results[cfg]["L3"]["per_category"][c] for c in cats]
    ax.bar(x + (i-0.5)*w, aurocs, w, label=label, color=color, alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels([c.replace("_", "\n") for c in cats], fontsize=8, rotation=45, ha='right')
ax.set_ylabel("AUROC (L3)")
ax.set_ylim(0.5, 1.05)
ax.legend(fontsize=9)
ax.set_title("(B) Per-Category: Diverse vs Homogeneous (L3)")

# Panel C: d-prime comparison
ax = axes[2]
for ln, color in [("L3", "#2196F3"), ("L32", "#FF9800")]:
    dps = [results[c][ln]["d_prime"] for c in configs]
    ax.plot(range(len(configs)), dps, 'o-', color=color, label=ln, markersize=8, linewidth=2)

ax.set_xticks(range(len(configs)))
ax.set_xticklabels(config_labels, fontsize=8)
ax.set_ylabel("d-prime")
ax.legend()
ax.set_title("(C) d-prime: Diversity Boosts Separation")

plt.tight_layout()
plt.savefig("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/latex/figures/fig138_cal_diversity.png", dpi=150, bbox_inches='tight')
print("Saved fig138")
