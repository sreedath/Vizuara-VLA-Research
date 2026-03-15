import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/mahalanobis_vs_cosine_20260315_010243.json") as f:
    data = json.load(f)

results = data["results"]
cats = data["ood_categories"]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel A: Per-category AUROC comparison
ax = axes[0]
x = np.arange(len(cats))
w = 0.2
configs = [
    ("L3", "cosine", "L3 Cosine", "#2196F3"),
    ("L3", "mahalanobis", "L3 Mahalanobis", "#4CAF50"),
    ("L32", "cosine", "L32 Cosine", "#FF9800"),
    ("L32", "mahalanobis", "L32 Mahalanobis", "#F44336"),
]
for i, (layer, metric, label, color) in enumerate(configs):
    aurocs = [results[layer][metric]["per_category"][c]["auroc"] for c in cats]
    ax.bar(x + (i - 1.5) * w, aurocs, w, label=label, color=color, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([c.replace("_", "\n") for c in cats], fontsize=8, rotation=45, ha='right')
ax.set_ylabel("AUROC")
ax.set_ylim(0.5, 1.05)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
ax.legend(fontsize=8, loc='lower left')
ax.set_title("(A) Per-Category AUROC: Cosine vs Mahalanobis")

# Panel B: Per-category d-prime comparison
ax = axes[1]
for i, (layer, metric, label, color) in enumerate(configs):
    dprimes = [results[layer][metric]["per_category"][c]["d_prime"] for c in cats]
    ax.bar(x + (i - 1.5) * w, dprimes, w, label=label, color=color, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([c.replace("_", "\n") for c in cats], fontsize=8, rotation=45, ha='right')
ax.set_ylabel("d-prime")
ax.legend(fontsize=8, loc='upper right')
ax.set_title("(B) Per-Category d-prime: Cosine vs Mahalanobis")

# Panel C: Summary comparison
ax = axes[2]
summary_data = []
labels_summary = []
colors_summary = []
for layer, metric, label, color in configs:
    overall = results[layer][metric]["overall"]
    summary_data.append(overall["auroc"])
    labels_summary.append(label)
    colors_summary.append(color)

bars = ax.barh(range(len(summary_data)), summary_data, color=colors_summary, alpha=0.85)
ax.set_yticks(range(len(summary_data)))
ax.set_yticklabels(labels_summary)
ax.set_xlabel("Overall AUROC")
ax.set_xlim(0.9, 1.01)
for bar, val in zip(bars, summary_data):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va='center', fontsize=10, fontweight='bold')
ax.set_title("(C) Overall AUROC Summary")

# Add d-prime annotations
for i, (layer, metric, label, color) in enumerate(configs):
    dp = results[layer][metric]["overall"]["d_prime"]
    ax.text(0.905, i, f"d'={dp:.1f}", va='center', fontsize=9, color='white', fontweight='bold')

plt.tight_layout()
plt.savefig("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/latex/figures/fig131_mahalanobis_vs_cosine.png", dpi=150, bbox_inches='tight')
print("Saved fig131")
