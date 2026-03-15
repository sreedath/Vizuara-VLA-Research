import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/recon_ood_20260315_012535.json") as f:
    data = json.load(f)

results = data["results"]
cats = data["ood_categories"]
k_values = data["k_values"]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel A: AUROC vs k for both layers
ax = axes[0]
for ln, color, marker in [("L3", "#2196F3", "o"), ("L32", "#FF9800", "s")]:
    aurocs = [results[ln][f"recon_k{k}"]["overall_auroc"] for k in k_values]
    ax.plot(k_values, aurocs, f'{marker}-', color=color, label=f'{ln} Recon', markersize=8, linewidth=2)
    # Add cosine baseline
    cos_auroc = results[ln]["cosine"]["overall_auroc"]
    ax.axhline(y=cos_auroc, color=color, linestyle='--', alpha=0.5, label=f'{ln} Cosine ({cos_auroc:.3f})')

ax.set_xlabel("PCA Components (k)")
ax.set_ylabel("Overall AUROC")
ax.set_title("(A) Reconstruction Error vs Cosine Distance")
ax.legend(fontsize=8)
ax.set_ylim(0.94, 1.01)

# Panel B: Per-category AUROC at k=2 vs cosine
ax = axes[1]
x = np.arange(len(cats))
w = 0.2
configs = [
    ("L3", "cosine", "L3 Cosine", "#90CAF9"),
    ("L3", "recon_k2", "L3 Recon k=2", "#2196F3"),
    ("L32", "cosine", "L32 Cosine", "#FFCC80"),
    ("L32", "recon_k2", "L32 Recon k=2", "#FF9800"),
]
for i, (ln, metric, label, color) in enumerate(configs):
    aurocs = [results[ln][metric]["per_category"][c]["auroc"] for c in cats]
    ax.bar(x + (i-1.5)*w, aurocs, w, label=label, color=color, alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels([c.replace("_", "\n") for c in cats], fontsize=8, rotation=45, ha='right')
ax.set_ylabel("AUROC")
ax.set_ylim(0.7, 1.05)
ax.legend(fontsize=7, ncol=2, loc='lower left')
ax.set_title("(B) Per-Category: Recon k=2 vs Cosine")

# Panel C: d-prime comparison
ax = axes[2]
methods = ["cosine"] + [f"recon_k{k}" for k in k_values]
method_labels = ["Cosine"] + [f"Recon k={k}" for k in k_values]
x = np.arange(len(methods))
w = 0.35
for ln, color, offset in [("L3", "#2196F3", -w/2), ("L32", "#FF9800", w/2)]:
    dps = [results[ln][m]["overall_d_prime"] for m in methods]
    ax.bar(x + offset, dps, w, label=ln, color=color, alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(method_labels, fontsize=8, rotation=45, ha='right')
ax.set_ylabel("d-prime")
ax.legend()
ax.set_title("(C) d-prime: Reconstruction vs Cosine")

plt.tight_layout()
plt.savefig("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/latex/figures/fig136_recon_ood.png", dpi=150, bbox_inches='tight')
print("Saved fig136")
