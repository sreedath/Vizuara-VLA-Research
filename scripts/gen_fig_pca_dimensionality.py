import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/pca_dimensionality_20260315_011123.json") as f:
    data = json.load(f)

results = data["results"]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel A: Singular value spectrum
ax = axes[0]
for ln, color, marker in [("L3", "#2196F3", "o"), ("L32", "#FF9800", "s")]:
    svs = results[ln]["singular_values_top20"]
    # Normalize to fraction of total
    total = sum(s**2 for s in svs)
    fracs = [(s**2)/total for s in svs]
    ax.semilogy(range(1, len(fracs)+1), fracs, f'{marker}-', color=color, label=f'{ln}', markersize=5, linewidth=2)

ax.set_xlabel("PCA Component")
ax.set_ylabel("Fraction of Variance (log scale)")
ax.set_title("(A) Singular Value Spectrum")
ax.legend()
ax.set_xlim(0.5, 20.5)

# Panel B: Cumulative explained variance
ax = axes[1]
for ln, color, marker in [("L3", "#2196F3", "o"), ("L32", "#FF9800", "s")]:
    cum = results[ln]["explained_ratio_cumulative"]
    ax.plot(range(1, len(cum)+1), cum, f'{marker}-', color=color, label=f'{ln}', markersize=5, linewidth=2)
ax.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5, label='95%')
ax.axhline(y=0.99, color='gray', linestyle=':', alpha=0.5, label='99%')
# Mark key points
ax.annotate(f'L3: 2 dims = 97.6%', xy=(2, results["L3"]["explained_ratio_cumulative"][1]),
            xytext=(5, 0.85), fontsize=9, arrowprops=dict(arrowstyle='->', color='#2196F3'))
ax.annotate(f'L32: 10 dims = 95.4%', xy=(10, results["L32"]["explained_ratio_cumulative"][9]),
            xytext=(13, 0.88), fontsize=9, arrowprops=dict(arrowstyle='->', color='#FF9800'))
ax.set_xlabel("Number of PCA Components")
ax.set_ylabel("Cumulative Explained Variance")
ax.set_title("(B) Cumulative Variance: L3 is 2D, L32 is 10D")
ax.legend(fontsize=8)
ax.set_xlim(0.5, 20.5)
ax.set_ylim(0.4, 1.02)

# Panel C: Reconstruction error (OOD vs ID) as function of k
ax = axes[2]
cats = data["ood_categories"]
k_values = [1, 2, 3, 5, 10, 15]
for ln, color in [("L3", "#2196F3"), ("L32", "#FF9800")]:
    # Mean OOD recon error / ID recon error ratio
    ratios = []
    for k in k_values:
        kk = f"k={k}"
        id_err = results[ln]["projection_results"][kk]["id_recon_error"]
        ood_errs = [results[ln]["projection_results"][kk]["per_category"][c]["recon_error"] for c in cats]
        mean_ood = np.mean(ood_errs)
        ratio = mean_ood / (id_err + 1e-10)
        ratios.append(ratio)
    ax.plot(k_values, ratios, 'o-', color=color, label=f'{ln} (OOD/ID recon ratio)', markersize=6, linewidth=2)

ax.set_xlabel("Number of PCA Components")
ax.set_ylabel("OOD / ID Reconstruction Error Ratio")
ax.set_title("(C) OOD Anomaly in PCA Space")
ax.legend()
ax.set_ylim(0, max(ax.get_ylim()[1], 60))

plt.tight_layout()
plt.savefig("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/latex/figures/fig133_pca_dimensionality.png", dpi=150, bbox_inches='tight')
print("Saved fig133")
