"""Generate Figure 109: Embedding Dimension Importance."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/dim_importance_20260314_231839.json") as f:
    data = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel A: Top-K vs Bottom-K vs Random subspace AUROC
ax = axes[0]
ks = [10, 50, 100, 256, 512, 1024, 2048]
top_d = [data['top_k_subspace'][str(k)]['d'] for k in ks]
bot_d = [data['bottom_k_subspace'][str(k)]['d'] for k in ks]
rand_d = [data['random_subspace'][str(k)]['d_mean'] for k in ks]
rand_std = [data['random_subspace'][str(k)]['d_std'] for k in ks]

ax.plot(ks, top_d, 'o-', color='#4CAF50', label='Top-K dims', linewidth=2, markersize=6)
ax.errorbar(ks, rand_d, yerr=rand_std, fmt='s-', color='#2196F3', label='Random-K dims', linewidth=2, markersize=5, capsize=3)
ax.plot(ks, bot_d, '^-', color='#FF5722', label='Bottom-K dims', linewidth=2, markersize=6)
ax.axhline(y=data['baseline']['d'], color='gray', linestyle='--', alpha=0.5, label=f"Baseline (all 4096): d={data['baseline']['d']:.1f}")
ax.set_xscale('log')
ax.set_xlabel("Number of Dimensions (K)")
ax.set_ylabel("D-prime")
ax.set_title("(A) Subspace Detection Strength")
ax.legend(fontsize=7, loc='center right')
ax.grid(True, alpha=0.3)

# Panel B: Ablation - remove top-K dims
ax = axes[1]
abl_ks = [10, 50, 100, 256, 512, 1024, 2048]
abl_d = [data['ablation_remove_top_k'][str(k)]['d'] for k in abl_ks]
ax.plot(abl_ks, abl_d, 'D-', color='#9C27B0', linewidth=2, markersize=6)
ax.axhline(y=data['baseline']['d'], color='gray', linestyle='--', alpha=0.5, label=f"Baseline: d={data['baseline']['d']:.1f}")
ax.axhline(y=3, color='red', linestyle='--', alpha=0.5, label='Detection threshold (d=3)')
ax.fill_between(abl_ks, 0, 3, alpha=0.1, color='red')
ax.set_xscale('log')
ax.set_xlabel("Dims Removed (Top-K)")
ax.set_ylabel("D-prime (remaining dims)")
ax.set_title("(B) Robustness to Dim Ablation")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 50)

# Panel C: Discriminability distribution
ax = axes[2]
disc = data['dim_discriminability']
categories = ['|d| > 1', '|d| > 3', '|d| > 5']
pcts = [disc['pct_above_1'], disc['pct_above_3'], disc['pct_above_5']]
bars = ax.bar(categories, pcts, color=['#4CAF50', '#FF9800', '#F44336'], alpha=0.7)
for bar, pct in zip(bars, pcts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{pct:.1f}%",
            ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylabel("% of 4096 Dimensions")
ax.set_title("(C) Per-Dim Discriminability")
ax.set_ylim(0, 70)
ax.grid(True, alpha=0.3, axis='y')

# Add stats annotation
stats_text = f"Max |d|: {disc['max_d']:.1f}\nMean |d|: {disc['mean_d']:.2f}\nMedian |d|: {disc['median_d']:.2f}"
ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle("Embedding Dimension Importance (Exp 123)\nOOD signal distributed across >50% of dimensions", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/latex/figures/fig109_dim_importance.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig109")
