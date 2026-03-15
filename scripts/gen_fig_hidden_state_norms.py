"""Generate Figure 123: Hidden State Norm Analysis."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/hidden_state_norms_20260315_002202.json") as f:
    data = json.load(f)

results = data['results']
layer_indices = data['layer_indices']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel A: Norm by layer for ID vs OOD
ax = axes[0]
id_means = [results[str(l)]['id_mean'] for l in layer_indices]
id_stds = [results[str(l)]['id_std'] for l in layer_indices]
ood_means = [results[str(l)]['ood_mean'] for l in layer_indices]
ood_stds = [results[str(l)]['ood_std'] for l in layer_indices]

ax.errorbar(layer_indices, id_means, yerr=id_stds, fmt='o-', color='#4CAF50',
            linewidth=2, markersize=8, capsize=4, label='ID')
ax.errorbar(layer_indices, ood_means, yerr=ood_stds, fmt='s-', color='#F44336',
            linewidth=2, markersize=8, capsize=4, label='OOD')
ax.set_xlabel("Layer Index")
ax.set_ylabel("L2 Norm")
ax.set_title("(A) Hidden State Norms by Layer")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel B: AUROC by layer (norm-based detection)
ax = axes[1]
aurocs = [results[str(l)]['auroc'] for l in layer_indices]
ds = [results[str(l)]['d_prime'] for l in layer_indices]
colors = ['#4CAF50' if a > 0.9 else '#FF9800' if a > 0.7 else '#F44336' for a in aurocs]
bars = ax.bar(range(len(layer_indices)), aurocs, color=colors, alpha=0.8)
ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.3, label='Perfect (cosine)')
ax.set_xticks(range(len(layer_indices)))
ax.set_xticklabels([f"L{l}" for l in layer_indices], fontsize=9)
ax.set_ylabel("AUROC")
ax.set_title("(B) Norm-Based Detection by Layer")
ax.set_ylim(0.4, 1.05)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')
for bar, a, d in zip(bars, aurocs, ds):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{a:.3f}\nd={d:.1f}", ha='center', fontsize=7)

# Panel C: Comparison of norm-based methods vs cosine
ax = axes[2]
methods = ['Single Layer\n(L32 norm)', 'Multi-Layer\nNorm Vector', 'L3/L32\nNorm Ratio',
           'Cosine Distance\n(baseline)']
method_aurocs = [
    results['32']['auroc'],
    results['multi_layer']['auroc'],
    results.get('norm_ratio_3_32', {}).get('auroc', 0.5),
    1.0
]
method_ds = [
    results['32']['d_prime'],
    results['multi_layer']['d_prime'],
    results.get('norm_ratio_3_32', {}).get('d_prime', 0),
    52.0  # from Exp 132
]
colors_comp = ['#FF9800', '#2196F3', '#9C27B0', '#4CAF50']
bars = ax.bar(range(len(methods)), method_aurocs, color=colors_comp, alpha=0.8)
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods, fontsize=8)
ax.set_ylabel("AUROC")
ax.set_title("(C) Norm vs Cosine Detection")
ax.set_ylim(0.8, 1.05)
ax.grid(True, alpha=0.3, axis='y')
for bar, a, d in zip(bars, method_aurocs, method_ds):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"AUROC={a:.3f}\nd={d:.1f}", ha='center', fontsize=8)

plt.suptitle("Hidden State Norm Analysis (Exp 137)\nLayer 32 norm achieves AUROC=0.963 but cosine distance (AUROC=1.000) remains superior",
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/latex/figures/fig123_hidden_state_norms.png",
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig123")
