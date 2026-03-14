"""Generate Figure 83: Action Dimension Analysis."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/action_dims_20260314_210520.json") as f:
    data = json.load(f)

dim_analysis = data['dim_analysis']
dims = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel (a): Per-dimension AUROC (deviation-based)
ax = axes[0]
aurocs = [dim_analysis[d]['deviation_auroc'] for d in dims]
colors = ['#4CAF50' if a > 0.9 else '#2196F3' if a > 0.7 else '#FF9800' if a > 0.5 else '#F44336' for a in aurocs]
bars = ax.bar(range(len(dims)), aurocs, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(dims)))
ax.set_xticklabels(dims, fontsize=10)
ax.set_ylabel('AUROC (deviation from ID mean)', fontsize=11)
ax.set_title('(a) Per-Dimension OOD Sensitivity', fontsize=12, fontweight='bold')
ax.axhline(y=0.5, color='black', linestyle=':', alpha=0.3)
ax.set_ylim(0.2, 1.05)
ax.grid(True, alpha=0.3, axis='y')

for i, a in enumerate(aurocs):
    ax.text(i, a + 0.02, f'{a:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Panel (b): ID vs OOD token values per dimension
ax = axes[1]
id_means = [dim_analysis[d]['id_token_mean'] for d in dims]
ood_means = [dim_analysis[d]['ood_token_mean'] for d in dims]
id_stds = [dim_analysis[d]['id_token_std'] for d in dims]
ood_stds = [dim_analysis[d]['ood_token_std'] for d in dims]

# Normalize to relative scale (subtract overall mean)
overall_mean = np.mean(id_means + ood_means)
id_rel = [m - overall_mean for m in id_means]
ood_rel = [m - overall_mean for m in ood_means]

x = np.arange(len(dims))
width = 0.35
ax.bar(x - width/2, id_stds, width, label='ID std', color='#4CAF50', alpha=0.7, edgecolor='black', linewidth=0.5)
ax.bar(x + width/2, ood_stds, width, label='OOD std', color='#F44336', alpha=0.7, edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(dims, fontsize=10)
ax.set_ylabel('Token Std Dev', fontsize=11)
ax.set_title('(b) Token Variability by Dimension', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Panel (c): Variance ratio (OOD/ID) per dimension
ax = axes[2]
var_ratios = [min(dim_analysis[d]['var_ratio'], 100) for d in dims]  # Cap for display
colors_vr = ['#F44336' if v > 5 else '#FF9800' if v > 2 else '#4CAF50' for v in var_ratios]
ax.bar(range(len(dims)), var_ratios, color=colors_vr, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(dims)))
ax.set_xticklabels(dims, fontsize=10)
ax.set_ylabel('Variance Ratio (OOD/ID)', fontsize=11)
ax.set_title('(c) OOD Variance Amplification', fontsize=12, fontweight='bold')
ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='No amplification')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Special annotation for x (very high ratio)
if var_ratios[0] >= 100:
    ax.text(0, var_ratios[0] + 2, f'∞*', ha='center', va='bottom', fontsize=10, fontweight='bold', color='darkred')
    ax.text(0, var_ratios[0] - 10, '(ID σ≈0)', ha='center', va='top', fontsize=7, color='darkred')

for i, v in enumerate(var_ratios):
    if v < 90:
        ax.text(i, v + 1, f'{v:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig83_action_dims.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig83_action_dims.pdf', dpi=200, bbox_inches='tight')
print("Saved fig83_action_dims.png/pdf")
