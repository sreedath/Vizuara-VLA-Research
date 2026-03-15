"""Generate Figure 122: Action Token Position Analysis."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/action_token_position_20260315_001952.json") as f:
    data = json.load(f)

dim_names = data['dim_names']
dim_results = data['per_dimension']
cat_profiles = data['per_category']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel A: Per-dimension AUROC
ax = axes[0]
aurocs = [dim_results[d]['auroc'] for d in dim_names]
ds = [dim_results[d]['d_prime'] for d in dim_names]
colors = ['#4CAF50' if a > 0.8 else '#FF9800' if a > 0.6 else '#F44336' for a in aurocs]
bars = ax.bar(range(7), aurocs, color=colors, alpha=0.8)
ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random (0.5)')
ax.axhline(y=data['combined']['auroc'], color='blue', linestyle='--', alpha=0.5,
           label=f'Combined ({data["combined"]["auroc"]:.3f})')
ax.set_xticks(range(7))
ax.set_xticklabels(['x_t', 'y_t', 'z_t', 'x_r', 'y_r', 'z_r', 'grip'], fontsize=9)
ax.set_ylabel("AUROC")
ax.set_title("(A) Per-Dimension OOD Detection")
ax.set_ylim(0.3, 1.0)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')
for bar, a, d in zip(bars, aurocs, ds):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{a:.3f}\nd={d:.1f}", ha='center', fontsize=7)

# Panel B: ID vs OOD action profiles
ax = axes[1]
id_means = [dim_results[d]['id_mean'] for d in dim_names]
id_stds = [dim_results[d]['id_std'] for d in dim_names]
ood_means = [dim_results[d]['ood_mean'] for d in dim_names]
ood_stds = [dim_results[d]['ood_std'] for d in dim_names]

x = np.arange(7)
width = 0.35
bars1 = ax.bar(x - width/2, id_means, width, yerr=id_stds, color='#4CAF50',
               alpha=0.7, label='ID', capsize=3)
bars2 = ax.bar(x + width/2, ood_means, width, yerr=ood_stds, color='#F44336',
               alpha=0.7, label='OOD', capsize=3)
ax.axhline(y=128, color='gray', linestyle=':', alpha=0.5, label='Neutral (128)')
ax.set_xticks(x)
ax.set_xticklabels(['x_t', 'y_t', 'z_t', 'x_r', 'y_r', 'z_r', 'grip'], fontsize=9)
ax.set_ylabel("Action Bin Value (0-255)")
ax.set_title("(B) ID vs OOD Action Profiles")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# Panel C: Per-dimension shift
ax = axes[2]
shifts = [dim_results[d]['shift'] for d in dim_names]
colors_shift = ['#F44336' if abs(s) > 30 else '#FF9800' if abs(s) > 15 else '#4CAF50' for s in shifts]
bars = ax.bar(range(7), shifts, color=colors_shift, alpha=0.8)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_xticks(range(7))
ax.set_xticklabels(['x_t', 'y_t', 'z_t', 'x_r', 'y_r', 'z_r', 'grip'], fontsize=9)
ax.set_ylabel("OOD - ID Shift (bins)")
ax.set_title("(C) Per-Dimension OOD Shift")
ax.grid(True, alpha=0.3, axis='y')
for bar, s in zip(bars, shifts):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + (2 if s > 0 else -5),
            f"{s:+.1f}", ha='center', fontsize=8)

plt.suptitle("Action Token Position Analysis (Exp 136)\nx_translation best single dim (AUROC=0.889); combined 7-dim only AUROC=0.787 — action space is a poor OOD detector",
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/latex/figures/fig122_action_token_position.png",
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig122")
