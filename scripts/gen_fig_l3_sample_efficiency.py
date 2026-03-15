"""Generate Figure 129: L3 Sample Efficiency."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/l3_sample_efficiency_20260315_004309.json") as f:
    data = json.load(f)

# Also load L32 sample efficiency for comparison
with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/sample_efficiency_20260315_001227.json") as f:
    l32_data = json.load(f)

results = data['results']
l32_results = l32_data['results']

sizes_l3 = sorted([int(k) for k in results.keys()])
sizes_l32 = sorted([int(k) for k in l32_results.keys()])

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel A: AUROC comparison L3 vs L32
ax = axes[0]
l3_aurocs = [results[str(s)]['mean_auroc'] for s in sizes_l3]
l32_aurocs = [l32_results[str(s)]['mean_auroc'] for s in sizes_l32]
ax.plot(sizes_l3, l3_aurocs, 'o-', color='#4CAF50', linewidth=2, markersize=8, label='L3')
ax.plot(sizes_l32, l32_aurocs, 's--', color='#F44336', linewidth=2, markersize=7, label='L32')
ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel("Calibration Samples")
ax.set_ylabel("Mean AUROC")
ax.set_title("(A) L3 vs L32 Sample Efficiency")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.8, 1.02)

# Panel B: D-prime comparison
ax = axes[1]
l3_ds = [results[str(s)]['mean_d'] for s in sizes_l3]
l3_std_ds = [results[str(s)]['std_d'] for s in sizes_l3]
l32_ds = [l32_results[str(s)]['mean_d'] for s in sizes_l32]
l32_std_ds = [l32_results[str(s)]['std_d'] for s in sizes_l32]

ax.errorbar(sizes_l3, l3_ds, yerr=l3_std_ds, fmt='o-', color='#4CAF50', linewidth=2,
            markersize=8, capsize=4, label='L3')
ax.errorbar(sizes_l32, l32_ds, yerr=l32_std_ds, fmt='s--', color='#F44336', linewidth=2,
            markersize=7, capsize=4, label='L32')
ax.set_xlabel("Calibration Samples")
ax.set_ylabel("D-prime (σ)")
ax.set_title("(B) D-prime: L3 >> L32")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel C: Key insight summary
ax = axes[2]
metrics = ['Mean\nAUROC', 'Mean\nD-prime', 'Min\nAUROC']
l3_vals = [np.mean(l3_aurocs), np.mean(l3_ds), min([results[str(s)]['min_auroc'] for s in sizes_l3])]
l32_vals = [np.mean(l32_aurocs), np.mean(l32_ds), min([l32_results[str(s)]['min_auroc'] for s in sizes_l32])]

x = np.arange(len(metrics))
width = 0.35
bars1 = ax.bar(x - width/2, l3_vals, width, color='#4CAF50', alpha=0.8, label='L3')
bars2 = ax.bar(x + width/2, l32_vals, width, color='#F44336', alpha=0.8, label='L32')
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=9)
ax.set_title("(C) L3 vs L32 Summary")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
# Normalize d-prime for display
for bar, v in zip(bars1, l3_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{v:.1f}" if v > 2 else f"{v:.3f}", ha='center', fontsize=8)
for bar, v in zip(bars2, l32_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{v:.1f}" if v > 2 else f"{v:.3f}", ha='center', fontsize=8)

plt.suptitle("L3 vs L32 Sample Efficiency (Exp 143)\nL3 has 5× higher d-prime but lower AUROC due to fog overlap; L32 achieves AUROC=1.0 at n≥8",
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/latex/figures/fig129_l3_sample_efficiency.png",
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig129")
