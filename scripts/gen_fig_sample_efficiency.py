"""Generate Figure 119: Sample Efficiency of OOD Detection."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/sample_efficiency_20260315_001227.json") as f:
    data = json.load(f)

results = data['results']
sizes = sorted([int(k) for k in results.keys()])

mean_aurocs = [results[str(s)]['mean_auroc'] for s in sizes]
std_aurocs = [results[str(s)]['std_auroc'] for s in sizes]
min_aurocs = [results[str(s)]['min_auroc'] for s in sizes]
mean_ds = [results[str(s)]['mean_d'] for s in sizes]
std_ds = [results[str(s)]['std_d'] for s in sizes]
min_ds = [results[str(s)]['min_d'] for s in sizes]
max_ds = [results[str(s)]['max_d'] for s in sizes]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel A: AUROC vs calibration size
ax = axes[0]
ax.plot(sizes, mean_aurocs, 'o-', color='#4CAF50', linewidth=2, markersize=8, label='Mean AUROC')
ax.fill_between(sizes, [m-s for m,s in zip(mean_aurocs, std_aurocs)],
                [min(1.0, m+s) for m,s in zip(mean_aurocs, std_aurocs)],
                alpha=0.2, color='#4CAF50')
ax.plot(sizes, min_aurocs, 's--', color='#F44336', linewidth=1.5, markersize=6, label='Worst-case AUROC')
ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
ax.axhline(y=0.95, color='red', linestyle=':', alpha=0.3, label='0.95 threshold')
ax.set_xlabel("Calibration Samples (n)")
ax.set_ylabel("AUROC")
ax.set_title("(A) Detection Quality vs Sample Size")
ax.set_ylim(0.94, 1.005)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
# Mark the transition point
ax.axvline(x=8, color='blue', linestyle='--', alpha=0.4, label='n=8: perfect')
ax.annotate('Perfect from n=8', xy=(8, 1.0), xytext=(12, 0.97),
            arrowprops=dict(arrowstyle='->', color='blue', alpha=0.6),
            fontsize=9, color='blue')

# Panel B: D-prime vs calibration size
ax = axes[1]
ax.errorbar(sizes, mean_ds, yerr=std_ds, fmt='o-', color='#2196F3', linewidth=2,
            markersize=8, capsize=4, label='Mean ± std')
ax.fill_between(sizes, min_ds, max_ds, alpha=0.15, color='#2196F3', label='Min-Max range')
ax.set_xlabel("Calibration Samples (n)")
ax.set_ylabel("D-prime (σ)")
ax.set_title("(B) Separation Strength vs Sample Size")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
# Mark: even n=1 has d>12
ax.axhline(y=12, color='red', linestyle=':', alpha=0.3)
ax.annotate('Worst single-sample: d=12.2', xy=(1, 12.2), xytext=(5, 8),
            arrowprops=dict(arrowstyle='->', color='red', alpha=0.6),
            fontsize=8, color='red')

# Panel C: Stability (std of d-prime)
ax = axes[2]
ax.plot(sizes, std_ds, 'o-', color='#9C27B0', linewidth=2, markersize=8)
ax.fill_between(sizes, 0, std_ds, alpha=0.2, color='#9C27B0')
ax.set_xlabel("Calibration Samples (n)")
ax.set_ylabel("Std of D-prime")
ax.set_title("(C) Centroid Stability")
ax.grid(True, alpha=0.3)
# Annotate key points
ax.annotate(f'n=1: σ={std_ds[0]:.1f}', xy=(1, std_ds[0]),
            xytext=(5, std_ds[0]+2), fontsize=9,
            arrowprops=dict(arrowstyle='->', color='purple', alpha=0.6))
ax.annotate(f'n=30: σ={std_ds[-1]:.1f}', xy=(30, std_ds[-1]),
            xytext=(20, std_ds[-1]+5), fontsize=9,
            arrowprops=dict(arrowstyle='->', color='purple', alpha=0.6))

plt.suptitle("Sample Efficiency of OOD Detection (Exp 133)\nn=1 gives AUROC=0.993; n≥8 gives perfect AUROC=1.000",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/latex/figures/fig119_sample_efficiency.png",
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig119")
