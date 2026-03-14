"""Generate Figure 99: Calibration Curve — Fine-Grained Size Analysis."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/calibration_curve_20260314_220822.json") as f:
    data = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

curve = data['calibration_curve']
ns = sorted([int(k) for k in curve.keys()])
aurocs = [curve[str(n)]['auroc'] for n in ns]
ds = [curve[str(n)]['d'] for n in ns]
centroid_sims = [curve[str(n)]['centroid_cosine_sim'] for n in ns]

# Panel (a): AUROC vs calibration size
ax = axes[0]
ax.plot(ns, aurocs, 'o-', color='#2196F3', linewidth=2, markersize=4)
ax.axhline(y=1.0, color='green', linestyle='--', linewidth=1, alpha=0.5)
ax.axhline(y=0.99, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='AUROC=0.99')
ax.set_xlabel('Calibration Set Size', fontsize=11)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(a) AUROC vs Calibration Size', fontsize=12, fontweight='bold')
ax.set_ylim(0.97, 1.005)
ax.grid(True, alpha=0.3)

# Add bootstrap error bars at key sizes
boot = data['bootstrap']
for n_str, b in boot.items():
    n = int(n_str)
    ax.errorbar(n, b['auroc_mean'], yerr=b['auroc_std'],
                fmt='s', color='red', markersize=8, capsize=4, capthick=2, zorder=5)
ax.legend(['Deterministic', 'AUROC=0.99 threshold', 'Bootstrap mean±std'], fontsize=8)

# Panel (b): Cohen's d vs calibration size
ax = axes[1]
ax.plot(ns, ds, 'o-', color='#F44336', linewidth=2, markersize=4)
ax.set_xlabel('Calibration Set Size', fontsize=11)
ax.set_ylabel("Cohen's d", fontsize=11)
ax.set_title("(b) Cohen's d vs Calibration Size", fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Highlight convergence
ax.axhline(y=np.mean(ds[-10:]), color='green', linestyle='--', linewidth=1.5,
           alpha=0.7, label=f'Asymptote d={np.mean(ds[-10:]):.1f}')
ax.legend(fontsize=9)

# Panel (c): Centroid stability
ax = axes[2]
ax.plot(ns, centroid_sims, 'o-', color='#9C27B0', linewidth=2, markersize=4)
ax.set_xlabel('Calibration Set Size', fontsize=11)
ax.set_ylabel('Cosine Similarity to Full Centroid', fontsize=11)
ax.set_title('(c) Centroid Stability', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_ylim(0.94, 0.965)

# Mark key thresholds
ax.axhline(y=centroid_sims[-1], color='green', linestyle='--', linewidth=1.5,
           alpha=0.7, label=f'n=50 sim={centroid_sims[-1]:.4f}')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig99_calibration_curve.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig99_calibration_curve.pdf', dpi=200, bbox_inches='tight')
print("Saved fig99_calibration_curve.png/pdf")
