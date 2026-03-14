"""Generate Figure 66: Temporal Stability of Calibration."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

# Data from experiment 80
drifts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
aurocs = [1.0]*11
cohens_d = [10.71, 10.64, 10.60, 10.40, 10.12, 9.73, 9.25, 8.54, 7.63, 6.54, 5.50]
id_dists = [0.087, 0.089, 0.091, 0.097, 0.105, 0.114, 0.128, 0.145, 0.172, 0.209, 0.242]
ood_dist = 0.408

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel (a): AUROC vs drift
ax = axes[0]
ax.plot(drifts, aurocs, 'go-', linewidth=2, markersize=10)
ax.fill_between(drifts, 0.95, 1.05, alpha=0.1, color='green')
ax.set_xlabel('Drift Level', fontsize=11)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(a) AUROC vs Temporal Drift', fontsize=12, fontweight='bold')
ax.set_ylim(0.9, 1.05)
ax.set_xlim(-0.05, 1.05)
ax.grid(True, alpha=0.3)
ax.annotate('Perfect AUROC at\nALL drift levels!', xy=(0.5, 1.0), xytext=(0.5, 0.94),
            fontsize=11, color='darkgreen', fontweight='bold', ha='center',
            arrowprops=dict(arrowstyle='->', color='darkgreen'))

# Panel (b): Cohen's d vs drift
ax = axes[1]
colors_d = plt.cm.RdYlGn_r(np.linspace(0, 0.7, len(drifts)))
ax.bar(range(len(drifts)), cohens_d, 0.7, color=colors_d,
       edgecolor='black', linewidth=0.5, alpha=0.85)
ax.set_xticks(range(len(drifts)))
ax.set_xticklabels([f'{d:.1f}' for d in drifts], fontsize=9)
ax.set_xlabel('Drift Level', fontsize=11)
ax.set_ylabel("Cohen's d", fontsize=11)
ax.set_title("(b) Separability vs Drift", fontsize=12, fontweight='bold')
ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.3, label='Large effect (0.8)')
ax.grid(True, alpha=0.3, axis='y')
ax.legend(fontsize=8)
# Annotate endpoints
ax.text(0, cohens_d[0]+0.3, f'{cohens_d[0]:.1f}', ha='center', fontsize=9, fontweight='bold')
ax.text(10, cohens_d[-1]+0.3, f'{cohens_d[-1]:.1f}', ha='center', fontsize=9, fontweight='bold', color='darkred')
ax.annotate('Graceful\ndegradation', xy=(8, 7.63), xytext=(6, 3.5),
            fontsize=10, color='darkred', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5))

# Panel (c): ID vs OOD distance trajectories
ax = axes[2]
ax.plot(drifts, id_dists, 'b^-', linewidth=2, markersize=8, label='ID (drifted)')
ax.axhline(y=ood_dist, color='red', linestyle='-', linewidth=2, label=f'OOD mean ({ood_dist:.3f})')
ax.fill_between(drifts, id_dists, ood_dist, alpha=0.15, color='green')
ax.set_xlabel('Drift Level', fontsize=11)
ax.set_ylabel('Cosine Distance to Centroid', fontsize=11)
ax.set_title('(c) ID-OOD Separation Gap', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.annotate('Gap remains\nlarge even at\nmax drift', xy=(1.0, 0.32), xytext=(0.5, 0.35),
            fontsize=9, color='darkgreen', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='darkgreen'))

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig66_temporal_stability.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig66_temporal_stability.pdf', dpi=200, bbox_inches='tight')
print("Saved fig66_temporal_stability.png/pdf")
