"""Generate Figure 104: Resolution Sensitivity."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/resolution_sensitivity_20260314_224344.json") as f:
    data = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

resolutions = data['resolutions']
aurocs = [data['results'][str(r)]['auroc'] for r in resolutions]
ds = [data['results'][str(r)]['d'] for r in resolutions]
sims = [data['centroid_similarity_to_native'][str(r)] for r in resolutions]

# Panel (a): AUROC and d by resolution
ax = axes[0]
ax2 = ax.twinx()
ln1 = ax.plot(resolutions, aurocs, 'o-', color='#2196F3', linewidth=2, markersize=8, label='AUROC')
ln2 = ax2.plot(resolutions, ds, 's-', color='#F44336', linewidth=2, markersize=8, label="Cohen's d")
ax.set_xlabel('Resolution (pixels)', fontsize=11)
ax.set_ylabel('AUROC', fontsize=11, color='#2196F3')
ax2.set_ylabel("Cohen's d", fontsize=11, color='#F44336')
ax.set_title('(a) Detection vs Resolution', fontsize=12, fontweight='bold')
ax.set_ylim(0.95, 1.02)
ax.grid(True, alpha=0.3)
lns = ln1 + ln2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, fontsize=9)
ax.text(0.5, 0.05, 'All AUROC = 1.000', transform=ax.transAxes,
        ha='center', fontsize=10, fontweight='bold', color='green')

# Panel (b): Centroid similarity to native resolution
ax = axes[1]
ax.plot(resolutions, sims, 'D-', color='#9C27B0', linewidth=2, markersize=8)
ax.set_xlabel('Resolution (pixels)', fontsize=11)
ax.set_ylabel('Cosine Similarity to Native Centroid', fontsize=11)
ax.set_title('(b) Centroid Stability vs Resolution', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
for i, (r, s) in enumerate(zip(resolutions, sims)):
    ax.annotate(f'{s:.3f}', (r, s), textcoords="offset points",
                xytext=(0, 10), ha='center', fontsize=8)

# Panel (c): d as bar chart
ax = axes[2]
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(resolutions)))
bars = ax.bar([str(r) for r in resolutions], ds, color=colors, alpha=0.8,
              edgecolor='black', linewidth=0.5)
ax.set_xlabel('Resolution', fontsize=11)
ax.set_ylabel("Cohen's d", fontsize=11)
ax.set_title("(c) Separation by Resolution", fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=ds[3], color='green', linestyle='--', linewidth=1.5, alpha=0.7,
           label=f'Native 256 (d={ds[3]:.1f})')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig104_resolution_sensitivity.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig104_resolution_sensitivity.pdf', dpi=200, bbox_inches='tight')
print("Saved fig104_resolution_sensitivity.png/pdf")
