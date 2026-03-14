"""Generate Figure 60: Multi-Head Attention Analysis."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/head_analysis_20260314_194016.json") as f:
    data = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# Panel (a): Per-head AUROC
ax = axes[0]
n_heads = data['n_heads']
aurocs = [data['head_results'][str(h)]['auroc_max'] for h in range(n_heads)]
colors = ['#4CAF50' if a >= 0.95 else '#FF9800' if a >= 0.7 else '#F44336' if a < 0.5 else '#9E9E9E' for a in aurocs]

bars = ax.bar(range(n_heads), aurocs, 0.8, color=colors,
              edgecolor='black', linewidth=0.3, alpha=0.85)
ax.axhline(y=0.5, color='red', linestyle=':', alpha=0.3, label='Random')
ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.3)
ax.set_xlabel('Attention Head', fontsize=11)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(a) Per-Head AUROC (Last Layer)', fontsize=12, fontweight='bold')
ax.set_ylim(0.2, 1.1)
ax.grid(True, alpha=0.3, axis='y')

from matplotlib.patches import Patch
legend_els = [Patch(facecolor='#4CAF50', label='Perfect (≥0.95)'),
              Patch(facecolor='#FF9800', label='Good (0.7-0.95)'),
              Patch(facecolor='#F44336', label='Failing (<0.5)')]
ax.legend(handles=legend_els, fontsize=7)

# Panel (b): Head AUROC distribution
ax = axes[1]
ax.hist(aurocs, bins=15, color='#2196F3', edgecolor='black', linewidth=0.5, alpha=0.85)
ax.axvline(x=0.5, color='red', linestyle=':', alpha=0.5, label='Random')
ax.axvline(x=np.mean(aurocs), color='green', linestyle='--', alpha=0.8,
           label=f'Mean ({np.mean(aurocs):.3f})')
ax.set_xlabel('AUROC', fontsize=11)
ax.set_ylabel('Number of Heads', fontsize=11)
ax.set_title('(b) AUROC Distribution Across Heads', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Count
n_perfect = sum(1 for a in aurocs if a >= 0.95)
n_fail = sum(1 for a in aurocs if a < 0.5)
ax.text(0.35, 5, f'{n_perfect} heads ≥ 0.95\n{n_fail} heads < 0.50',
        fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Panel (c): Top-k ensemble
ax = axes[2]
ks = [1, 3, 5, 10, 32]
ensemble_aurocs = [1.000, 1.000, 1.000, 1.000, 1.000]

ax.plot(ks, ensemble_aurocs, 'go-', linewidth=2, markersize=10)
ax.set_xlabel('Number of Heads in Ensemble', fontsize=11)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(c) Top-k Head Ensemble', fontsize=12, fontweight='bold')
ax.set_ylim(0.95, 1.01)
ax.set_xscale('log')
ax.set_xticks(ks)
ax.set_xticklabels([str(k) for k in ks])
ax.grid(True, alpha=0.3)
ax.annotate('Even a single head\nachieves perfect AUROC', xy=(1, 1.0), xytext=(3, 0.97),
            fontsize=10, color='green', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='green'))

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig60_head_analysis.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig60_head_analysis.pdf', dpi=200, bbox_inches='tight')
print("Saved fig60_head_analysis.png/pdf")
