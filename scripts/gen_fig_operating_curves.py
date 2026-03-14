"""Generate Figure 38: Operating Characteristic Curves."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/operating_curves_20260314_173754.json") as f:
    data = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Panel (a): ROC curve (full pipeline)
ax = axes[0]
fpr = data['full_roc']['fpr']
tpr = data['full_roc']['tpr']
auroc = data['roc_data']['Full pipeline']['auroc']

ax.plot(fpr, tpr, 'b-', linewidth=2.5, label=f'Full pipeline (AUROC={auroc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')

# Mark operating points
opt = data['optimal_threshold']
ax.plot(opt['fpr'], opt['tpr'], 'r*', markersize=15, zorder=5,
        label=f"Optimal (J={opt['youdens_j']:.3f})")

# EER point
eer = data['eer']
ax.plot(eer, 1-eer, 'gs', markersize=10, zorder=5,
        label=f'EER={eer:.3f}')

ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.set_title('(a) ROC Curve (Full Pipeline)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)

# Panel (b): Coverage vs Safety tradeoff
ax = axes[1]
ops = data['operating_points']
alphas = sorted(ops.keys(), key=lambda x: float(x))
coverages = [ops[a]['coverage'] for a in alphas]
safeties = [ops[a]['safety'] for a in alphas]

ax.plot(coverages, safeties, 'bo-', linewidth=2, markersize=8, zorder=3)

for a, cov, saf in zip(alphas, coverages, safeties):
    if float(a) in [0.01, 0.05, 0.10, 0.20]:
        ax.annotate(f'α={a}', xy=(cov, saf),
                    xytext=(cov + 0.05, saf - 0.02),
                    fontsize=9, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='gray'))

ax.set_xlabel('ID Coverage (1 - FPR)', fontsize=11)
ax.set_ylabel('OOD Safety Rate (TPR)', fontsize=11)
ax.set_title('(b) Coverage-Safety Tradeoff', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(0.85, 1.02)

# Shade the "desirable" region
ax.axvspan(0.5, 1.0, alpha=0.1, color='green', label='High coverage')
ax.axhspan(0.95, 1.0, alpha=0.1, color='blue', label='High safety')
ax.legend(fontsize=8, loc='lower left')

# Panel (c): Method comparison bar chart
ax = axes[2]
method_names = ['Global cos\n(1f)', 'Per-scene\ncos (1f)', 'Action\nmass (1f)',
                'Per-scene\ncos (5s)', 'Full\npipeline']
method_keys = ['Global cosine (1f)', 'Per-scene cosine (1f)', 'Action mass (1f)',
               'Per-scene cosine (5s)', 'Full pipeline']
aurocs = [data['roc_data'][k]['auroc'] for k in method_keys]
aps = [data['roc_data'][k]['ap'] for k in method_keys]

x = np.arange(len(method_names))
width = 0.35
bars1 = ax.bar(x - width/2, aurocs, width, label='AUROC', color='#2196F3',
               edgecolor='black', linewidth=0.5, alpha=0.85)
bars2 = ax.bar(x + width/2, aps, width, label='Average Precision', color='#4CAF50',
               edgecolor='black', linewidth=0.5, alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(method_names, fontsize=8)
ax.set_ylabel('Score', fontsize=11)
ax.set_title('(c) AUROC vs Average Precision', fontsize=12, fontweight='bold')
ax.set_ylim(0.4, 1.05)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

for bar, val in zip(list(bars1) + list(bars2), aurocs + aps):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.2f}',
            ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig38_operating_curves.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig38_operating_curves.pdf', dpi=200, bbox_inches='tight')
print("Saved fig38_operating_curves.png/pdf")
