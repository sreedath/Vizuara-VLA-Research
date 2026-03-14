"""Generate Figure 70: Threshold Sensitivity Analysis."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel (a): ID vs OOD score distributions
ax = axes[0]
id_mean, id_std = 0.0883, 0.0085
ood_mean, ood_std = 0.3734, 0.0777
id_min, id_max = 0.0772, 0.1238
ood_min, ood_max = 0.2465, 0.4823
youden = 0.2465

x_id = np.linspace(0.05, 0.15, 200)
x_ood = np.linspace(0.15, 0.55, 200)
y_id = np.exp(-0.5*((x_id - id_mean)/id_std)**2) / (id_std * np.sqrt(2*np.pi))
y_ood = np.exp(-0.5*((x_ood - ood_mean)/ood_std)**2) / (ood_std * np.sqrt(2*np.pi))

ax.fill_between(x_id, y_id, alpha=0.4, color='#4CAF50', label=f'ID (μ={id_mean:.3f})')
ax.fill_between(x_ood, y_ood, alpha=0.4, color='#F44336', label=f'OOD (μ={ood_mean:.3f})')
ax.axvline(x=youden, color='black', linestyle='--', linewidth=2, label=f'Youden τ={youden:.3f}')
ax.axvline(x=id_max, color='blue', linestyle=':', alpha=0.5, label=f'ID max={id_max:.3f}')
ax.axvline(x=ood_min, color='red', linestyle=':', alpha=0.5, label=f'OOD min={ood_min:.3f}')

ax.set_xlabel('Cosine Distance', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('(a) Score Distributions', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)

# Annotate gap
ax.annotate('Gap: 0.123', xy=(0.185, 5), fontsize=10, color='darkgreen',
            fontweight='bold', ha='center')
ax.annotate('', xy=(id_max, 4), xytext=(ood_min, 4),
            arrowprops=dict(arrowstyle='<->', color='darkgreen', lw=2))

# Panel (b): Threshold strategies comparison
ax = axes[1]
strategies = ['Youden', 'FPR<1%', 'FPR<5%', 'EER', 'μ+2σ', 'μ+3σ', 'μ+5σ']
thresholds = [0.2465, 0.2465, 0.2465, 0.2465, 0.0985, 0.1049, 0.1176]
fprs = [0.0, 0.0, 0.0, 0.0, 0.10, 0.025, 0.025]
tprs = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

x = np.arange(len(strategies))
width = 0.35
bars1 = ax.bar(x - width/2, tprs, width, label='TPR', color='#4CAF50', alpha=0.85,
               edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, fprs, width, label='FPR', color='#F44336', alpha=0.85,
               edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(strategies, fontsize=9, rotation=30, ha='right')
ax.set_ylabel('Rate', fontsize=11)
ax.set_title('(b) Threshold Strategy Comparison', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 1.15)

for bar, v in zip(bars2, fprs):
    if v > 0:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{v:.1%}', ha='center', va='bottom', fontsize=8, fontweight='bold', color='red')

ax.annotate('Data-driven thresholds\nachieve 0% FPR + 100% TPR', xy=(1, 0.9),
            fontsize=9, color='darkgreen', fontweight='bold')

# Panel (c): Per-category TPR at Youden
ax = axes[2]
cats = ['blackout', 'indoor', 'inverted', 'noise', 'twilight', 'snow']
cat_tprs = [1.0, 1.0, 1.0, 1.0, 1.0, 0.9]
colors = ['#4CAF50' if t == 1.0 else '#FF9800' for t in cat_tprs]

bars = ax.bar(range(len(cats)), cat_tprs, 0.6, color=colors,
              edgecolor='black', linewidth=0.5, alpha=0.85)
ax.set_xticks(range(len(cats)))
ax.set_xticklabels(cats, fontsize=10, rotation=30, ha='right')
ax.set_ylabel('TPR at Youden Threshold', fontsize=11)
ax.set_title('(c) Per-Category Detection Rate', fontsize=12, fontweight='bold')
ax.set_ylim(0.8, 1.05)
ax.grid(True, alpha=0.3, axis='y')

for bar, v in zip(bars, cat_tprs):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
            f'{v:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.annotate('Snow: hardest\n(90% TPR)', xy=(5, 0.9), xytext=(3.5, 0.85),
            fontsize=9, color='darkorange', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='darkorange'))

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig70_threshold_analysis.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig70_threshold_analysis.pdf', dpi=200, bbox_inches='tight')
print("Saved fig70_threshold_analysis.png/pdf")
