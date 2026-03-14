"""Generate Figure 73: Cross-Seed Robustness."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

seeds = ['0', '10K', '20K', '30K', '40K']
aurocs = [1.000, 1.000, 1.000, 1.000, 1.000]
cohens_d = [5.73, 5.45, 5.37, 5.69, 5.67]
id_means = [0.0876, 0.0886, 0.0842, 0.0899, 0.0874]
ood_means = [0.3747, 0.3736, 0.3760, 0.3788, 0.3691]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel (a): AUROC across seeds
ax = axes[0]
ax.bar(range(len(seeds)), aurocs, 0.6, color='#4CAF50',
       edgecolor='black', linewidth=0.5, alpha=0.85)
ax.set_xticks(range(len(seeds)))
ax.set_xticklabels(seeds, fontsize=11)
ax.set_xlabel('Seed Offset', fontsize=11)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(a) AUROC Across Seeds', fontsize=12, fontweight='bold')
ax.set_ylim(0.9, 1.05)
ax.grid(True, alpha=0.3, axis='y')
ax.annotate('Perfect 1.000\nacross ALL seeds', xy=(2, 1.0), xytext=(2, 0.94),
            fontsize=11, color='darkgreen', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='darkgreen'))

# Panel (b): Cohen's d across seeds
ax = axes[1]
colors = plt.cm.YlOrRd(np.array(cohens_d) / max(cohens_d))
bars = ax.bar(range(len(seeds)), cohens_d, 0.6, color=colors,
              edgecolor='black', linewidth=0.5, alpha=0.85)
ax.set_xticks(range(len(seeds)))
ax.set_xticklabels(seeds, fontsize=11)
ax.set_xlabel('Seed Offset', fontsize=11)
ax.set_ylabel("Cohen's d", fontsize=11)
ax.set_title("(b) Effect Size Stability", fontsize=12, fontweight='bold')
ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.3, label='Large effect (0.8)')
ax.grid(True, alpha=0.3, axis='y')
ax.legend(fontsize=8)

for bar, v in zip(bars, cohens_d):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
            f'{v:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

mean_d = np.mean(cohens_d)
std_d = np.std(cohens_d)
ax.axhline(y=mean_d, color='blue', linestyle='-', alpha=0.5, linewidth=1)
ax.text(4.3, mean_d, f'μ={mean_d:.2f}\nσ={std_d:.2f}', fontsize=9,
        color='blue', fontweight='bold')

# Panel (c): ID vs OOD distances
ax = axes[2]
x = np.arange(len(seeds))
width = 0.35
ax.bar(x - width/2, id_means, width, label='ID mean dist', color='#4CAF50',
       edgecolor='black', linewidth=0.5, alpha=0.85)
ax.bar(x + width/2, ood_means, width, label='OOD mean dist', color='#F44336',
       edgecolor='black', linewidth=0.5, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(seeds, fontsize=11)
ax.set_xlabel('Seed Offset', fontsize=11)
ax.set_ylabel('Cosine Distance', fontsize=11)
ax.set_title('(c) Score Stability', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Gap annotation
for i in range(len(seeds)):
    gap = ood_means[i] - id_means[i]
    ax.annotate('', xy=(i+width/2, id_means[i]), xytext=(i+width/2, ood_means[i]),
                arrowprops=dict(arrowstyle='<->', color='gray', lw=1))

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig73_cross_seed.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig73_cross_seed.pdf', dpi=200, bbox_inches='tight')
print("Saved fig73_cross_seed.png/pdf")
