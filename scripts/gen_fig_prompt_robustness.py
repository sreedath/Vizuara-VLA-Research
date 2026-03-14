"""Generate Figure 40: Prompt Robustness Analysis."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Data from experiment
prompts = ['original', 'speed_50', 'cautious', 'simple', 'different']
prompt_labels = ['Original\n(25 m/s)', 'Speed\n(50 m/s)', 'Cautious\ndriving', 'Simple\nnavigate', 'Different\nprediction']
aurocs = [0.917, 0.883, 0.802, 0.870, 0.888]
id_cos = [0.5164, 0.6054, 0.5387, 0.4424, 0.4930]
ood_cos = [0.7585, 0.7736, 0.7146, 0.6378, 0.6745]

# Panel (a): Per-prompt AUROC
ax = axes[0]
colors = ['#2196F3' if a >= 0.85 else '#FF9800' if a >= 0.8 else '#F44336' for a in aurocs]
bars = ax.bar(range(len(prompts)), aurocs, color=colors, edgecolor='black',
              linewidth=0.5, alpha=0.85)
ax.set_xticks(range(len(prompts)))
ax.set_xticklabels(prompt_labels, fontsize=8)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(a) Self-Calibrated AUROC', fontsize=12, fontweight='bold')
ax.set_ylim(0.6, 1.0)
ax.grid(True, alpha=0.3, axis='y')

# Add mean line
mean_auroc = np.mean(aurocs)
ax.axhline(y=mean_auroc, color='red', linestyle='--', alpha=0.5,
           label=f'Mean: {mean_auroc:.3f}±{np.std(aurocs):.3f}')
ax.legend(fontsize=9)

for bar, val in zip(bars, aurocs):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.005, f'{val:.3f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# Panel (b): ID vs OOD cosine distance per prompt
ax = axes[1]
x = np.arange(len(prompts))
width = 0.35
bars1 = ax.bar(x - width/2, id_cos, width, label='ID', color='#2196F3',
               edgecolor='black', linewidth=0.5, alpha=0.85)
bars2 = ax.bar(x + width/2, ood_cos, width, label='OOD', color='#F44336',
               edgecolor='black', linewidth=0.5, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(prompt_labels, fontsize=8)
ax.set_ylabel('Mean Cosine Distance', fontsize=11)
ax.set_title('(b) ID vs OOD Separation', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Panel (c): Cross-prompt calibration
ax = axes[2]
self_cal = [0.883, 0.802, 0.870, 0.888]
cross_cal = [0.867, 0.789, 0.841, 0.781]
cross_prompts = ['speed_50', 'cautious', 'simple', 'different']
cross_labels = ['Speed\n(50 m/s)', 'Cautious', 'Simple', 'Different']

x = np.arange(len(cross_prompts))
width = 0.35
bars1 = ax.bar(x - width/2, self_cal, width, label='Self-calibrated',
               color='#4CAF50', edgecolor='black', linewidth=0.5, alpha=0.85)
bars2 = ax.bar(x + width/2, cross_cal, width, label='Cross-calibrated\n(original centroid)',
               color='#FF9800', edgecolor='black', linewidth=0.5, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(cross_labels, fontsize=8)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(c) Cross-Prompt Calibration Transfer', fontsize=12, fontweight='bold')
ax.set_ylim(0.6, 1.0)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

for i, (s, c) in enumerate(zip(self_cal, cross_cal)):
    drop = s - c
    ax.text(i + width/2, c + 0.005, f'-{drop:.2f}', ha='center', va='bottom',
            fontsize=8, color='red', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig40_prompt_robustness.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig40_prompt_robustness.pdf', dpi=200, bbox_inches='tight')
print("Saved fig40_prompt_robustness.png/pdf")
