"""Generate Figure 51: Attention Detection Robustness Under Perturbation."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

perturbations = ['none', 'blur_1', 'blur_3', 'blur_5', 'bright_0.5',
                 'bright_1.5', 'bright_2.0', 'jpeg_10', 'jpeg_50',
                 'noise_25', 'noise_50']
pretty_names = ['None', 'Blur\nr=1', 'Blur\nr=3', 'Blur\nr=5', 'Bright\n0.5×',
                'Bright\n1.5×', 'Bright\n2.0×', 'JPEG\nq=10', 'JPEG\nq=50',
                'Noise\nσ=25', 'Noise\nσ=50']
attn_max = [1.000, 0.972, 0.389, 0.528, 1.000, 0.972, 0.722, 1.000, 1.000, 1.000, 1.000]
attn_ent = [1.000, 1.000, 0.472, 0.389, 1.000, 1.000, 0.931, 1.000, 1.000, 1.000, 1.000]
cosine = [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000]

# Panel (a): All three methods across perturbations
ax = axes[0]
x = np.arange(len(perturbations))
ax.plot(x, attn_max, 'o-', color='#4CAF50', linewidth=2, markersize=6, label='Attn Max (cal-free)')
ax.plot(x, attn_ent, 's-', color='#2196F3', linewidth=2, markersize=6, label='Attn Entropy (cal-free)')
ax.plot(x, cosine, '^-', color='#FF9800', linewidth=2, markersize=6, label='Cosine (calibrated)')
ax.axhline(y=0.5, color='red', linestyle=':', alpha=0.3, label='Random')
ax.set_xticks(x)
ax.set_xticklabels(pretty_names, fontsize=7)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(a) Detection Robustness Across Perturbations', fontsize=12, fontweight='bold')
ax.set_ylim(0.3, 1.1)
ax.legend(fontsize=7, loc='lower left')
ax.grid(True, alpha=0.3)

# Highlight vulnerability zone
ax.axvspan(1.5, 3.5, alpha=0.1, color='red')
ax.annotate('Blur\nvulnerability', xy=(2.5, 0.45), fontsize=8,
            color='red', ha='center', fontweight='bold')

# Panel (b): Complementarity analysis
ax = axes[1]
# For each perturbation, plot max(attn_max, cosine)
best_of_both = [max(a, c) for a, c in zip(attn_max, cosine)]
ax.bar(x - 0.2, attn_max, 0.2, label='Attn Max', color='#4CAF50',
       edgecolor='black', linewidth=0.5, alpha=0.85)
ax.bar(x, cosine, 0.2, label='Cosine', color='#FF9800',
       edgecolor='black', linewidth=0.5, alpha=0.85)
ax.bar(x + 0.2, best_of_both, 0.2, label='Best of Both', color='#9C27B0',
       edgecolor='black', linewidth=0.5, alpha=0.85)
ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.3)
ax.set_xticks(x)
ax.set_xticklabels(pretty_names, fontsize=7)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(b) Complementary Strengths', fontsize=12, fontweight='bold')
ax.set_ylim(0.3, 1.1)
ax.legend(fontsize=7, loc='lower left')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig51_attn_robust.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig51_attn_robust.pdf', dpi=200, bbox_inches='tight')
print("Saved fig51_attn_robust.png/pdf")
