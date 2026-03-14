"""Generate Figure 94: Gradient-Free Sensitivity Analysis."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/sensitivity_analysis_20260314_214725.json") as f:
    data = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel (a): Gaussian noise sensitivity
ax = axes[0]
gauss = data['gaussian_noise']
sigmas = [g['sigma'] for g in gauss]
scores = [g['score'] for g in gauss]
ax.semilogx(sigmas, scores, 'o-', color='#F44336', linewidth=2, markersize=8)
ax.axhline(y=data['base_score'], color='green', linestyle='--', linewidth=1.5,
           label=f'Base ({data["base_score"]:.4f})')
ax.axhline(y=0.035, color='red', linestyle=':', linewidth=1.5,
           label='Typical threshold')
ax.set_xlabel('Gaussian Noise Sigma', fontsize=11)
ax.set_ylabel('Cosine Distance Score', fontsize=11)
ax.set_title('(a) Gaussian Noise Sensitivity', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel (b): Brightness shift
ax = axes[1]
bright = data['brightness_shift']
shifts = [b['shift'] for b in bright]
b_scores = [b['score'] for b in bright]
colors_b = ['#2196F3' if s < 0 else '#FF9800' for s in shifts]
ax.bar(range(len(shifts)), b_scores, color=colors_b, alpha=0.7,
       edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(shifts)))
ax.set_xticklabels([str(s) for s in shifts], fontsize=7, rotation=45)
ax.axhline(y=data['base_score'], color='green', linestyle='--', linewidth=1.5)
ax.axhline(y=0.035, color='red', linestyle=':', linewidth=1.5)
ax.set_xlabel('Brightness Shift', fontsize=11)
ax.set_ylabel('Cosine Distance Score', fontsize=11)
ax.set_title('(b) Brightness Sensitivity', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Panel (c): Progressive corruption
ax = axes[2]
corr = data['progressive_corruption']
alphas = [c['alpha'] for c in corr]
c_scores = [c['score'] for c in corr]
ax.plot(alphas, c_scores, 'o-', color='#9C27B0', linewidth=2, markersize=8)
ax.axhline(y=data['base_score'], color='green', linestyle='--', linewidth=1.5,
           label=f'Base ({data["base_score"]:.4f})')
ax.axhline(y=0.035, color='red', linestyle=':', linewidth=1.5,
           label='Typical threshold')
ax.fill_between([0, 0.05], [0, 0], [0.55, 0.55], alpha=0.1, color='green', label='Sub-threshold')
ax.set_xlabel('Noise Blend Alpha', fontsize=11)
ax.set_ylabel('Cosine Distance Score', fontsize=11)
ax.set_title('(c) Progressive Corruption', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig94_sensitivity_analysis.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig94_sensitivity_analysis.pdf', dpi=200, bbox_inches='tight')
print("Saved fig94_sensitivity_analysis.png/pdf")
