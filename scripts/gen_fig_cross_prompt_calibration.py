"""Generate Figure 108: Cross-Prompt Calibration Transfer."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/cross_prompt_calibration_20260314_231413.json") as f:
    data = json.load(f)

prompts = data['prompt_names']
short = ['drive_fwd', 'lane_keep', 'slow_down', 'navigate', 'avoid_obs']
auroc = np.array(data['auroc_matrix'])
d_mat = np.array(data['d_matrix'])
sim = np.array(data['centroid_similarity'])

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel A: D-prime transfer matrix
ax = axes[0]
im = ax.imshow(d_mat, cmap='YlOrRd', aspect='equal')
for i in range(len(prompts)):
    for j in range(len(prompts)):
        color = 'white' if d_mat[i,j] > 40 else 'black'
        ax.text(j, i, f"{d_mat[i,j]:.1f}", ha='center', va='center', fontsize=8, color=color)
ax.set_xticks(range(len(prompts)))
ax.set_xticklabels(short, rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(len(prompts)))
ax.set_yticklabels(short, fontsize=8)
ax.set_xlabel("Test Prompt")
ax.set_ylabel("Calibration Prompt")
ax.set_title("(A) D-prime Transfer Matrix")
plt.colorbar(im, ax=ax, shrink=0.8)

# Panel B: Centroid similarity matrix
ax = axes[1]
im2 = ax.imshow(sim, cmap='Blues', vmin=0.4, vmax=1.0, aspect='equal')
for i in range(len(prompts)):
    for j in range(len(prompts)):
        color = 'white' if sim[i,j] > 0.8 else 'black'
        ax.text(j, i, f"{sim[i,j]:.2f}", ha='center', va='center', fontsize=8, color=color)
ax.set_xticks(range(len(prompts)))
ax.set_xticklabels(short, rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(len(prompts)))
ax.set_yticklabels(short, fontsize=8)
ax.set_xlabel("Prompt")
ax.set_ylabel("Prompt")
ax.set_title("(B) Centroid Cosine Similarity")
plt.colorbar(im2, ax=ax, shrink=0.8)

# Panel C: Same-prompt vs cross-prompt d-prime
ax = axes[2]
diag_d = [d_mat[i,i] for i in range(len(prompts))]
off_d = []
for i in range(len(prompts)):
    for j in range(len(prompts)):
        if i != j:
            off_d.append(d_mat[i,j])

positions = [0, 1]
bp = ax.boxplot([diag_d, off_d], positions=positions, widths=0.5, patch_artist=True)
colors = ['#2196F3', '#FF9800']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_xticks(positions)
ax.set_xticklabels(['Same Prompt', 'Cross Prompt'])
ax.set_ylabel("D-prime")
ax.set_title("(C) Detection Strength")
ax.axhline(y=3, color='red', linestyle='--', alpha=0.5, label='Detection threshold')
ax.legend(fontsize=8)

# Add annotation
min_cross = min(off_d)
ax.annotate(f"Min cross: {min_cross:.1f}", xy=(1, min_cross), fontsize=8,
            xytext=(1.3, min_cross+5), arrowprops=dict(arrowstyle='->', color='red'),
            color='red')

plt.suptitle("Cross-Prompt Calibration Transfer (Exp 122)\nAll 25 cells AUROC = 1.000", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/latex/figures/fig108_cross_prompt_calibration.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig108")
