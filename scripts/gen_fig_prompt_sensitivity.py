"""Generate Figure 57: Prompt Sensitivity Analysis."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# Panel (a): AUROC by prompt
ax = axes[0]
prompts = ['Original\n(drive 25)', 'Simplified\n(drive)', 'Fast\n(drive 60)',
           'Lane\nChange', 'Emergency\nBrake']
cos_aurocs = [1.000, 1.000, 1.000, 1.000, 1.000]
attn_aurocs = [1.000, 1.000, 1.000, 0.996, 0.840]

x = np.arange(len(prompts))
width = 0.35
bars1 = ax.bar(x - width/2, cos_aurocs, width, label='Cosine',
               color='#FF5722', edgecolor='black', linewidth=0.5, alpha=0.85)
bars2 = ax.bar(x + width/2, attn_aurocs, width, label='Attn Max',
               color='#4CAF50', edgecolor='black', linewidth=0.5, alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(prompts, fontsize=9)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(a) Detection AUROC by Prompt', fontsize=12, fontweight='bold')
ax.set_ylim(0.7, 1.1)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
ax.annotate('Attention degrades\non non-driving prompts', xy=(4, 0.84), xytext=(2.5, 0.78),
            fontsize=8, color='red', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='red'))

for bar, v in zip(bars1, cos_aurocs):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{v:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
for bar, v in zip(bars2, attn_aurocs):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{v:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# Panel (b): Cross-prompt centroid distances (heatmap)
ax = axes[1]
prompt_labels = ['Original', 'Simplified', 'Fast', 'Lane', 'Brake']
dist_matrix = np.zeros((5, 5))
dists = {
    (0,1): 0.380, (0,2): 0.037, (0,3): 0.453, (0,4): 0.523,
    (1,2): 0.403, (1,3): 0.443, (1,4): 0.477,
    (2,3): 0.467, (2,4): 0.533,
    (3,4): 0.470,
}
for (i,j), d in dists.items():
    dist_matrix[i,j] = d
    dist_matrix[j,i] = d

im = ax.imshow(dist_matrix, cmap='YlOrRd', vmin=0, vmax=0.6)
ax.set_xticks(range(5))
ax.set_yticks(range(5))
ax.set_xticklabels(prompt_labels, fontsize=9, rotation=30, ha='right')
ax.set_yticklabels(prompt_labels, fontsize=9)
ax.set_title('(b) Cross-Prompt Centroid\nCosine Distance', fontsize=12, fontweight='bold')
for i in range(5):
    for j in range(5):
        color = 'white' if dist_matrix[i,j] > 0.3 else 'black'
        ax.text(j, i, f'{dist_matrix[i,j]:.2f}', ha='center', va='center',
                fontsize=9, color=color, fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.8)

# Panel (c): Key finding summary
ax = axes[2]
ax.axis('off')
findings = [
    ("Cosine Distance", "1.000 across ALL prompts", "#FF5722", True),
    ("Attn Max (driving)", "1.000 on driving prompts", "#4CAF50", True),
    ("Attn Max (brake)", "0.840 on brake prompt", "#F44336", False),
    ("Centroid shift", "Up to 0.533 between prompts", "#FF9800", False),
]

for i, (method, result, color, good) in enumerate(findings):
    y = 0.85 - i * 0.22
    marker = 'o' if good else 'x'
    ax.text(0.05, y, marker, fontsize=20, color=color, fontweight='bold',
            transform=ax.transAxes, va='center')
    ax.text(0.15, y, method, fontsize=11, fontweight='bold',
            transform=ax.transAxes, va='center')
    ax.text(0.15, y - 0.08, result, fontsize=10, color='gray',
            transform=ax.transAxes, va='center')

ax.set_title('(c) Key Findings', fontsize=12, fontweight='bold')
ax.text(0.05, 0.05, 'Cosine distance is prompt-invariant\nbecause it measures visual embedding\ndistance, not text-conditioned features.',
        fontsize=9, transform=ax.transAxes, va='bottom',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig57_prompt_sensitivity.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig57_prompt_sensitivity.pdf', dpi=200, bbox_inches='tight')
print("Saved fig57_prompt_sensitivity.png/pdf")
