"""Generate Figure 93: Inference Stability."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/inference_stability_20260314_214445.json") as f:
    data = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel (a): Pairwise cosine distance comparison
ax = axes[0]
categories = ['Same Highway\n(20 passes)', 'Same Noise\n(20 passes)', 'Different\nHighways (20)']
means = [
    data['same_highway']['pairwise_cosine']['mean'],
    data['same_noise']['pairwise_cosine']['mean'],
    data['different_highway']['pairwise_cosine']['mean'],
]
colors = ['#4CAF50', '#F44336', '#2196F3']

bars = ax.bar(range(len(categories)), means, color=colors, alpha=0.7,
              edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(categories)))
ax.set_xticklabels(categories, fontsize=9)
ax.set_ylabel('Mean Pairwise Cosine Distance', fontsize=11)
ax.set_title('(a) Embedding Variation', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

for i, v in enumerate(means):
    ax.text(i, v + 0.001, f'{v:.6f}', ha='center', fontsize=9, fontweight='bold')

# Panel (b): Score stability
ax = axes[1]
score_cats = ['Highway\n(same image)', 'Noise\n(same image)']
score_means = [data['same_highway']['score_mean'], data['same_noise']['score_mean']]
score_stds = [data['same_highway']['score_std'], data['same_noise']['score_std']]
score_colors = ['#4CAF50', '#F44336']

bars = ax.bar(range(len(score_cats)), score_means, yerr=score_stds, color=score_colors,
              alpha=0.7, edgecolor='black', linewidth=0.5, capsize=5)
ax.set_xticks(range(len(score_cats)))
ax.set_xticklabels(score_cats, fontsize=10)
ax.set_ylabel('Cosine Distance Score', fontsize=11)
ax.set_title('(b) Score Determinism', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

for i, (m, s) in enumerate(zip(score_means, score_stds)):
    ax.text(i, m + 0.02, f'std={s:.0e}', ha='center', fontsize=9, fontweight='bold')

# Panel (c): Bit-exact results summary
ax = axes[2]
labels = ['Highway\nBit-Exact', 'Noise\nBit-Exact', 'Per-Dim Var\n(same)', 'Per-Dim Var\n(different)']
values = [
    data['same_highway']['bit_exact_matches'] / (data['same_highway']['n_passes'] - 1) * 100,
    data['same_noise']['bit_exact_matches'] / (data['same_noise']['n_passes'] - 1) * 100,
    0,  # placeholder for log scale
    0,
]

# Bar chart for bit-exact rate
ax2_colors = ['#4CAF50', '#F44336']
ax.bar([0, 1], [values[0], values[1]], color=ax2_colors, alpha=0.7,
       edgecolor='black', linewidth=0.5)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Highway\nBit-Exact', 'Noise\nBit-Exact'], fontsize=10)
ax.set_ylabel('Match Rate (%)', fontsize=11)
ax.set_ylim(0, 110)
ax.set_title('(c) Determinism Rate', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

for i, v in enumerate([values[0], values[1]]):
    ax.text(i, v + 2, f'{v:.0f}%', ha='center', fontsize=12, fontweight='bold')

# Add text annotation
ax.text(0.5, 60, f'19/19 matches\nzero variance\nacross all dims',
        transform=ax.transAxes, ha='center', va='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig93_inference_stability.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig93_inference_stability.pdf', dpi=200, bbox_inches='tight')
print("Saved fig93_inference_stability.png/pdf")
