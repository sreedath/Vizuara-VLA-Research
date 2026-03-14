"""Generate Figure 34: OOD Detection Baselines Comparison."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# Panel (a): Simple images — AUROC comparison
ax = axes[0]
methods_simple = [
    ('Cosine\ndistance', 0.965, '#2196F3', 'Ours'),
    ('Entropy', 0.844, '#9E9E9E', 'Standard'),
    ('MSP', 0.823, '#FF9800', 'Hendrycks 17'),
    ('Max\nlogit', 0.776, '#F44336', 'Hendrycks 22'),
    ('Energy\nscore', 0.750, '#9C27B0', 'Liu 2020'),
    ('Action\nmass', 0.620, '#4CAF50', 'Ours'),
]

names = [m[0] for m in methods_simple]
vals = [m[1] for m in methods_simple]
colors = [m[2] for m in methods_simple]

bars = ax.bar(range(len(names)), vals, color=colors, edgecolor='black',
              linewidth=0.5, alpha=0.85)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, fontsize=8)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(a) Simple Images: Baselines vs Ours', fontsize=12, fontweight='bold')
ax.set_ylim(0.5, 1.05)
ax.axhline(y=0.5, color='red', linestyle=':', alpha=0.3)
ax.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.3f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# Mark "Ours" methods
for i in [0, 5]:
    ax.text(i, 0.52, 'Ours', ha='center', fontsize=8, fontweight='bold',
            color='white', bbox=dict(boxstyle='round', facecolor=colors[i], alpha=0.8))

# Panel (b): Realistic images — AUROC comparison
ax = axes[1]
methods_realistic = [
    ('Action\nmass', 0.550, '#4CAF50', 'Ours'),
    ('Combo\n(0.7c+0.3m)', 0.516, '#1565C0', 'Ours'),
    ('Per-scene\ncosine', 0.497, '#2196F3', 'Ours'),
    ('Cosine\ndist', 0.491, '#42A5F5', 'Ours'),
    ('Energy', 0.394, '#9C27B0', 'Liu 2020'),
    ('Max\nlogit', 0.373, '#F44336', 'Hendrycks'),
    ('MSP', 0.219, '#FF9800', 'Hendrycks'),
    ('Entropy', 0.219, '#9E9E9E', 'Standard'),
]

names_r = [m[0] for m in methods_realistic]
vals_r = [m[1] for m in methods_realistic]
colors_r = [m[2] for m in methods_realistic]

bars = ax.bar(range(len(names_r)), vals_r, color=colors_r, edgecolor='black',
              linewidth=0.5, alpha=0.85)
ax.set_xticks(range(len(names_r)))
ax.set_xticklabels(names_r, fontsize=7)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(b) Realistic Images: Baselines vs Ours', fontsize=12, fontweight='bold')
ax.set_ylim(0.1, 0.7)
ax.axhline(y=0.5, color='red', linestyle=':', alpha=0.3, label='Random')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, vals_r):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.3f}',
            ha='center', va='bottom', fontsize=8, fontweight='bold')

# Panel (c): Method category comparison (grouped)
ax = axes[2]
categories = ['Standard\nbaselines', 'Our\nmethods']

# Simple images
simple_standard = np.mean([0.823, 0.750, 0.776, 0.844])  # MSP, Energy, MaxLogit, Entropy
simple_ours = np.mean([0.965, 0.620])  # Cosine, Action mass (mean)
simple_ours_best = 0.965  # Best

# Realistic images
real_standard = np.mean([0.219, 0.394, 0.373, 0.219])  # MSP, Energy, MaxLogit, Entropy
real_ours = np.mean([0.491, 0.550, 0.497, 0.516])  # All ours
real_ours_best = 0.550  # Best

x = np.arange(2)
width = 0.3

bars1 = ax.bar(x - width/2, [simple_standard, real_standard], width,
               label='Baselines (avg)', color='#FF9800', edgecolor='black',
               linewidth=0.5, alpha=0.7)
bars2 = ax.bar(x + width/2, [simple_ours_best, real_ours_best], width,
               label='Ours (best)', color='#2196F3', edgecolor='black',
               linewidth=0.5, alpha=0.9)

ax.set_xticks(x)
ax.set_xticklabels(['Simple\nimages', 'Realistic\nimages'], fontsize=11)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(c) Category Comparison', fontsize=12, fontweight='bold')
ax.set_ylim(0, 1.1)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

for bar, val in zip(list(bars1) + list(bars2),
                    [simple_standard, real_standard, simple_ours_best, real_ours_best]):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add improvement arrows
for i, (base, ours) in enumerate([(simple_standard, simple_ours_best),
                                   (real_standard, real_ours_best)]):
    gain = ours - base
    ax.annotate(f'+{gain:.2f}', xy=(i + width/2, ours - 0.02),
                fontsize=10, fontweight='bold', color='green', ha='center')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig34_ood_baselines.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig34_ood_baselines.pdf', dpi=200, bbox_inches='tight')
print("Saved fig34_ood_baselines.png/pdf")
