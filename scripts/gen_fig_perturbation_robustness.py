"""Generate Figure 39: Perturbation Robustness of OOD Detection."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

fig, axes = plt.subplots(1, 4, figsize=(18, 4))

# Data from experiment
perturbation_data = {
    'Gaussian Blur': {
        'levels': ['None', 'r=1', 'r=2', 'r=3', 'r=5'],
        'aurocs': [0.920, 0.950, 0.830, 0.720, 0.390],
        'color': '#2196F3',
    },
    'Brightness': {
        'levels': ['1.0x', '0.5x', '0.3x', '1.5x', '2.0x'],
        'aurocs': [0.920, 0.650, 0.770, 0.860, 0.900],
        'color': '#FF9800',
    },
    'JPEG Quality': {
        'levels': ['q95', 'q50', 'q20', 'q5', 'q1'],
        'aurocs': [0.920, 0.880, 0.900, 0.950, 0.680],
        'color': '#4CAF50',
    },
    'Gauss Noise': {
        'levels': ['σ=0', 'σ=10', 'σ=25', 'σ=50', 'σ=100'],
        'aurocs': [0.920, 0.940, 0.960, 0.580, 0.840],
        'color': '#9C27B0',
    },
}

for i, (name, pdata) in enumerate(perturbation_data.items()):
    ax = axes[i]
    levels = pdata['levels']
    aurocs = pdata['aurocs']
    color = pdata['color']

    bars = ax.bar(range(len(levels)), aurocs, color=color, edgecolor='black',
                  linewidth=0.5, alpha=0.85)

    # Color degraded bars red
    for j, (bar, val) in enumerate(zip(bars, aurocs)):
        if val < 0.7:
            bar.set_facecolor('#F44336')
            bar.set_alpha(0.7)
        elif val < 0.85:
            bar.set_facecolor('#FF9800')
            bar.set_alpha(0.7)

    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(levels, fontsize=9, rotation=30)
    ax.set_ylabel('AUROC', fontsize=10)
    ax.set_title(name, fontsize=11, fontweight='bold')
    ax.set_ylim(0.2, 1.05)
    ax.axhline(y=0.5, color='red', linestyle=':', alpha=0.3)
    ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.suptitle('Perturbation Robustness of Cosine Distance OOD Detection',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig39_perturbation_robustness.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig39_perturbation_robustness.pdf', dpi=200, bbox_inches='tight')
print("Saved fig39_perturbation_robustness.png/pdf")
