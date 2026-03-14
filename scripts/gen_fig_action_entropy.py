"""Generate Figure 59: Action Token Entropy Analysis."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# Panel (a): AUROC by logit feature
ax = axes[0]
features = ['Top-1\nProb', 'Logit\nStd', 'Logit\nMax', 'Entropy', 'Norm\nEntropy',
            'Top-5\nProb', 'Top-10\nProb', 'Energy']
aurocs = [0.750, 0.707, 0.624, 0.618, 0.618, 0.500, 0.441, 0.298]
colors = ['#FF9800' if a > 0.6 else '#F44336' if a < 0.5 else '#9E9E9E' for a in aurocs]

bars = ax.bar(range(len(features)), aurocs, 0.6, color=colors,
              edgecolor='black', linewidth=0.5, alpha=0.85)
ax.axhline(y=0.5, color='red', linestyle=':', alpha=0.3, label='Random')
ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.3, label='Cosine/Attn (hidden)')
ax.set_xticks(range(len(features)))
ax.set_xticklabels(features, fontsize=8)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(a) Output Logit Features for OOD', fontsize=12, fontweight='bold')
ax.set_ylim(0.2, 1.1)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')
for bar, v in zip(bars, aurocs):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
            f'{v:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# Panel (b): Per-scenario entropy
ax = axes[1]
scenarios = ['Highway', 'Urban', 'Noise', 'Indoor', 'Blackout']
entropies = [1.54, 2.11, 1.73, 1.86, 4.56]
is_ood = [False, False, True, True, True]
colors_s = ['#2196F3' if not o else '#F44336' for o in is_ood]

bars = ax.bar(range(len(scenarios)), entropies, 0.6, color=colors_s,
              edgecolor='black', linewidth=0.5, alpha=0.85)
ax.set_xticks(range(len(scenarios)))
ax.set_xticklabels(scenarios, fontsize=10)
ax.set_ylabel('Mean Output Entropy', fontsize=11)
ax.set_title('(b) Per-Scenario Entropy', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#2196F3', label='ID'),
                   Patch(facecolor='#F44336', label='OOD')]
ax.legend(handles=legend_elements, fontsize=9)
ax.annotate('Blackout: very high\nentropy (uncertain)', xy=(4, 4.56), xytext=(2.5, 3.5),
            fontsize=8, color='red', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='red'))

# Panel (c): Method hierarchy comparison
ax = axes[2]
methods = ['Cosine\n(hidden)', 'Attn Max\n(hidden)', 'Top-1 Prob\n(output)',
           'Logit Std\n(output)', 'Entropy\n(output)', 'Energy\n(output)']
aurocs_compare = [1.000, 1.000, 0.750, 0.707, 0.618, 0.298]
colors_c = ['#FF5722', '#4CAF50', '#FF9800', '#FF9800', '#9E9E9E', '#F44336']

bars = ax.barh(range(len(methods)), aurocs_compare, 0.6, color=colors_c,
               edgecolor='black', linewidth=0.5, alpha=0.85)
ax.set_yticks(range(len(methods)))
ax.set_yticklabels(methods, fontsize=9)
ax.set_xlabel('AUROC', fontsize=11)
ax.set_title('(c) Hidden vs Output Features', fontsize=12, fontweight='bold')
ax.set_xlim(0.2, 1.1)
ax.axvline(x=0.5, color='red', linestyle=':', alpha=0.3)
ax.grid(True, alpha=0.3, axis='x')
for bar, v in zip(bars, aurocs_compare):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2.,
            f'{v:.3f}', ha='left', va='center', fontsize=9, fontweight='bold')

ax.annotate('Hidden state\nmethods dominate', xy=(1.0, 0.5), xytext=(0.7, 3),
            fontsize=8, color='green', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='green'))

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig59_action_entropy.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig59_action_entropy.pdf', dpi=200, bbox_inches='tight')
print("Saved fig59_action_entropy.png/pdf")
