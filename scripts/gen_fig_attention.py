"""Generate Figure 49: Attention Pattern Analysis."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# Panel (a): Layer-wise attention AUROC
ax = axes[0]
layers = ['L16', 'L24', 'L28', 'L31']
entropy_aurocs = [0.859, 0.586, 0.534, 0.987]
max_attn_aurocs = [0.802, 0.534, 0.540, 1.000]

x = np.arange(len(layers))
width = 0.35
bars1 = ax.bar(x - width/2, entropy_aurocs, width, label='Entropy AUROC',
               color='#2196F3', edgecolor='black', linewidth=0.5, alpha=0.85)
bars2 = ax.bar(x + width/2, max_attn_aurocs, width, label='Max Attn AUROC',
               color='#FF9800', edgecolor='black', linewidth=0.5, alpha=0.85)
ax.axhline(y=0.5, color='red', linestyle=':', alpha=0.3)
ax.set_xticks(x)
ax.set_xticklabels(layers, fontsize=10)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(a) Attention OOD Detection by Layer', fontsize=12, fontweight='bold')
ax.set_ylim(0.4, 1.1)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
ax.annotate('Last layer:\nperfect detection', xy=(3, 1.0), xytext=(1.5, 0.85),
            fontsize=8, color='#FF9800', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#FF9800'))

# Panel (b): ID vs OOD attention entropy and max attention (L31)
ax = axes[1]
scenarios = ['Highway', 'Urban', 'Noise', 'Indoor', 'Inverted', 'Blackout']
# Data from the "last layer" analysis (L16 based on the per-scenario section)
# Using L31 data: ID entropy 2.434, OOD entropy 2.005
# For per-scenario entropy from the log:
entropies_l16 = [1.574, 1.651, 1.828, 1.745, 1.608, 1.687]
is_ood = [False, False, True, True, True, True]
colors = ['#2196F3', '#2196F3', '#F44336', '#FF9800', '#9C27B0', '#333333']

bars = ax.bar(range(len(scenarios)), entropies_l16, 0.6, color=colors,
              edgecolor='black', linewidth=0.5, alpha=0.85)
ax.set_xticks(range(len(scenarios)))
ax.set_xticklabels(scenarios, fontsize=9, rotation=20, ha='right')
ax.set_ylabel('Mean Attention Entropy (L16)', fontsize=11)
ax.set_title('(b) Per-Scenario Attention Entropy', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
id_mean = np.mean([entropies_l16[0], entropies_l16[1]])
ax.axhline(y=id_mean, color='blue', linestyle='--', alpha=0.5, label=f'ID mean ({id_mean:.3f})')
ax.legend(fontsize=8)

# Panel (c): Attention vs Hidden State detection comparison
ax = axes[2]
methods = ['Cosine\n(hidden)', 'Attn Entropy\n(L31)', 'Max Attn\n(L31)', 'Attn Entropy\n(L16)']
aurocs = [0.933, 0.987, 1.000, 0.859]
colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']

bars = ax.bar(range(len(methods)), aurocs, 0.6, color=colors,
              edgecolor='black', linewidth=0.5, alpha=0.85)
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods, fontsize=9)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(c) Attention vs Hidden State Detection', fontsize=12, fontweight='bold')
ax.set_ylim(0.8, 1.05)
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, aurocs):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig49_attention.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig49_attention.pdf', dpi=200, bbox_inches='tight')
print("Saved fig49_attention.png/pdf")
