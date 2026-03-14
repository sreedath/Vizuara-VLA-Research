"""Generate Figure 52: Comprehensive Method Comparison."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Panel (a): All methods ranked
ax = axes[0]
methods = ['Attn Max\n(cal-free)', 'All Equal\nFusion', 'Attn Entropy\n(cal-free)',
           'Cos+Attn\nFusion', 'Norm Diff\n(calibrated)', 'Cos+Mass\nFusion',
           'Cosine\n(calibrated)', '1-Mass\n(output)', '1-MSP\n(output)', 'Energy\n(output)']
aurocs = [1.000, 0.999, 0.993, 0.979, 0.969, 0.919, 0.913, 0.746, 0.733, 0.693]
colors = ['#4CAF50', '#9C27B0', '#2196F3', '#FF9800', '#795548',
          '#E91E63', '#FF5722', '#9E9E9E', '#9E9E9E', '#9E9E9E']
hatches = ['', '', '', '', '', '', '', '///', '///', '///']

bars = ax.barh(range(len(methods)), aurocs, 0.6, color=colors,
               edgecolor='black', linewidth=0.5, alpha=0.85)
for bar, h in zip(bars, hatches):
    bar.set_hatch(h)

ax.axvline(x=0.5, color='red', linestyle=':', alpha=0.3)
ax.axvline(x=0.9, color='green', linestyle='--', alpha=0.3, label='0.9 threshold')
ax.set_yticks(range(len(methods)))
ax.set_yticklabels(methods, fontsize=9)
ax.set_xlabel('AUROC', fontsize=11)
ax.set_title('(a) Detection Methods Ranked', fontsize=12, fontweight='bold')
ax.set_xlim(0.6, 1.05)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='x')

for bar, v in zip(bars, aurocs):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2.,
            f'{v:.3f}', ha='left', va='center', fontsize=9, fontweight='bold')

# Add cal-free / calibrated labels
ax.annotate('Cal-free', xy=(0.98, 0), fontsize=8, color='#4CAF50',
            fontweight='bold')
ax.annotate('Calibrated', xy=(0.88, 5), fontsize=8, color='#FF5722',
            fontweight='bold')
ax.annotate('Output only', xy=(0.65, 8), fontsize=8, color='#9E9E9E',
            fontweight='bold')

# Panel (b): Per-OOD type for top methods
ax = axes[1]
ood_types = ['Noise', 'Indoor', 'Inverted', 'Blackout']
cosine = [0.979, 0.849, 0.823, 1.000]
attn_max = [1.000, 1.000, 1.000, 1.000]
cos_mass = [1.000, 0.812, 0.865, 1.000]
attn_ent = [1.000, 1.000, 0.974, 1.000]

x = np.arange(len(ood_types))
width = 0.2
ax.bar(x - 1.5*width, cosine, width, label='Cosine', color='#FF5722',
       edgecolor='black', linewidth=0.5, alpha=0.85)
ax.bar(x - 0.5*width, attn_max, width, label='Attn Max', color='#4CAF50',
       edgecolor='black', linewidth=0.5, alpha=0.85)
ax.bar(x + 0.5*width, attn_ent, width, label='Attn Entropy', color='#2196F3',
       edgecolor='black', linewidth=0.5, alpha=0.85)
ax.bar(x + 1.5*width, cos_mass, width, label='Cos+Mass', color='#E91E63',
       edgecolor='black', linewidth=0.5, alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(ood_types, fontsize=10)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(b) Per-OOD Type (Top Methods)', fontsize=12, fontweight='bold')
ax.set_ylim(0.7, 1.1)
ax.legend(fontsize=8, loc='lower left')
ax.grid(True, alpha=0.3, axis='y')
ax.annotate('Attn: perfect\nacross all types', xy=(2, 1.0), xytext=(0.5, 0.8),
            fontsize=8, color='#4CAF50', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#4CAF50'))

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig52_comprehensive.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig52_comprehensive.pdf', dpi=200, bbox_inches='tight')
print("Saved fig52_comprehensive.png/pdf")
