"""Generate Figure 50: Calibration-Free OOD Detection via Attention."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# Panel (a): Calibration-free vs calibrated methods
ax = axes[0]
methods = ['Attn Max\n(cal-free)', 'Attn Entropy\n(cal-free)', 'All-Equal\nFusion', 'Cosine\n(calibrated)', '1-Mass\n(cal-free)']
aurocs = [1.000, 0.983, 0.929, 0.589, 0.622]
colors = ['#4CAF50', '#2196F3', '#9C27B0', '#FF9800', '#F44336']

bars = ax.bar(range(len(methods)), aurocs, 0.6, color=colors,
              edgecolor='black', linewidth=0.5, alpha=0.85)
ax.axhline(y=0.5, color='red', linestyle=':', alpha=0.3)
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods, fontsize=9)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(a) Calibration-Free vs Calibrated', fontsize=12, fontweight='bold')
ax.set_ylim(0.4, 1.1)
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, aurocs):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.015,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.annotate('No calibration\ndata needed!', xy=(0, 1.0), xytext=(1.5, 0.7),
            fontsize=9, color='#4CAF50', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#4CAF50'))

# Panel (b): Per-OOD type (calibration-free attn_max)
ax = axes[1]
ood_types = ['Noise', 'Indoor', 'Inverted', 'Blackout']
attn_ent_aurocs = [1.000, 1.000, 0.933, 1.000]
attn_max_aurocs = [1.000, 1.000, 1.000, 1.000]

x = np.arange(len(ood_types))
width = 0.35
bars1 = ax.bar(x - width/2, attn_ent_aurocs, width, label='Attn Entropy',
               color='#2196F3', edgecolor='black', linewidth=0.5, alpha=0.85)
bars2 = ax.bar(x + width/2, attn_max_aurocs, width, label='Attn Max',
               color='#4CAF50', edgecolor='black', linewidth=0.5, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(ood_types, fontsize=10)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(b) Per-OOD Type (Cal-Free)', fontsize=12, fontweight='bold')
ax.set_ylim(0.8, 1.1)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
ax.annotate('All perfect!', xy=(2, 1.0), xytext=(1.0, 0.88),
            fontsize=10, fontweight='bold', color='#4CAF50')

# Panel (c): Signal comparison (ID vs OOD distributions)
ax = axes[2]
# ID and OOD attention max values
id_max = 0.3114
ood_max = 0.3916
id_ent = 2.4291
ood_ent = 2.0158

signals = ['Attn Max', 'Attn Entropy']
id_vals = [id_max, id_ent]
ood_vals = [ood_max, ood_ent]

x = np.arange(len(signals))
width = 0.35
bars1 = ax.bar(x - width/2, id_vals, width, label='ID (mean)',
               color='#2196F3', edgecolor='black', linewidth=0.5, alpha=0.85)
bars2 = ax.bar(x + width/2, ood_vals, width, label='OOD (mean)',
               color='#F44336', edgecolor='black', linewidth=0.5, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(signals, fontsize=10)
ax.set_ylabel('Value', fontsize=11)
ax.set_title('(c) ID vs OOD Signal Distributions', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
# Annotate the separation
ax.annotate('OOD: higher\nmax attention', xy=(0.17, ood_max), xytext=(0.5, 0.45),
            fontsize=8, color='#F44336',
            arrowprops=dict(arrowstyle='->', color='#F44336'))
ax.annotate('OOD: lower\nentropy', xy=(1.17, ood_ent), xytext=(1.3, 2.6),
            fontsize=8, color='#F44336',
            arrowprops=dict(arrowstyle='->', color='#F44336'))

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig50_attn_calibfree.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig50_attn_calibfree.pdf', dpi=200, bbox_inches='tight')
print("Saved fig50_attn_calibfree.png/pdf")
