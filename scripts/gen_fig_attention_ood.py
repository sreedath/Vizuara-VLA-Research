"""Generate Figure 103: Attention Pattern OOD Analysis."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/attention_ood_20260314_223709.json") as f:
    data = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel (a): Detector comparison
ax = axes[0]
detectors = ['cosine_baseline', 'max_attention', 'attention_entropy', 'top5_concentration', 'first_quarter_attn']
labels = ['Cosine\n(hidden)', 'Max\nAttention', 'Attention\nEntropy', 'Top-5\nConcentration', 'First-Q\nAttention']
aurocs = [data['detector_results'][d]['auroc'] for d in detectors]

colors = ['#4CAF50' if a >= 1.0 else '#FF9800' if a >= 0.95 else '#F44336' for a in aurocs]
bars = ax.bar(range(len(detectors)), aurocs, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
for bar, auroc in zip(bars, aurocs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{auroc:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_xticks(range(len(detectors)))
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(a) Attention-Based OOD Detectors', fontsize=12, fontweight='bold')
ax.set_ylim(0.9, 1.02)
ax.grid(True, alpha=0.3, axis='y')

# Panel (b): Per-category attention stats
ax = axes[1]
cats = ['highway', 'urban', 'snow', 'indoor', 'twilight', 'noise']
cat_colors = ['#4CAF50', '#4CAF50', '#F44336', '#F44336', '#F44336', '#F44336']
entropies = [data['per_category'][c]['entropy_mean'] for c in cats]
max_attns = [data['per_category'][c]['max_attn_mean'] for c in cats]

x = np.arange(len(cats))
width = 0.35
bars1 = ax.bar(x - width/2, entropies, width, label='Attn Entropy', color='#2196F3', alpha=0.8)
ax2 = ax.twinx()
bars2 = ax2.bar(x + width/2, max_attns, width, label='Max Attn', color='#FF9800', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(cats, fontsize=9)
ax.set_ylabel('Attention Entropy', fontsize=11, color='#2196F3')
ax2.set_ylabel('Max Attention Weight', fontsize=11, color='#FF9800')
ax.set_title('(b) Attention Stats by Category', fontsize=12, fontweight='bold')

# Add ID/OOD markers
for i, c in enumerate(cat_colors):
    ax.text(i, -0.02, 'ID' if c == '#4CAF50' else 'OOD',
            ha='center', va='top', fontsize=7, fontweight='bold', color=c,
            transform=ax.get_xaxis_transform())

lns = [bars1, bars2]
labs = ['Attn Entropy', 'Max Attn']
ax.legend([bars1], ['Entropy'], loc='upper left', fontsize=8)
ax2.legend([bars2], ['Max Attn'], loc='upper right', fontsize=8)

# Panel (c): Per-head AUROC histogram
ax = axes[2]
head_aurocs = data['head_aurocs']
n_heads = len(head_aurocs)
sorted_aurocs = np.sort(head_aurocs)[::-1]

colors_head = ['#4CAF50' if a >= 1.0 else '#FF9800' if a >= 0.9 else '#2196F3' if a >= 0.7 else '#9E9E9E'
               for a in sorted_aurocs]
ax.bar(range(n_heads), sorted_aurocs, color=colors_head, alpha=0.8, edgecolor='black', linewidth=0.3)
ax.axhline(y=1.0, color='green', linestyle='--', linewidth=0.8, alpha=0.5)
ax.axhline(y=0.5, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_xlabel('Head (sorted by AUROC)', fontsize=11)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title(f'(c) Per-Head AUROC ({n_heads} heads)', fontsize=12, fontweight='bold')
ax.set_ylim(0.4, 1.1)
ax.grid(True, alpha=0.3, axis='y')
ax.text(0, 1.02, f'Head 7: AUROC=1.000', fontsize=8, color='green', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig103_attention_ood.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig103_attention_ood.pdf', dpi=200, bbox_inches='tight')
print("Saved fig103_attention_ood.png/pdf")
