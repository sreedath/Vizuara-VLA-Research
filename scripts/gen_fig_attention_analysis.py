"""Generate Figure 27: Attention Pattern Analysis."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/attention_analysis_20260314_162053.json"
OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

with open(RESULTS) as f:
    data = json.load(f)

results = data['results']
easy = [r for r in results if r['difficulty'] == 'easy']
ood = [r for r in results if r['difficulty'] == 'ood']

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Panel (a): Per-scenario attention entropy
ax = axes[0]
scenarios = ['highway', 'urban', 'ood_noise', 'ood_blank', 'ood_indoor', 'ood_inverted', 'ood_blackout']
labels_s = ['highway', 'urban', 'noise', 'blank', 'indoor', 'inverted', 'blackout']
colors_s = ['#2196F3', '#4CAF50', '#e41a1c', '#ff7f00', '#984ea3', '#a65628', '#333333']

means = []
stds = []
for s in scenarios:
    s_results = [r for r in results if r['scenario'] == s]
    means.append(np.mean([r['mean_attn_entropy'] for r in s_results]))
    stds.append(np.std([r['mean_attn_entropy'] for r in s_results]))

bars = ax.bar(range(len(labels_s)), means, yerr=stds, color=colors_s,
              edgecolor='black', linewidth=0.5, alpha=0.8, capsize=3)
ax.set_xticks(range(len(labels_s)))
ax.set_xticklabels(labels_s, fontsize=9, rotation=30)
ax.set_ylabel('Mean Attention Entropy', fontsize=11)
ax.set_title('(a) Attention Entropy by Scenario', fontsize=12, fontweight='bold')
ax.axvline(x=1.5, color='gray', linestyle='--', alpha=0.5)
ax.text(0.5, max(means) * 0.95, 'ID', ha='center', fontsize=10, fontweight='bold', color='#2196F3')
ax.text(4, max(means) * 0.95, 'OOD', ha='center', fontsize=10, fontweight='bold', color='#e41a1c')
ax.grid(True, alpha=0.3, axis='y')

# Panel (b): AUROC comparison
ax = axes[1]
methods = {
    'Cosine dist': 0.982,
    'Attn ent std': 0.830,
    'Action mass': 0.698,
    'Image attn': 0.217,
    'Mean attn ent': 0.145,
    'Top-5 attn': 0.095,
}
# For the inverted ones, show the correct AUROC
# (lower entropy = OOD, so 1-AUROC for those measured in wrong direction)
corrected = {
    'Cosine dist': 0.982,
    'Top-5 attn': 1 - 0.095,  # 0.905
    'Max attn': 1 - 0.113,  # 0.887
    'Lower attn ent': 1 - 0.145,  # 0.855
    'Attn ent std': 0.830,
    'Action mass': 1 - 0.698,  # flip because negated
}
# Actually let me just use the raw correct values
auroc_data = [
    ('Cosine dist', 0.982),
    ('Top-5 attn*', 0.905),
    ('Max attn*', 0.887),
    ('Low attn ent*', 0.855),
    ('Attn ent std', 0.830),
    ('Action mass', 0.698),
]
names = [d[0] for d in auroc_data]
vals = [d[1] for d in auroc_data]
colors_bar = ['#2196F3' if v > 0.9 else '#64B5F6' if v > 0.8 else '#90CAF9' if v > 0.7 else '#BBDEFB' for v in vals]

bars = ax.barh(range(len(names)), vals, color=colors_bar, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=9)
ax.set_xlabel('AUROC', fontsize=11)
ax.set_title('(b) OOD Detection: Attn vs Cosine', fontsize=12, fontweight='bold')
ax.set_xlim(0.5, 1.02)
ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')
ax.text(0.52, 5.5, '* sign-corrected', fontsize=7, style='italic', color='gray')

for bar, val in zip(bars, vals):
    ax.text(val - 0.02, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', ha='right', va='center', fontsize=9, fontweight='bold', color='white')

# Panel (c): Scatter: cosine distance vs attention entropy
ax = axes[2]
cos_easy = [r['cos_dist'] for r in easy]
ent_easy = [r['mean_attn_entropy'] for r in easy]
cos_ood = [r['cos_dist'] for r in ood]
ent_ood = [r['mean_attn_entropy'] for r in ood]

ax.scatter(cos_easy, ent_easy, color='#2196F3', alpha=0.7, s=40,
           label='In-distribution', marker='o', edgecolors='black', linewidths=0.5)
ax.scatter(cos_ood, ent_ood, color='#e41a1c', alpha=0.7, s=40,
           label='OOD', marker='^', edgecolors='black', linewidths=0.5)

# Correlation line
all_cos = cos_easy + cos_ood
all_ent = ent_easy + ent_ood
z = np.polyfit(all_cos, all_ent, 1)
p = np.poly1d(z)
x_line = np.linspace(min(all_cos), max(all_cos), 100)
ax.plot(x_line, p(x_line), 'k--', linewidth=1.5, alpha=0.5)

r = np.corrcoef(all_cos, all_ent)[0, 1]
ax.set_xlabel('Cosine Distance', fontsize=11)
ax.set_ylabel('Attention Entropy', fontsize=11)
ax.set_title(f'(c) Cosine vs Attn Entropy (r={r:.2f})', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='lower left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig27_attention_analysis.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig27_attention_analysis.pdf', dpi=200, bbox_inches='tight')
print("Saved fig27_attention_analysis.png/pdf")
