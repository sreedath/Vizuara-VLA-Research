"""Generate Figure 30: Action Plausibility as OOD Signal."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/action_plausibility_20260314_164334.json"
OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

with open(RESULTS) as f:
    data = json.load(f)

simple = data['simple_results']
realistic = data['realistic_results']

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# Panel (a): Simple vs Realistic AUROC comparison
ax = axes[0]
from sklearn.metrics import roc_auc_score

# Simple AUROCs
s_easy = [r for r in simple if not r['is_ood']]
s_ood = [r for r in simple if r['is_ood']]
s_labels = [0]*len(s_easy) + [1]*len(s_ood)
s_all = s_easy + s_ood

# Realistic AUROCs
r_easy = [r for r in realistic if not r['is_ood']]
r_ood = [r for r in realistic if r['is_ood']]
r_labels = [0]*len(r_easy) + [1]*len(r_ood)
r_all = r_easy + r_ood

signals = ['Cosine dist', 'Action spread', 'Action mass', 'Roughness', 'Entropy std']
simple_aurocs = [
    roc_auc_score(s_labels, [r['cos_dist'] for r in s_all]),
    roc_auc_score(s_labels, [r['action_spread'] for r in s_all]),
    roc_auc_score(s_labels, [1-r['action_mass'] for r in s_all]),
    roc_auc_score(s_labels, [r['action_roughness'] for r in s_all]),
    roc_auc_score(s_labels, [r['entropy_std'] for r in s_all]),
]
realistic_aurocs = [
    roc_auc_score(r_labels, [r['cos_global'] for r in r_all]),
    roc_auc_score(r_labels, [r['action_spread'] for r in r_all]),
    roc_auc_score(r_labels, [1-r['action_mass'] for r in r_all]),
    roc_auc_score(r_labels, [r['action_roughness'] for r in r_all]),
    roc_auc_score(r_labels, [r['entropy_std'] for r in r_all]),
]

x = np.arange(len(signals))
width = 0.35
bars1 = ax.bar(x - width/2, simple_aurocs, width, label='Simple images',
               color='#2196F3', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, realistic_aurocs, width, label='Realistic images',
               color='#FF9800', edgecolor='black', linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(signals, fontsize=8, rotation=20)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(a) Signal Strength: Simple vs Realistic', fontsize=12, fontweight='bold')
ax.set_ylim(0.3, 1.05)
ax.axhline(y=0.5, color='red', linestyle=':', alpha=0.5)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

for bar, val in zip(list(bars1) + list(bars2), simple_aurocs + realistic_aurocs):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}',
            ha='center', va='bottom', fontsize=7, fontweight='bold')

# Panel (b): Per-scenario action spread (realistic)
ax = axes[1]
scenarios = ['highway_r', 'urban_r', 'snow', 'offroad', 'tunnel']
labels_short = ['highway', 'urban', 'snow', 'offroad', 'tunnel']
colors_sc = ['#2196F3', '#4CAF50', '#e41a1c', '#FF9800', '#9C27B0']

for i, (s, label, color) in enumerate(zip(scenarios, labels_short, colors_sc)):
    s_r = [r for r in realistic if r['scenario'] == s]
    spreads = [r['action_spread'] for r in s_r]
    ax.bar(i, np.mean(spreads), yerr=np.std(spreads), color=color,
           edgecolor='black', linewidth=0.5, alpha=0.8, capsize=3)

ax.set_xticks(range(len(labels_short)))
ax.set_xticklabels(labels_short, fontsize=9, rotation=20)
ax.set_ylabel('Action Spread (std of bin indices)', fontsize=10)
ax.set_title('(b) Action Spread by Scenario', fontsize=12, fontweight='bold')
ax.axvline(x=1.5, color='gray', linestyle='--', alpha=0.5)
ax.text(0.75, ax.get_ylim()[1]*0.9, 'ID', ha='center', fontsize=10,
        fontweight='bold', color='#2196F3')
ax.text(3, ax.get_ylim()[1]*0.9, 'OOD', ha='center', fontsize=10,
        fontweight='bold', color='#e41a1c')
ax.grid(True, alpha=0.3, axis='y')

# Panel (c): Scatter of cosine vs action spread (realistic)
ax = axes[2]
for s, marker, color, label in [
    ('highway_r', 'o', '#2196F3', 'highway (ID)'),
    ('urban_r', 's', '#4CAF50', 'urban (ID)'),
    ('snow', '^', '#e41a1c', 'snow (OOD)'),
    ('offroad', 'D', '#FF9800', 'offroad (OOD)'),
    ('tunnel', 'v', '#9C27B0', 'tunnel (OOD)'),
]:
    s_r = [r for r in realistic if r['scenario'] == s]
    cos_vals = [r['cos_global'] for r in s_r]
    spread_vals = [r['action_spread'] for r in s_r]
    ax.scatter(cos_vals, spread_vals, marker=marker, color=color, s=50,
              alpha=0.8, edgecolors='black', linewidths=0.5, label=label)

ax.set_xlabel('Cosine Distance', fontsize=11)
ax.set_ylabel('Action Spread', fontsize=11)
ax.set_title('(c) Complementary OOD Signals', fontsize=12, fontweight='bold')
ax.legend(fontsize=7, loc='upper left')
ax.grid(True, alpha=0.3)

# Add note about complementarity
ax.annotate('Spread catches\noffroad', xy=(0.63, 42), fontsize=8,
            fontweight='bold', color='#FF9800',
            arrowprops=dict(arrowstyle='->', color='#FF9800', lw=1.5),
            xytext=(0.5, 50))

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig30_action_plausibility.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig30_action_plausibility.pdf', dpi=200, bbox_inches='tight')
print("Saved fig30_action_plausibility.png/pdf")
