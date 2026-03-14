"""Generate Figure 31: Optimal Realistic OOD Detection."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

RESULTS = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/optimal_realistic_20260314_164926.json"
OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

with open(RESULTS) as f:
    data = json.load(f)

traj = data['trajectories']
id_t = [t for t in traj if not t['is_ood']]
ood_t = [t for t in traj if t['is_ood']]
labels = [0]*len(id_t) + [1]*len(ood_t)
all_t = id_t + ood_t

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# Panel (a): Method progression showing improvement
ax = axes[0]
progression = [
    ('Global cos\n(single)', 0.543),
    ('Global cos\n(temporal)', 0.625),
    ('Per-scene cos\n(single)', 0.767),
    ('Per-scene cos\n(temporal)', 0.875),
    ('0.7cos+0.3mass\n(temporal)', 0.917),
]
names = [p[0] for p in progression]
vals = [p[1] for p in progression]
colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(vals)))

bars = ax.bar(range(len(names)), vals, color=colors, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, fontsize=7)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(a) Progressive Improvement', fontsize=12, fontweight='bold')
ax.set_ylim(0.4, 1.0)
ax.axhline(y=0.984, color='green', linestyle='--', alpha=0.5, label='Simple img (0.984)')
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.3f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# Draw improvement arrows
for i in range(len(vals)-1):
    diff = vals[i+1] - vals[i]
    if diff > 0:
        ax.annotate('', xy=(i+1, vals[i+1]-0.01), xytext=(i, vals[i]+0.02),
                    arrowprops=dict(arrowstyle='->', color='green', lw=1.5, alpha=0.5))

# Panel (b): Per-OOD type comparison of single cosine vs optimal combo
ax = axes[1]
ood_types = ['snow', 'flooded', 'offroad', 'tunnel']

def traj_score(traj_list, key):
    return [np.mean([s[key] for s in t['steps']]) for t in traj_list]

def norm01(arr):
    mn, mx = min(arr), max(arr)
    if mx - mn < 1e-10:
        return [0]*len(arr)
    return [(v - mn) / (mx - mn) for v in arr]

cos_scores = traj_score(all_t, 'cos_per_scene')
mass_scores = [1 - v for v in traj_score(all_t, 'action_mass')]
cos_n = norm01(cos_scores)
mass_n = norm01(mass_scores)
combo_scores = [0.7*c + 0.3*m for c, m in zip(cos_n, mass_n)]

cos_per_type = []
combo_per_type = []
for ood_type in ood_types:
    type_ood_idx = [i for i, t in enumerate(all_t) if t['is_ood'] and t['scene'] == ood_type]
    type_id_idx = [i for i, t in enumerate(all_t) if not t['is_ood']]
    type_labels = [0]*len(type_id_idx) + [1]*len(type_ood_idx)

    type_cos = [cos_scores[i] for i in type_id_idx] + [cos_scores[i] for i in type_ood_idx]
    type_combo = [combo_scores[i] for i in type_id_idx] + [combo_scores[i] for i in type_ood_idx]

    cos_per_type.append(roc_auc_score(type_labels, type_cos))
    combo_per_type.append(roc_auc_score(type_labels, type_combo))

x = np.arange(len(ood_types))
width = 0.35
bars1 = ax.bar(x - width/2, cos_per_type, width, label='Per-scene cosine',
               color='#BBDEFB', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, combo_per_type, width, label='Optimal combo',
               color='#2196F3', edgecolor='black', linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(ood_types, fontsize=10)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(b) Per-OOD Type: Cosine vs Optimal', fontsize=12, fontweight='bold')
ax.set_ylim(0.2, 1.05)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

for bar, val in zip(list(bars1) + list(bars2), cos_per_type + combo_per_type):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# Panel (c): Weight sensitivity heatmap
ax = axes[2]
# Recompute for different weight combinations
weights_cos = np.arange(0, 1.05, 0.1)
weights_mass = 1 - weights_cos
aurocs_sweep = []

for wc in weights_cos:
    wm = 1 - wc
    combo = [wc * c + wm * m for c, m in zip(cos_n, mass_n)]
    aurocs_sweep.append(roc_auc_score(labels, combo))

ax.plot(weights_cos, aurocs_sweep, 'o-', color='#2196F3', linewidth=2, markersize=8)
ax.axvline(x=0.7, color='red', linestyle='--', alpha=0.7, label='Optimal (w=0.7)')
ax.set_xlabel('Cosine Weight (1-w = Mass Weight)', fontsize=11)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(c) Weight Sensitivity', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Mark max
max_idx = np.argmax(aurocs_sweep)
ax.annotate(f'{aurocs_sweep[max_idx]:.3f}', (weights_cos[max_idx], aurocs_sweep[max_idx]),
            textcoords="offset points", xytext=(10, 10), fontsize=10,
            fontweight='bold', color='red',
            arrowprops=dict(arrowstyle='->', color='red'))

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig31_optimal_realistic.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig31_optimal_realistic.pdf', dpi=200, bbox_inches='tight')
print("Saved fig31_optimal_realistic.png/pdf")
