"""Generate Figure 29: Temporal Trajectory with Realistic Images."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/temporal_realistic_20260314_164011.json"
OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

with open(RESULTS) as f:
    data = json.load(f)

traj = data['trajectories']
id_traj = [t for t in traj if not t['is_ood']]
ood_traj = [t for t in traj if t['is_ood']]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# Panel (a): AUROC vs window size
ax = axes[0]
windows = [1, 2, 3, 5, 8]
global_aurocs = []
perscene_aurocs = []

from sklearn.metrics import roc_auc_score
labels = [0]*len(id_traj) + [1]*len(ood_traj)
all_t = id_traj + ood_traj

for w in windows:
    global_scores = []
    perscene_scores = []
    for t in all_t:
        g_vals = [s['cos_global'] for s in t['steps'][:w]]
        p_vals = [s['cos_per_scene'] for s in t['steps'][:w]]
        global_scores.append(np.mean(g_vals))
        perscene_scores.append(np.mean(p_vals))
    global_aurocs.append(roc_auc_score(labels, global_scores))
    perscene_aurocs.append(roc_auc_score(labels, perscene_scores))

ax.plot(windows, global_aurocs, 'o-', color='#FF9800', linewidth=2, markersize=8,
        label='Global centroid', zorder=5)
ax.plot(windows, perscene_aurocs, 's-', color='#2196F3', linewidth=2, markersize=8,
        label='Per-scene min', zorder=5)
ax.axhline(y=0.984, color='green', linestyle='--', alpha=0.5, label='Simple img (0.984)')
ax.axhline(y=0.767, color='gray', linestyle=':', alpha=0.5, label='Exp 41 single-frame (0.767)')
ax.set_xlabel('Trajectory Window (steps)', fontsize=11)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(a) Temporal Aggregation Improves Detection', fontsize=12, fontweight='bold')
ax.set_xticks(windows)
ax.set_ylim(0.45, 1.02)
ax.legend(fontsize=8, loc='center right')
ax.grid(True, alpha=0.3)

for w, g, p in zip(windows, global_aurocs, perscene_aurocs):
    ax.annotate(f'{p:.3f}', (w, p), textcoords="offset points",
                xytext=(0, 10), fontsize=8, ha='center', fontweight='bold', color='#2196F3')

# Panel (b): Per-OOD type comparison (single vs 8-step per-scene)
ax = axes[1]
ood_types = ['snow', 'flooded', 'offroad', 'tunnel']

single_aurocs = []
temporal_aurocs = []

for ood_type in ood_types:
    type_ood = [t for t in ood_traj if t['scene'] == ood_type]
    type_labels = [0]*len(id_traj) + [1]*len(type_ood)
    type_all = id_traj + type_ood

    # Single frame
    single_scores = [t['steps'][0]['cos_per_scene'] for t in type_all]
    single_aurocs.append(roc_auc_score(type_labels, single_scores))

    # 8-step mean
    temp_scores = [np.mean([s['cos_per_scene'] for s in t['steps']]) for t in type_all]
    temporal_aurocs.append(roc_auc_score(type_labels, temp_scores))

x = np.arange(len(ood_types))
width = 0.35
bars1 = ax.bar(x - width/2, single_aurocs, width, label='Single frame', color='#BBDEFB',
               edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, temporal_aurocs, width, label='8-step mean', color='#2196F3',
               edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(ood_types, fontsize=10)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(b) Per-OOD Type: Single vs Temporal', fontsize=12, fontweight='bold')
ax.set_ylim(0.2, 1.05)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

for bar, val in zip(list(bars1) + list(bars2), single_aurocs + temporal_aurocs):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# Panel (c): Per-step cosine distance (ID vs OOD)
ax = axes[2]
steps = range(8)
id_per_step = [np.mean([t['steps'][s]['cos_per_scene'] for t in id_traj]) for s in steps]
ood_per_step = [np.mean([t['steps'][s]['cos_per_scene'] for t in ood_traj]) for s in steps]
id_std = [np.std([t['steps'][s]['cos_per_scene'] for t in id_traj]) for s in steps]
ood_std = [np.std([t['steps'][s]['cos_per_scene'] for t in ood_traj]) for s in steps]

ax.fill_between(steps, [m-s for m,s in zip(id_per_step, id_std)],
                [m+s for m,s in zip(id_per_step, id_std)], alpha=0.2, color='#2196F3')
ax.fill_between(steps, [m-s for m,s in zip(ood_per_step, ood_std)],
                [m+s for m,s in zip(ood_per_step, ood_std)], alpha=0.2, color='#e41a1c')
ax.plot(steps, id_per_step, 'o-', color='#2196F3', linewidth=2, label='ID (mean ± std)')
ax.plot(steps, ood_per_step, 's-', color='#e41a1c', linewidth=2, label='OOD (mean ± std)')
ax.set_xlabel('Time Step', fontsize=11)
ax.set_ylabel('Per-Scene Min Cosine Distance', fontsize=11)
ax.set_title('(c) Temporal Cosine Trajectories', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig29_temporal_realistic.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig29_temporal_realistic.pdf', dpi=200, bbox_inches='tight')
print("Saved fig29_temporal_realistic.png/pdf")
