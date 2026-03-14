"""Generate Figure 25: Temporal Trajectory Analysis."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/temporal_trajectory_20260314_160322.json"
OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

with open(RESULTS) as f:
    data = json.load(f)

trajectories = data['trajectories']
threshold = data['threshold']
T = data['trajectory_length']

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Panel (a): Cosine distance over trajectory steps by scenario
ax = axes[0]
scenarios = ['highway', 'urban', 'ood_noise', 'ood_blank', 'ood_indoor', 'ood_inverted', 'ood_blackout']
colors = {
    'highway': '#2196F3', 'urban': '#4CAF50',
    'ood_noise': '#e41a1c', 'ood_blank': '#ff7f00', 'ood_indoor': '#984ea3',
    'ood_inverted': '#a65628', 'ood_blackout': '#333333'
}
styles = {
    'highway': '-', 'urban': '-',
    'ood_noise': '--', 'ood_blank': '--', 'ood_indoor': '--',
    'ood_inverted': '--', 'ood_blackout': '--'
}

for scenario in scenarios:
    s_trajs = [t for t in trajectories if t['scenario'] == scenario]
    step_means = []
    step_stds = []
    for step in range(T):
        vals = [t['steps'][step]['cos_dist'] for t in s_trajs]
        step_means.append(np.mean(vals))
        step_stds.append(np.std(vals))
    label = scenario.replace('ood_', '')
    ax.plot(range(T), step_means, marker='o', color=colors[scenario],
            linestyle=styles[scenario], label=label, linewidth=2, markersize=4)

ax.axhline(y=threshold, color='gray', linestyle=':', linewidth=1.5, label='Threshold')
ax.fill_between(range(T), 0, threshold, alpha=0.1, color='green')
ax.set_xlabel('Trajectory Step', fontsize=11)
ax.set_ylabel('Cosine Distance', fontsize=11)
ax.set_title('(a) Per-Step Cosine Distance', fontsize=12, fontweight='bold')
ax.legend(fontsize=7, loc='center right', ncol=1)
ax.set_xlim(-0.2, T - 0.8)
ax.grid(True, alpha=0.3)

# Panel (b): Cumulative AUROC by number of steps
ax = axes[1]
easy_trajs = [t for t in trajectories if not t['is_ood']]
ood_trajs = [t for t in trajectories if t['is_ood']]
labels = [0] * len(easy_trajs) + [1] * len(ood_trajs)
all_trajs = easy_trajs + ood_trajs

from sklearn.metrics import roc_auc_score

aurocs = []
ood_flag_rates = []
easy_flag_rates = []
for n_steps in range(1, T + 1):
    scores = [np.mean([t['steps'][s]['cos_dist'] for s in range(n_steps)])
              for t in all_trajs]
    aurocs.append(roc_auc_score(labels, scores))
    ood_scores = scores[len(easy_trajs):]
    easy_scores = scores[:len(easy_trajs)]
    ood_flag_rates.append(sum(1 for s in ood_scores if s > threshold) / len(ood_scores) * 100)
    easy_flag_rates.append(sum(1 for s in easy_scores if s > threshold) / len(easy_scores) * 100)

ax.plot(range(1, T + 1), aurocs, 'b-o', linewidth=2.5, markersize=8, label='AUROC', zorder=3)
ax.plot(range(1, T + 1), [r/100 for r in ood_flag_rates], 'r--s', linewidth=2,
        markersize=6, label='OOD Flag Rate')
ax.plot(range(1, T + 1), [r/100 for r in easy_flag_rates], 'g--^', linewidth=2,
        markersize=6, label='Easy FP Rate')

ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Number of Steps Used', fontsize=11)
ax.set_ylabel('Rate', fontsize=11)
ax.set_title('(b) Cumulative Detection', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='center right')
ax.set_xlim(0.8, T + 0.2)
ax.set_ylim(-0.05, 1.1)
ax.grid(True, alpha=0.3)

# Annotate key points
ax.annotate(f'{aurocs[0]:.3f}', (1, aurocs[0]), textcoords="offset points",
            xytext=(10, -15), fontsize=9, fontweight='bold')
ax.annotate('1.000', (3, 1.0), textcoords="offset points",
            xytext=(10, -10), fontsize=9, fontweight='bold', color='blue')

# Panel (c): Within-trajectory variance
ax = axes[2]
scenario_names = []
mean_stds = []
colors_bar = []
for scenario in scenarios:
    s_trajs = [t for t in trajectories if t['scenario'] == scenario]
    variances = [np.std([s['cos_dist'] for s in t['steps']]) for t in s_trajs]
    scenario_names.append(scenario.replace('ood_', ''))
    mean_stds.append(np.mean(variances))
    colors_bar.append(colors[scenario])

bars = ax.bar(range(len(scenario_names)), mean_stds, color=colors_bar,
              edgecolor='black', linewidth=0.5, alpha=0.8)
ax.set_xticks(range(len(scenario_names)))
ax.set_xticklabels(scenario_names, fontsize=9, rotation=30)
ax.set_ylabel('Within-Trajectory Std', fontsize=11)
ax.set_title('(c) Temporal Stability', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add ID/OOD labels
ax.axvline(x=1.5, color='gray', linestyle='--', alpha=0.5)
ax.text(0.5, max(mean_stds) * 0.95, 'ID', ha='center', fontsize=10,
        fontweight='bold', color='#2196F3')
ax.text(4, max(mean_stds) * 0.95, 'OOD', ha='center', fontsize=10,
        fontweight='bold', color='#e41a1c')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig25_temporal_trajectory.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig25_temporal_trajectory.pdf', dpi=200, bbox_inches='tight')
print("Saved fig25_temporal_trajectory.png/pdf")
