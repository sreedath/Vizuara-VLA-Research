"""Generate Figure 45: Temporal Autocorrelation of OOD Scores."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/temporal_autocorr_20260314_181413.json") as f:
    data = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# Panel (a): Trajectory score traces
ax = axes[0]
colors_map = {'highway': '#2196F3', 'urban': '#4CAF50', 'noise': '#F44336',
              'indoor': '#FF9800', 'transition': '#9C27B0'}
for traj in data['trajectories']:
    color = colors_map.get(traj['type'], '#666666')
    linestyle = '--' if traj['is_ood'] and traj['type'] != 'transition' else '-'
    if traj['type'] == 'transition':
        linestyle = ':'
    alpha = 0.6 if traj['type'] in ['highway', 'urban'] else 0.8
    ax.plot(range(len(traj['scores'])), traj['scores'], color=color,
            linestyle=linestyle, alpha=alpha, linewidth=1.5)

# Legend entries
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='#2196F3', label='Highway (ID)', linewidth=2),
    Line2D([0], [0], color='#4CAF50', label='Urban (ID)', linewidth=2),
    Line2D([0], [0], color='#F44336', linestyle='--', label='Noise (OOD)', linewidth=2),
    Line2D([0], [0], color='#FF9800', linestyle='--', label='Indoor (OOD)', linewidth=2),
    Line2D([0], [0], color='#9C27B0', linestyle=':', label='Transition', linewidth=2),
]
ax.legend(handles=legend_elements, fontsize=7, loc='upper left')
ax.set_xlabel('Step in Trajectory', fontsize=11)
ax.set_ylabel('Cosine Distance Score', fontsize=11)
ax.set_title('(a) Score Traces Across Trajectories', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Panel (b): Detection improvement with temporal context
ax = axes[1]
methods = ['Per-Step\n(raw)', '3-Step\nWindow', 'Traj Mean\n(8 steps)', 'EMA\n(α=0.5)']
aurocs = [0.953, 0.996, 1.000, 1.000]
colors = ['#FF9800', '#4CAF50', '#2196F3', '#9C27B0']

bars = ax.bar(range(len(methods)), aurocs, 0.6, color=colors,
              edgecolor='black', linewidth=0.5, alpha=0.85)
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods, fontsize=9)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(b) Temporal Aggregation Improves Detection', fontsize=12, fontweight='bold')
ax.set_ylim(0.9, 1.01)
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, aurocs):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Panel (c): Transition detection
ax = axes[2]
trans_trajs = [t for t in data['trajectories'] if t['type'] == 'transition']
for i, traj in enumerate(trans_trajs):
    ax.plot(range(len(traj['scores'])), traj['scores'], 'o-',
            color=f'C{i}', linewidth=1.5, markersize=6, alpha=0.8,
            label=f'Traj {i}')

# Mark transition zone
ax.axvspan(2.5, 4.5, alpha=0.15, color='red', label='Transition zone')
ax.axhline(y=np.mean([np.mean(t['scores'][:3]) for t in trans_trajs]),
           color='blue', linestyle='--', alpha=0.5, label='ID baseline')
ax.set_xlabel('Step in Trajectory', fontsize=11)
ax.set_ylabel('Cosine Distance Score', fontsize=11)
ax.set_title('(c) ID→OOD Transition Detection', fontsize=12, fontweight='bold')
ax.legend(fontsize=7, loc='upper left')
ax.grid(True, alpha=0.3)
ax.annotate(f'Mean jump:\n+{data["transition_jump"]["mean"]:.3f}',
            xy=(6, 0.8), fontsize=9, fontweight='bold', color='red',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig45_temporal_autocorr.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig45_temporal_autocorr.pdf', dpi=200, bbox_inches='tight')
print("Saved fig45_temporal_autocorr.png/pdf")
