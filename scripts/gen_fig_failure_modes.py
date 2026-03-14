"""Generate Figure 87: Failure Mode Analysis."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/failure_modes_20260314_212153.json") as f:
    data = json.load(f)

results = data['results']
threshold = data['threshold']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel (a): Score distributions for each scenario
ax = axes[0]
scenarios = ['id_baseline', 'shifted_horizon', 'textured_road', 'red_highway',
             'rotated_highway', 'inverted_highway', 'green_highway']
labels = ['ID\nBaseline', 'Shifted\nHorizon', 'Textured\nRoad', 'Red\nSky',
          'Rotated\n90°', 'Inverted\nColors', 'Green\nSky']
means = [results[s]['mean'] for s in scenarios]
stds = [results[s].get('std', 0) for s in scenarios]
colors = ['#4CAF50'] + ['#F44336'] * 6

bars = ax.bar(range(len(scenarios)), means, yerr=stds, color=colors, alpha=0.7,
              edgecolor='black', linewidth=0.5, capsize=3)
ax.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5,
           label=f'Threshold ({threshold:.4f})')
ax.set_xticks(range(len(scenarios)))
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel('Cosine Distance', fontsize=11)
ax.set_title('(a) Failure Mode Scores', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# Panel (b): Detection rate and margin
ax = axes[1]
ood_scenarios = ['shifted_horizon', 'textured_road', 'red_highway',
                 'rotated_highway', 'inverted_highway', 'green_highway']
ood_labels = ['Shifted\nHorizon', 'Textured\nRoad', 'Red\nSky',
              'Rotated\n90°', 'Inverted\nColors', 'Green\nSky']
margins = [(results[s]['mean'] - threshold) / threshold * 100 for s in ood_scenarios]

colors_m = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(ood_scenarios)))
ax.barh(range(len(ood_scenarios)), margins, color=colors_m, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(ood_scenarios)))
ax.set_yticklabels(ood_labels, fontsize=9)
ax.set_xlabel('Margin above threshold (%)', fontsize=11)
ax.set_title('(b) Detection Margin', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

for i, m in enumerate(margins):
    ax.text(m + 5, i, f'{m:.0f}%', va='center', fontsize=9, fontweight='bold')

# Panel (c): Difficulty ranking with ID reference
ax = axes[2]
all_scenarios = ood_scenarios
all_means = [results[s]['mean'] for s in all_scenarios]
sorted_idx = np.argsort(all_means)

sorted_names = [ood_labels[i] for i in sorted_idx]
sorted_means = [all_means[i] for i in sorted_idx]

colors_rank = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(sorted_names)))
ax.barh(range(len(sorted_names)), sorted_means, color=colors_rank, alpha=0.8,
        edgecolor='black', linewidth=0.5)
ax.axvline(x=threshold, color='red', linestyle='--', linewidth=1.5, label='Threshold')
ax.axvline(x=results['id_baseline']['mean'], color='#4CAF50', linestyle='--',
           linewidth=1.5, label=f'ID mean ({results["id_baseline"]["mean"]:.4f})')
ax.set_yticks(range(len(sorted_names)))
ax.set_yticklabels(sorted_names, fontsize=9)
ax.set_xlabel('Cosine Distance', fontsize=11)
ax.set_title('(c) OOD Difficulty Ranking', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig87_failure_modes.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig87_failure_modes.pdf', dpi=200, bbox_inches='tight')
print("Saved fig87_failure_modes.png/pdf")
