"""Generate Figure 35: Safe Fallback Action System."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Panel (a): Action deviation ID vs OOD
ax = axes[0]
scenarios = ['highway', 'urban', 'noise', 'indoor', 'inverted', 'blackout']
is_ood = [False, False, True, True, True, True]
colors_map = {False: '#2196F3', True: '#F44336'}

# Load data
with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/safe_fallback_20260314_171826.json") as f:
    data = json.load(f)

results = data['results']

# Per-scenario stats
for s in scenarios:
    s_r = [r for r in results if r['scenario'] == s]
    devs = [r['action_dev'] for r in s_r]
    ood = s_r[0]['is_ood']
    color = colors_map[ood]
    label = 'OOD' if ood else 'ID'
    ax.bar(s, np.mean(devs), yerr=np.std(devs), color=color,
           edgecolor='black', linewidth=0.5, alpha=0.85, capsize=4)

# Add legend manually
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor='#2196F3', label='In-Distribution'),
                    Patch(facecolor='#F44336', label='Out-of-Distribution')],
          fontsize=9)
ax.set_ylabel('Action Deviation from Safe Default', fontsize=10)
ax.set_title('(a) Action Deviation by Scenario', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0, color='black', linewidth=0.5)

# Add mean values
for i, s in enumerate(scenarios):
    s_r = [r for r in results if r['scenario'] == s]
    mean_dev = np.mean([r['action_dev'] for r in s_r])
    ax.text(i, mean_dev + np.std([r['action_dev'] for r in s_r]) + 1,
            f'{mean_dev:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Panel (b): Safety pipeline at different thresholds
ax = axes[1]
alphas = [0.05, 0.10, 0.20]
thresholds_vals = [data['thresholds']['0.05'], data['thresholds']['0.1'], data['thresholds']['0.2']]

id_results = [r for r in results if not r['is_ood']]
ood_results = [r for r in results if r['is_ood']]
n_id = len(id_results)
n_ood = len(ood_results)

coverages = []
safety_rates = []

for alpha, thresh in zip(alphas, thresholds_vals):
    id_pass = sum(1 for r in id_results if r['cos_dist'] <= thresh)
    ood_caught = sum(1 for r in ood_results if r['cos_dist'] > thresh)
    coverages.append(id_pass / n_id)
    safety_rates.append(ood_caught / n_ood)

x = np.arange(len(alphas))
width = 0.35
bars1 = ax.bar(x - width/2, coverages, width, label='ID Coverage',
               color='#2196F3', edgecolor='black', linewidth=0.5, alpha=0.85)
bars2 = ax.bar(x + width/2, safety_rates, width, label='OOD Safety Rate',
               color='#4CAF50', edgecolor='black', linewidth=0.5, alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels([f'α={a}' for a in alphas], fontsize=11)
ax.set_ylabel('Rate', fontsize=11)
ax.set_title('(b) Safety Pipeline Performance', fontsize=12, fontweight='bold')
ax.set_ylim(0, 1.15)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

for bar, val in zip(list(bars1) + list(bars2), coverages + safety_rates):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# Panel (c): Cosine distance vs action deviation scatter
ax = axes[2]
id_cos = [r['cos_dist'] for r in id_results]
id_dev = [r['action_dev'] for r in id_results]
ood_cos = [r['cos_dist'] for r in ood_results]
ood_dev = [r['action_dev'] for r in ood_results]

ax.scatter(id_cos, id_dev, c='#2196F3', s=50, alpha=0.7, edgecolors='black',
           linewidths=0.5, label='ID', zorder=3)
ax.scatter(ood_cos, ood_dev, c='#F44336', s=50, alpha=0.7, edgecolors='black',
           linewidths=0.5, label='OOD', zorder=3)

# Fit line
all_cos = [r['cos_dist'] for r in results]
all_dev = [r['action_dev'] for r in results]
z = np.polyfit(all_cos, all_dev, 1)
p = np.poly1d(z)
x_line = np.linspace(min(all_cos), max(all_cos), 100)
ax.plot(x_line, p(x_line), 'k--', alpha=0.5, linewidth=1.5, label=f'r = {data["correlation_cos_dev"]:.3f}')

# Threshold line
thresh_10 = data['thresholds']['0.1']
ax.axvline(x=thresh_10, color='green', linestyle=':', alpha=0.7, linewidth=2,
           label=f'Threshold (α=0.10)')

ax.set_xlabel('Cosine Distance', fontsize=11)
ax.set_ylabel('Action Deviation', fontsize=11)
ax.set_title('(c) OOD Detection ↔ Action Quality', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig35_safe_fallback.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig35_safe_fallback.pdf', dpi=200, bbox_inches='tight')
print("Saved fig35_safe_fallback.png/pdf")
