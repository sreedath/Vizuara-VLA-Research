"""Generate Figure 28: Improved Realistic Image OOD Detection."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/improved_realistic_20260314_163425.json"
OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

with open(RESULTS) as f:
    data = json.load(f)

results = data['results']
easy = [r for r in results if not r['is_ood']]
ood = [r for r in results if r['is_ood']]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# Panel (a): Method comparison bar chart
ax = axes[0]
methods = [
    ('Per-scene\nmin', 0.767, '#2196F3'),
    ('Combined\n(cos+mass+ent)', 0.670, '#4CAF50'),
    ('Global\ncentroid', 0.611, '#FF9800'),
    ('Norm-\naware', 0.607, '#9C27B0'),
    ('Action\nmass', 0.589, '#F44336'),
    ('Z-scored\ncosine', 0.500, '#795548'),
]
names = [m[0] for m in methods]
vals = [m[1] for m in methods]
colors = [m[2] for m in methods]

bars = ax.bar(range(len(names)), vals, color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, fontsize=8)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(a) Detection Methods (Realistic Images)', fontsize=12, fontweight='bold')
ax.set_ylim(0.4, 0.85)
ax.axhline(y=0.5, color='red', linestyle=':', alpha=0.5, label='Random')
ax.axhline(y=0.984, color='green', linestyle='--', alpha=0.5, label='Simple img (0.984)')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.3f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# Panel (b): Per-OOD type heatmap
ax = axes[1]
ood_types = ['snow_road', 'flooded_road', 'offroad', 'tunnel']
method_names = ['Global', 'Per-scene', 'Norm-aware', 'Combined']
heatmap_data = np.array([
    [0.819, 0.743, 0.458, 0.424],  # Global
    [0.917, 0.868, 0.642, 0.642],  # Per-scene
    [0.816, 0.691, 0.476, 0.444],  # Norm-aware
    [0.760, 0.792, 0.497, 0.632],  # Combined
])

im = ax.imshow(heatmap_data, cmap='RdYlGn', vmin=0.3, vmax=1.0, aspect='auto')
ax.set_xticks(range(4))
ax.set_xticklabels(['snow', 'flooded', 'offroad', 'tunnel'], fontsize=9, rotation=20)
ax.set_yticks(range(4))
ax.set_yticklabels(method_names, fontsize=9)
ax.set_title('(b) Per-OOD Type AUROC', fontsize=12, fontweight='bold')

for i in range(4):
    for j in range(4):
        color = 'white' if heatmap_data[i, j] < 0.6 else 'black'
        ax.text(j, i, f'{heatmap_data[i, j]:.2f}', ha='center', va='center',
                fontsize=10, fontweight='bold', color=color)

plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Panel (c): Per-scenario mean cosine distance
ax = axes[2]
scenarios_id = ['highway_realistic', 'urban_realistic', 'night_driving', 'foggy_road']
scenarios_ood = ['snow_road', 'flooded_road', 'offroad', 'tunnel']
all_scenarios = scenarios_id + scenarios_ood
labels_short = ['highway', 'urban', 'night', 'foggy', 'snow', 'flooded', 'offroad', 'tunnel']

mean_global = []
mean_perscene = []
for s in all_scenarios:
    s_r = [r for r in results if r['scenario'] == s]
    mean_global.append(np.mean([r['cos_global'] for r in s_r]))
    mean_perscene.append(np.mean([r['cos_per_scene'] for r in s_r]))

x = np.arange(len(all_scenarios))
width = 0.35
colors_bar = ['#2196F3'] * 4 + ['#e41a1c'] * 4

ax.bar(x - width/2, mean_global, width, label='Global centroid', alpha=0.7,
       edgecolor='black', linewidth=0.5, color=['#64B5F6']*4 + ['#EF9A9A']*4)
ax.bar(x + width/2, mean_perscene, width, label='Per-scene min', alpha=0.7,
       edgecolor='black', linewidth=0.5, color=['#2196F3']*4 + ['#e41a1c']*4)

ax.set_xticks(x)
ax.set_xticklabels(labels_short, fontsize=8, rotation=30)
ax.set_ylabel('Mean Cosine Distance', fontsize=11)
ax.set_title('(c) ID vs OOD Score Overlap', fontsize=12, fontweight='bold')
ax.axvline(x=3.5, color='gray', linestyle='--', alpha=0.5)
ax.text(1.5, max(mean_global) * 0.95, 'ID', ha='center', fontsize=10,
        fontweight='bold', color='#2196F3')
ax.text(5.5, max(mean_global) * 0.95, 'OOD', ha='center', fontsize=10,
        fontweight='bold', color='#e41a1c')
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig28_improved_realistic.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig28_improved_realistic.pdf', dpi=200, bbox_inches='tight')
print("Saved fig28_improved_realistic.png/pdf")
