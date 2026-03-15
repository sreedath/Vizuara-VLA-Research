"""Generate Figure 120: Embedding Drift Under Input Perturbation."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/embedding_drift_20260315_001446.json") as f:
    data = json.load(f)

ood_ref = data['ood_reference_distance']
results = data['results']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel A: All perturbation curves
ax = axes[0]
colors = {
    'brightness': '#FF9800',
    'contrast': '#2196F3',
    'gaussian_noise': '#F44336',
    'blur': '#4CAF50',
    'occlusion': '#9C27B0',
    'color_jitter': '#795548',
}
for pert_name, pert_data in results.items():
    levels = [d['level'] for d in pert_data['data']]
    fracs = [d['fraction_of_ood'] * 100 for d in pert_data['data']]
    ax.plot(range(len(levels)), fracs, 'o-', color=colors[pert_name],
            linewidth=2, markersize=5, label=pert_name.replace('_', ' ').title())
    ax.set_xticks(range(len(levels)))

ax.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='OOD distance')
ax.axhline(y=50, color='orange', linestyle=':', alpha=0.5, label='50% of OOD')
ax.set_xlabel("Perturbation Level (increasing severity →)")
ax.set_ylabel("% of OOD Reference Distance")
ax.set_title("(A) Embedding Drift by Perturbation Type")
ax.legend(fontsize=7, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 110)

# Panel B: Maximum drift per perturbation
ax = axes[1]
max_drifts = []
pert_names = []
for pert_name, pert_data in results.items():
    max_frac = max(d['fraction_of_ood'] for d in pert_data['data'])
    max_drifts.append(max_frac * 100)
    pert_names.append(pert_name.replace('_', ' ').title())

sorted_idx = np.argsort(max_drifts)[::-1]
pert_names_sorted = [pert_names[i] for i in sorted_idx]
max_drifts_sorted = [max_drifts[i] for i in sorted_idx]
bar_colors = ['#F44336' if d > 80 else '#FF9800' if d > 50 else '#4CAF50' for d in max_drifts_sorted]

bars = ax.barh(range(len(pert_names_sorted)), max_drifts_sorted, color=bar_colors, alpha=0.8)
ax.axvline(x=100, color='red', linestyle='--', alpha=0.5)
ax.set_yticks(range(len(pert_names_sorted)))
ax.set_yticklabels(pert_names_sorted, fontsize=9)
ax.set_xlabel("Max Drift (% of OOD distance)")
ax.set_title("(B) Maximum Drift at Extreme Levels")
ax.grid(True, alpha=0.3, axis='x')
for bar, d in zip(bars, max_drifts_sorted):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
            f"{d:.0f}%", va='center', fontsize=9)
ax.invert_yaxis()

# Panel C: Sensitivity ranking at moderate levels
ax = axes[2]
# Pick moderate perturbation levels
moderate = {
    'brightness': 0.5,  # factor 0.5
    'contrast': 0.5,
    'gaussian_noise': 30,
    'blur': 3,
    'occlusion': 0.1,
    'color_jitter': 40,
}
mod_dists = []
mod_names = []
for pert_name, mod_level in moderate.items():
    for d in results[pert_name]['data']:
        if d['level'] == mod_level:
            mod_dists.append(d['mean_distance'])
            mod_names.append(f"{pert_name.replace('_', ' ').title()}\n(level={mod_level})")
            break

sorted_idx = np.argsort(mod_dists)[::-1]
mod_names_sorted = [mod_names[i] for i in sorted_idx]
mod_dists_sorted = [mod_dists[i] for i in sorted_idx]

bars = ax.bar(range(len(mod_names_sorted)), mod_dists_sorted,
              color=['#F44336' if d > 0.2 else '#FF9800' if d > 0.1 else '#4CAF50' for d in mod_dists_sorted],
              alpha=0.8)
ax.axhline(y=ood_ref, color='red', linestyle='--', alpha=0.5, label=f'OOD ref ({ood_ref:.3f})')
ax.set_xticks(range(len(mod_names_sorted)))
ax.set_xticklabels(mod_names_sorted, fontsize=7)
ax.set_ylabel("Cosine Distance from ID Centroid")
ax.set_title("(C) Drift at Moderate Perturbation")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')
for bar, d in zip(bars, mod_dists_sorted):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{d:.3f}", ha='center', fontsize=8)

plt.suptitle("Embedding Drift Under Input Perturbation (Exp 134)\nOcclusion causes largest drift (96% of OOD); color jitter causes least (25%)",
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/latex/figures/fig120_embedding_drift.png",
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig120")
