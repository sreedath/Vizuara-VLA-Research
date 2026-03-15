"""Generate Figure 125: Fog OOD Detection Deep Dive."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/fog_deep_dive_20260315_002713.json") as f:
    data = json.load(f)

fog_results = data['fog_results']
opacities = sorted([float(k) for k in fog_results.keys()])
layers = data['layers']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel A: AUROC vs opacity at different layers
ax = axes[0]
layer_colors = {3: '#4CAF50', 8: '#2196F3', 16: '#FF9800', 24: '#9C27B0', 32: '#F44336'}
for layer in layers:
    aurocs = [fog_results[str(op)]['per_layer'][str(layer)]['auroc'] for op in opacities]
    ax.plot(opacities, aurocs, 'o-', color=layer_colors[layer], linewidth=2,
            markersize=6, label=f'Layer {layer}')
ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
ax.axhline(y=0.5, color='red', linestyle=':', alpha=0.3, label='Random')
ax.set_xlabel("Fog Opacity")
ax.set_ylabel("AUROC")
ax.set_title("(A) Detection Quality vs Fog Opacity")
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.3, 1.05)

# Panel B: D-prime by layer for different opacities
ax = axes[1]
selected_opacities = [0.3, 0.5, 0.7, 0.9]
x = np.arange(len(layers))
width = 0.2
for i, op in enumerate(selected_opacities):
    ds = [fog_results[str(op)]['per_layer'][str(l)]['d_prime'] for l in layers]
    ax.bar(x + i*width, ds, width, alpha=0.8, label=f'Fog {int(op*100)}%')
ax.set_xticks(x + width*1.5)
ax.set_xticklabels([f'L{l}' for l in layers], fontsize=9)
ax.set_ylabel("D-prime (σ)")
ax.set_title("(B) Layer Effectiveness for Fog")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# Panel C: Best layer per opacity
ax = axes[2]
best_layers = []
best_ds = []
l32_ds = []
for op in opacities:
    best_l = max(layers, key=lambda l: fog_results[str(op)]['per_layer'][str(l)]['d_prime'])
    best_d = fog_results[str(op)]['per_layer'][str(best_l)]['d_prime']
    l32_d = fog_results[str(op)]['per_layer']['32']['d_prime']
    best_layers.append(best_l)
    best_ds.append(best_d)
    l32_ds.append(l32_d)

ax.plot(opacities, best_ds, 'o-', color='#4CAF50', linewidth=2.5, markersize=8,
        label='Best layer (L3)')
ax.plot(opacities, l32_ds, 's--', color='#F44336', linewidth=2, markersize=7,
        label='Last layer (L32)')
ax.fill_between(opacities, l32_ds, best_ds, alpha=0.15, color='green')
ax.set_xlabel("Fog Opacity")
ax.set_ylabel("D-prime (σ)")
ax.set_title("(C) L3 vs L32 for Fog Detection")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
# Annotate improvement
for op, bd, ld in zip(opacities, best_ds, l32_ds):
    if bd > ld and op in [0.3, 0.5, 0.7]:
        ax.annotate(f'{bd/ld:.1f}×', xy=(op, (bd+ld)/2),
                    fontsize=8, ha='center', color='green')

plt.suptitle("Fog Detection Deep Dive (Exp 139)\nLayer 3 perfectly detects fog from 30% opacity; Layer 32 fails below 40%. L3 is 3.6× better at 50% fog.",
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/latex/figures/fig125_fog_deep_dive.png",
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig125")
