"""Generate Figure 127: Fine-Grained Early Layer Sweep."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/early_layer_sweep_20260315_003403.json") as f:
    data = json.load(f)

results = data['results']
layers = data['layers']
ds = [results[str(l)]['d_prime'] for l in layers]
aurocs = [results[str(l)]['auroc'] for l in layers]
gaps = [results[str(l)]['gap'] for l in layers]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel A: D-prime across all layers
ax = axes[0]
colors = ['#4CAF50' if d > 100 else '#2196F3' if d > 50 else '#FF9800' if d > 30 else '#F44336' for d in ds]
bars = ax.bar(range(len(layers)), ds, color=colors, alpha=0.8)
ax.set_xticks(range(len(layers)))
ax.set_xticklabels([f'L{l}' for l in layers], fontsize=8, rotation=45)
ax.set_ylabel("D-prime (σ)")
ax.set_title("(A) D-prime by Layer")
ax.grid(True, alpha=0.3, axis='y')
# Highlight L3
ax.annotate('L3: d=175.2\n(OPTIMAL)', xy=(2, ds[2]),
            xytext=(5, ds[2]+20), fontsize=9, fontweight='bold', color='green',
            arrowprops=dict(arrowstyle='->', color='green'))

# Panel B: Zoomed early layers (1-10)
ax = axes[1]
early_layers = list(range(1, 11))
early_ds = [results[str(l)]['d_prime'] for l in early_layers]
early_aurocs = [results[str(l)]['auroc'] for l in early_layers]
colors2 = ['#4CAF50' if a == 1.0 else '#F44336' for a in early_aurocs]
bars = ax.bar(range(len(early_layers)), early_ds, color=colors2, alpha=0.8)
ax.set_xticks(range(len(early_layers)))
ax.set_xticklabels([f'L{l}' for l in early_layers], fontsize=9)
ax.set_ylabel("D-prime (σ)")
ax.set_title("(B) Early Layers 1-10 (green=AUROC 1.0)")
ax.grid(True, alpha=0.3, axis='y')
for bar, d, a in zip(bars, early_ds, early_aurocs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f"{d:.0f}", ha='center', fontsize=8)

# Panel C: AUROC heatmap-style
ax = axes[2]
ax.plot(range(len(layers)), aurocs, 'o-', color='#2196F3', linewidth=2, markersize=8)
ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect')
ax.axhline(y=0.5, color='red', linestyle=':', alpha=0.3, label='Random')
ax.fill_between(range(len(layers)), aurocs, 0.5,
                where=[a < 1.0 for a in aurocs], alpha=0.2, color='red')
ax.set_xticks(range(len(layers)))
ax.set_xticklabels([f'L{l}' for l in layers], fontsize=8, rotation=45)
ax.set_ylabel("AUROC")
ax.set_title("(C) Perfect Detection Region")
ax.set_ylim(0.7, 1.02)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
# Mark the valley
ax.annotate('Valley: L8-L16\n(fog overlap)', xy=(7, aurocs[7]),
            xytext=(9, 0.78), fontsize=8, color='red',
            arrowprops=dict(arrowstyle='->', color='red'))

plt.suptitle("Fine-Grained Early Layer Sweep (Exp 141)\nLayer 3 is optimal (d=175.2); layers 1-7 all achieve AUROC=1.000; L8-L16 fail due to fog overlap",
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/latex/figures/fig127_early_layer_sweep.png",
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig127")
