"""Generate Figure 114: Multi-Layer Embedding Fusion."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/multi_layer_fusion_20260314_234756.json") as f:
    data = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel A: Single layer d-prime
ax = axes[0]
layers = [3, 8, 16, 24, 28, 32]
d_vals = [data['single_layer'][str(l)]['d'] for l in layers]
colors = ['#F44336' if l == 3 else '#2196F3' for l in layers]
bars = ax.bar(range(len(layers)), d_vals, color=colors, alpha=0.8)
ax.set_xticks(range(len(layers)))
ax.set_xticklabels([f"L{l}" for l in layers])
ax.set_ylabel("D-prime")
ax.set_title("(A) Single Layer Detection")
ax.grid(True, alpha=0.3, axis='y')
for bar, d in zip(bars, d_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{d:.1f}", ha='center', fontsize=8, fontweight='bold')
bars[0].set_edgecolor('gold')
bars[0].set_linewidth(3)

# Panel B: Fusion strategy comparison
ax = axes[1]
strategies = ['layer_3\n(best single)', 'avg\n(all 6)', 'concat\n(all 6)', 'avg\n(28+32)', 'layer_32\n(last)']
s_d = [113.2, 35.24, 33.05, 30.81, 28.10]
s_colors = ['#F44336', '#4CAF50', '#FF9800', '#9C27B0', '#2196F3']
bars = ax.bar(range(len(strategies)), s_d, color=s_colors, alpha=0.8)
ax.set_xticks(range(len(strategies)))
ax.set_xticklabels(strategies, fontsize=7)
ax.set_ylabel("D-prime")
ax.set_title("(B) Fusion vs Best Single")
ax.grid(True, alpha=0.3, axis='y')
for bar, d in zip(bars, s_d):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{d:.1f}", ha='center', fontsize=8, fontweight='bold')

# Panel C: Layer depth vs d-prime trend
ax = axes[2]
ax.plot(layers, d_vals, 'o-', color='#2196F3', linewidth=2, markersize=8)
ax.fill_between(layers, 0, d_vals, alpha=0.1, color='#2196F3')
ax.set_xlabel("Layer Index")
ax.set_ylabel("D-prime")
ax.set_title("(C) D-prime vs Layer Depth")
ax.grid(True, alpha=0.3)
ax.annotate(f"Layer 3: d={d_vals[0]:.1f}", xy=(3, d_vals[0]),
            xytext=(10, d_vals[0]-20), fontsize=9,
            arrowprops=dict(arrowstyle='->', color='red'), color='red', fontweight='bold')
ax.annotate(f"Layer 32: d={d_vals[-1]:.1f}", xy=(32, d_vals[-1]),
            xytext=(20, d_vals[-1]+20), fontsize=9,
            arrowprops=dict(arrowstyle='->', color='blue'), color='blue')

plt.suptitle("Multi-Layer Fusion (Exp 128)\nLayer 3 alone (d=113.2) outperforms all fusion strategies",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/latex/figures/fig114_multi_layer_fusion.png",
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig114")
