"""Generate Figure 76: Attention Pattern Analysis."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/attention_patterns_20260314_203635.json") as f:
    data = json.load(f)

layers = list(range(32))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel (a): Attention max per layer (ID vs OOD)
ax = axes[0]
for name, color, marker in [('highway', '#4CAF50', 'o'), ('urban', '#2196F3', 's'),
                              ('noise', '#F44336', '^'), ('indoor', '#FF9800', 'D')]:
    profile = data['profiles'][name]['avg_profile']
    maxes = [profile[str(l)]['max'] for l in layers]
    ax.plot(layers, maxes, f'{marker}-', color=color, linewidth=1.5, markersize=4,
            label=name, alpha=0.8)

ax.set_xlabel('Layer', fontsize=11)
ax.set_ylabel('Attention Max', fontsize=11)
ax.set_title('(a) Attention Max by Layer', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel (b): Attention entropy per layer
ax = axes[1]
for name, color, marker in [('highway', '#4CAF50', 'o'), ('urban', '#2196F3', 's'),
                              ('noise', '#F44336', '^'), ('indoor', '#FF9800', 'D')]:
    profile = data['profiles'][name]['avg_profile']
    entropies = [profile[str(l)]['entropy'] for l in layers]
    ax.plot(layers, entropies, f'{marker}-', color=color, linewidth=1.5, markersize=4,
            label=name, alpha=0.8)

ax.set_xlabel('Layer', fontsize=11)
ax.set_ylabel('Attention Entropy', fontsize=11)
ax.set_title('(b) Attention Entropy by Layer', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel (c): ID vs OOD difference (max) per layer
ax = axes[2]
id_maxes = []
ood_maxes = []
for l in layers:
    id_avg = np.mean([data['profiles']['highway']['avg_profile'][str(l)]['max'],
                       data['profiles']['urban']['avg_profile'][str(l)]['max']])
    ood_avg = np.mean([data['profiles']['noise']['avg_profile'][str(l)]['max'],
                        data['profiles']['indoor']['avg_profile'][str(l)]['max']])
    id_maxes.append(id_avg)
    ood_maxes.append(ood_avg)

diff = np.array(ood_maxes) - np.array(id_maxes)
colors = ['#F44336' if d > 0 else '#4CAF50' for d in diff]
ax.bar(layers, diff, 0.8, color=colors, alpha=0.7, edgecolor='black', linewidth=0.3)
ax.set_xlabel('Layer', fontsize=11)
ax.set_ylabel('OOD - ID Attention Max', fontsize=11)
ax.set_title('(c) OOD-ID Attention Difference', fontsize=12, fontweight='bold')
ax.axhline(y=0, color='black', linewidth=0.5)
ax.grid(True, alpha=0.3, axis='y')

# Find layers with largest positive difference
top_layers = np.argsort(diff)[-3:][::-1]
for l in top_layers:
    if diff[l] > 0.01:
        ax.annotate(f'L{l}', xy=(l, diff[l]), xytext=(l, diff[l]+0.01),
                    fontsize=8, ha='center', fontweight='bold', color='darkred')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig76_attention_patterns.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig76_attention_patterns.pdf', dpi=200, bbox_inches='tight')
print("Saved fig76_attention_patterns.png/pdf")
