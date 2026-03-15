"""Generate Figure 121: Prompt-Conditioned Embedding Geometry."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/prompt_embedding_geometry_20260315_001730.json") as f:
    data = json.load(f)

results = data['results']
prompts = list(results.keys())
sim_matrix = np.array(data['centroid_similarity'])
prompt_labels = [p.replace('_', '\n') for p in prompts]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel A: D-prime by prompt
ax = axes[0]
ds = [results[p]['d_prime'] for p in prompts]
radii = [results[p]['id_radius'] for p in prompts]
sorted_idx = np.argsort(ds)[::-1]
bars = ax.barh(range(len(prompts)), [ds[i] for i in sorted_idx],
               color='#4CAF50', alpha=0.8)
ax.set_yticks(range(len(prompts)))
ax.set_yticklabels([prompt_labels[i] for i in sorted_idx], fontsize=8)
ax.set_xlabel("D-prime (σ)")
ax.set_title("(A) Detection Strength by Prompt")
ax.grid(True, alpha=0.3, axis='x')
for bar, d in zip(bars, [ds[i] for i in sorted_idx]):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f"d={d:.1f}", va='center', fontsize=8)
ax.invert_yaxis()

# Panel B: Centroid similarity heatmap
ax = axes[1]
im = ax.imshow(sim_matrix, cmap='RdYlGn', vmin=0.4, vmax=1.0, aspect='auto')
ax.set_xticks(range(len(prompts)))
ax.set_yticks(range(len(prompts)))
short_labels = [p[:8] for p in prompts]
ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=7)
ax.set_yticklabels(short_labels, fontsize=7)
ax.set_title("(B) Centroid Cosine Similarity")
plt.colorbar(im, ax=ax, shrink=0.8)
# Add text annotations
for i in range(len(prompts)):
    for j in range(len(prompts)):
        color = 'white' if sim_matrix[i,j] > 0.8 or sim_matrix[i,j] < 0.55 else 'black'
        ax.text(j, i, f"{sim_matrix[i,j]:.2f}", ha='center', va='center',
                fontsize=6, color=color)

# Panel C: ID radius vs gap
ax = axes[2]
gaps = [results[p]['gap'] for p in prompts]
ratios = [results[p]['separation_ratio'] for p in prompts]
colors_scatter = plt.cm.Set2(np.linspace(0, 1, len(prompts)))
for i, p in enumerate(prompts):
    ax.scatter(radii[i], gaps[i], s=100, c=[colors_scatter[i]], edgecolors='black',
               linewidth=0.5, zorder=5)
    ax.annotate(p.replace('_', ' '), (radii[i], gaps[i]),
                textcoords="offset points", xytext=(5, 5), fontsize=7)
ax.set_xlabel("ID Cluster Radius")
ax.set_ylabel("ID-OOD Gap")
ax.set_title("(C) Cluster Geometry by Prompt")
ax.grid(True, alpha=0.3)
# Add ratio reference lines
for ratio_val in [2.0, 2.5, 3.0]:
    x = np.linspace(0.07, 0.12, 100)
    ax.plot(x, x * ratio_val, ':', color='gray', alpha=0.3)
    ax.text(0.12, 0.12 * ratio_val, f"ratio={ratio_val}", fontsize=7, color='gray')

plt.suptitle("Prompt-Conditioned Embedding Geometry (Exp 135)\nAll 8 prompts AUROC=1.000; centroids only 61.6% similar; stop↔park↔reverse cluster together",
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/latex/figures/fig121_prompt_embedding_geometry.png",
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig121")
