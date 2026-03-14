"""Generate Figure 97: Action Token Vocabulary Analysis."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/action_vocabulary_20260314_215719.json") as f:
    data = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

dims = sorted(data['dim_results'].keys(), key=lambda k: int(k.split('_')[1]))
dim_ids = [int(d.split('_')[1]) for d in dims]

# Panel (a): Unique tokens per dim — ID vs OOD
ax = axes[0]
id_unique = [data['dim_results'][d]['id_unique'] for d in dims]
ood_unique = [data['dim_results'][d]['ood_unique'] for d in dims]

x = np.arange(len(dims))
width = 0.35
bars1 = ax.bar(x - width/2, id_unique, width, label='ID', color='#4CAF50', alpha=0.8, edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, ood_unique, width, label='OOD', color='#F44336', alpha=0.8, edgecolor='black', linewidth=0.5)
ax.set_xlabel('Action Dimension', fontsize=11)
ax.set_ylabel('Unique Token Count', fontsize=11)
ax.set_title('(a) Vocabulary Size: ID vs OOD', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(dim_ids)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Panel (b): Entropy per dim — ID vs OOD
ax = axes[1]
id_entropy = [data['dim_results'][d]['id_entropy'] for d in dims]
ood_entropy = [data['dim_results'][d]['ood_entropy'] for d in dims]

ax.plot(dim_ids, id_entropy, 'o-', color='#4CAF50', label='ID entropy', linewidth=2, markersize=8)
ax.plot(dim_ids, ood_entropy, 's-', color='#F44336', label='OOD entropy', linewidth=2, markersize=8)
ax.fill_between(dim_ids, id_entropy, ood_entropy, alpha=0.15, color='#FF9800')
ax.set_xlabel('Action Dimension', fontsize=11)
ax.set_ylabel('Shannon Entropy (bits)', fontsize=11)
ax.set_title('(b) Action Token Entropy', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel (c): Category consistency
ax = axes[2]
cats = ['highway', 'urban', 'noise', 'indoor', 'twilight', 'snow']
consistencies = [data['category_consistency'][c]['mean_consistency'] for c in cats]
groups = [data['category_consistency'][c]['group'] for c in cats]
colors = ['#4CAF50' if g == 'ID' else '#F44336' for g in groups]

bars = ax.bar(range(len(cats)), consistencies, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(cats)))
ax.set_xticklabels(cats, rotation=30, ha='right')
ax.set_ylabel('Mean Consistency', fontsize=11)
ax.set_title('(c) Action Consistency by Category', fontsize=12, fontweight='bold')
ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.grid(True, alpha=0.3, axis='y')

# Add ID/OOD legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#4CAF50', alpha=0.8, label='ID'),
                   Patch(facecolor='#F44336', alpha=0.8, label='OOD')]
ax.legend(handles=legend_elements, fontsize=9)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig97_action_vocabulary.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig97_action_vocabulary.pdf', dpi=200, bbox_inches='tight')
print("Saved fig97_action_vocabulary.png/pdf")
