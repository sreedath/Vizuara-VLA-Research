"""Generate Figure 68: Action Prediction Consistency."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

scenarios = ['highway', 'urban', 'noise', 'indoor', 'twilight', 'snow']
is_ood = [False, False, True, True, True, True]
unique_tokens = [1, 1, 3, 3, 2, 5]
agreements = [1.00, 1.00, 0.80, 0.70, 0.90, 0.60]
entropies = [1.590, 1.827, 1.911, 1.912, 1.168, 2.411]
top_probs = [0.647, 0.574, 0.441, 0.469, 0.754, 0.312]
cos_dists = [0.083, 0.094, 0.435, 0.351, 0.440, 0.269]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

colors = ['#4CAF50' if not ood else '#F44336' for ood in is_ood]

# Panel (a): Token agreement rate
ax = axes[0]
bars = ax.bar(range(len(scenarios)), agreements, 0.6, color=colors,
              edgecolor='black', linewidth=0.5, alpha=0.85)
ax.set_xticks(range(len(scenarios)))
ax.set_xticklabels(scenarios, fontsize=10, rotation=30, ha='right')
ax.set_ylabel('Token Agreement Rate', fontsize=11)
ax.set_title('(a) Action Consistency', fontsize=12, fontweight='bold')
ax.set_ylim(0.4, 1.1)
ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.3)
ax.grid(True, alpha=0.3, axis='y')

for bar, v, u in zip(bars, agreements, unique_tokens):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
            f'{v:.0%}\n({u} tok)', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.annotate('ID: Perfect\nconsistency', xy=(0.5, 1.0), xytext=(0.5, 0.55),
            fontsize=10, color='darkgreen', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='darkgreen'))
ax.annotate('Snow: Most\nerratic (5 tokens)', xy=(5, 0.6), xytext=(3.5, 0.48),
            fontsize=9, color='darkred', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='darkred'))

# Panel (b): Entropy comparison
ax = axes[1]
bars = ax.bar(range(len(scenarios)), entropies, 0.6, color=colors,
              edgecolor='black', linewidth=0.5, alpha=0.85)
ax.set_xticks(range(len(scenarios)))
ax.set_xticklabels(scenarios, fontsize=10, rotation=30, ha='right')
ax.set_ylabel('Output Entropy', fontsize=11)
ax.set_title('(b) Action Distribution Entropy', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

for bar, v in zip(bars, entropies):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.03,
            f'{v:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Panel (c): Cosine distance vs agreement scatter
ax = axes[2]
for i, (cd, ag, name) in enumerate(zip(cos_dists, agreements, scenarios)):
    c = colors[i]
    marker = 's' if is_ood[i] else 'o'
    ax.scatter(cd, ag, c=c, s=120, marker=marker, edgecolors='black',
               linewidth=0.5, zorder=5)
    ax.annotate(name, (cd, ag), textcoords="offset points",
                xytext=(8, 5), fontsize=9)

ax.set_xlabel('Cosine Distance to Centroid', fontsize=11)
ax.set_ylabel('Token Agreement Rate', fontsize=11)
ax.set_title('(c) OOD Distance vs Consistency', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Add correlation line
from numpy.polynomial import polynomial as P
coefs = P.polyfit(cos_dists, agreements, 1)
x_fit = np.linspace(0, 0.5, 100)
y_fit = P.polyval(x_fit, coefs)
ax.plot(x_fit, y_fit, 'k--', alpha=0.3, linewidth=1)

# Correlation coefficient
r = np.corrcoef(cos_dists, agreements)[0, 1]
ax.text(0.35, 0.95, f'r = {r:.2f}', fontsize=10, fontweight='bold')

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#4CAF50', label='ID'),
                   Patch(facecolor='#F44336', label='OOD')]
ax.legend(handles=legend_elements, fontsize=9)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig68_action_consistency.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig68_action_consistency.pdf', dpi=200, bbox_inches='tight')
print("Saved fig68_action_consistency.png/pdf")
