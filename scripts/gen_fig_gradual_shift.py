"""Generate Figure 22: Gradual Distribution Shift Analysis."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/gradual_shift_20260314_154333.json"
OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

with open(RESULTS) as f:
    data = json.load(f)

results = data['results']
severities = data['severities']
corruptions = data['corruptions']
threshold = data['conformal_threshold']

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Panel (a): Cosine distance vs severity for each corruption
ax = axes[0]
colors = {'noise': '#e41a1c', 'darken': '#377eb8', 'invert': '#4daf4a',
          'blur': '#984ea3', 'occlude': '#ff7f00'}
markers = {'noise': 'o', 'darken': 's', 'invert': '^', 'blur': 'D', 'occlude': 'v'}

for c in corruptions:
    means = []
    stds = []
    for sev in severities:
        vals = [r['cos_dist'] for r in results
                if r['corruption'] == c and r['severity'] == sev]
        means.append(np.mean(vals))
        stds.append(np.std(vals))
    ax.errorbar(severities, means, yerr=stds, marker=markers[c],
                color=colors[c], label=c.capitalize(), linewidth=2,
                markersize=6, capsize=3)

ax.axhline(y=threshold, color='gray', linestyle='--', linewidth=1.5,
           label=f'Conformal thr. (α=0.10)')
ax.set_xlabel('Corruption Severity', fontsize=11)
ax.set_ylabel('Cosine Distance', fontsize=11)
ax.set_title('(a) Cosine Distance vs Severity', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='lower right')
ax.set_xlim(-0.02, 1.02)
ax.grid(True, alpha=0.3)

# Panel (b): Action mass vs severity (flat — showing it fails)
ax = axes[1]
for c in corruptions:
    means = []
    for sev in severities:
        vals = [r['action_mass'] for r in results
                if r['corruption'] == c and r['severity'] == sev]
        means.append(np.mean(vals))
    ax.plot(severities, means, marker=markers[c], color=colors[c],
            label=c.capitalize(), linewidth=2, markersize=6)

ax.set_xlabel('Corruption Severity', fontsize=11)
ax.set_ylabel('Action Mass', fontsize=11)
ax.set_title('(b) Action Mass vs Severity', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='lower left')
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(0.7, 1.02)
ax.grid(True, alpha=0.3)

# Panel (c): Flag rate vs severity (stacked area or grouped bar)
ax = axes[2]
for c in corruptions:
    rates = []
    for sev in severities:
        samples = [r for r in results
                   if r['corruption'] == c and r['severity'] == sev]
        flagged = sum(1 for r in samples if r['flagged'])
        rates.append(flagged / len(samples) if samples else 0)
    ax.plot(severities, [r * 100 for r in rates], marker=markers[c],
            color=colors[c], label=c.capitalize(), linewidth=2, markersize=6)

ax.axhline(y=50, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax.set_xlabel('Corruption Severity', fontsize=11)
ax.set_ylabel('OOD Flag Rate (%)', fontsize=11)
ax.set_title('(c) Conformal Flag Rate vs Severity', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='center right')
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-5, 105)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig22_gradual_shift.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig22_gradual_shift.pdf', dpi=200, bbox_inches='tight')
print("Saved fig22_gradual_shift.png/pdf")
