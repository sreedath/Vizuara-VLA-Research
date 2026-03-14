"""Generate Figure 23: Action Prediction Consistency Under Distribution Shift."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/action_consistency_20260314_155047.json"
OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

with open(RESULTS) as f:
    data = json.load(f)

results = data['results']
corruptions = ['noise', 'darken', 'invert', 'blur', 'occlude']
severities = [0.25, 0.50, 0.75, 1.0]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

colors = {'noise': '#e41a1c', 'darken': '#377eb8', 'invert': '#4daf4a',
          'blur': '#984ea3', 'occlude': '#ff7f00'}
markers = {'noise': 'o', 'darken': 's', 'invert': '^', 'blur': 'D', 'occlude': 'v'}

# Panel (a): Token agreement vs severity
ax = axes[0]
for c in corruptions:
    means = [1.0]  # severity 0 = perfect agreement
    for sev in severities:
        vals = [r['token_agreement'] for r in results
                if r['corruption'] == c and r['severity'] == sev]
        means.append(np.mean(vals))
    ax.plot([0.0] + severities, means, marker=markers[c], color=colors[c],
            label=c.capitalize(), linewidth=2, markersize=6)

ax.set_xlabel('Corruption Severity', fontsize=11)
ax.set_ylabel('Token Agreement (7-dim)', fontsize=11)
ax.set_title('(a) Action Token Agreement', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.3)

# Panel (b): Cosine dist vs action L2 scatter
ax = axes[1]
cos_dists = [r['cos_dist'] for r in results]
action_l2s = [r['action_l2'] for r in results]
corr_colors = [colors[r['corruption']] for r in results]

for c in corruptions:
    c_cos = [r['cos_dist'] for r in results if r['corruption'] == c]
    c_l2 = [r['action_l2'] for r in results if r['corruption'] == c]
    ax.scatter(c_cos, c_l2, color=colors[c], alpha=0.5, s=20,
              label=c.capitalize(), marker=markers[c])

# Correlation line
z = np.polyfit(cos_dists, action_l2s, 1)
p = np.poly1d(z)
x_line = np.linspace(min(cos_dists), max(cos_dists), 100)
ax.plot(x_line, p(x_line), 'k--', linewidth=1.5, alpha=0.7)

r = np.corrcoef(cos_dists, action_l2s)[0, 1]
ax.set_xlabel('Cosine Distance', fontsize=11)
ax.set_ylabel('Action L2 Distance', fontsize=11)
ax.set_title(f'(b) Cosine Dist vs Action L2 (r={r:+.2f})', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='lower right')
ax.grid(True, alpha=0.3)

# Panel (c): Binned action quality by cosine distance
ax = axes[2]
bins = [(0.0, 0.55), (0.55, 0.65), (0.65, 0.75), (0.75, 0.85), (0.85, 1.0)]
bin_labels = ['<0.55', '0.55-\n0.65', '0.65-\n0.75', '0.75-\n0.85', '>0.85']
agree_means = []
agree_stds = []
l2_means = []

for lo, hi in bins:
    bin_samples = [r for r in results if lo <= r['cos_dist'] < hi]
    if bin_samples:
        agree_means.append(np.mean([r['token_agreement'] for r in bin_samples]))
        agree_stds.append(np.std([r['token_agreement'] for r in bin_samples]))
        l2_means.append(np.mean([r['action_l2'] for r in bin_samples]))
    else:
        agree_means.append(0)
        agree_stds.append(0)
        l2_means.append(0)

x = np.arange(len(bins))
width = 0.35

bars1 = ax.bar(x - width/2, agree_means, width, yerr=agree_stds,
               label='Token Agreement', color='#2196F3', alpha=0.8, capsize=3)

ax2 = ax.twinx()
bars2 = ax2.bar(x + width/2, l2_means, width,
                label='Action L2', color='#FF5722', alpha=0.8)

ax.set_xlabel('Cosine Distance Bin', fontsize=11)
ax.set_ylabel('Token Agreement', fontsize=11, color='#2196F3')
ax2.set_ylabel('Action L2 Distance', fontsize=11, color='#FF5722')
ax.set_title('(c) Action Quality by Cosine Bin', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(bin_labels, fontsize=9)
ax.set_ylim(0, 0.35)
ax2.set_ylim(0, 200)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig23_action_consistency.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig23_action_consistency.pdf', dpi=200, bbox_inches='tight')
print("Saved fig23_action_consistency.png/pdf")
