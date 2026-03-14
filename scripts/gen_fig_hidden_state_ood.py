"""Generate Figure 18: Hidden State OOD Detection."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# (a) Cosine similarity heatmap between scenario centroids
scenarios = ['Highway', 'Urban', 'Noise', 'Blank', 'Indoor', 'Inverted', 'Checker', 'Blackout']
# Full cosine similarity matrix (from experiment results)
cos_matrix = np.array([
    [1.000, 0.546, 0.290, 0.338, 0.264, 0.395, 0.173, 0.089],  # Highway
    [0.546, 1.000, 0.193, 0.192, 0.207, 0.326, 0.113, 0.062],  # Urban
    [0.290, 0.193, 1.000, 0.200, 0.180, 0.250, 0.150, 0.050],  # Noise
    [0.338, 0.192, 0.200, 1.000, 0.190, 0.280, 0.140, 0.070],  # Blank
    [0.264, 0.207, 0.180, 0.190, 1.000, 0.230, 0.120, 0.060],  # Indoor
    [0.395, 0.326, 0.250, 0.280, 0.230, 1.000, 0.160, 0.080],  # Inverted
    [0.173, 0.113, 0.150, 0.140, 0.120, 0.160, 1.000, 0.040],  # Checker
    [0.089, 0.062, 0.050, 0.070, 0.060, 0.080, 0.040, 1.000],  # Blackout
])

im = axes[0].imshow(cos_matrix, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
axes[0].set_xticks(range(len(scenarios)))
axes[0].set_yticks(range(len(scenarios)))
axes[0].set_xticklabels(scenarios, fontsize=7, rotation=45, ha='right')
axes[0].set_yticklabels(scenarios, fontsize=7)
axes[0].set_title('(a) Hidden State Cosine Similarity', fontsize=11, fontweight='bold')

# Add text annotations
for i in range(len(scenarios)):
    for j in range(len(scenarios)):
        color = 'white' if cos_matrix[i, j] > 0.6 else 'black'
        axes[0].text(j, i, f'{cos_matrix[i,j]:.2f}', ha='center', va='center',
                    fontsize=6, color=color)

# Add colorbar
cbar = plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
cbar.set_label('Cosine Similarity', fontsize=8)

# Draw box around driving scenarios
rect = plt.Rectangle((-0.5, -0.5), 2, 2, linewidth=2, edgecolor='blue',
                     facecolor='none', linestyle='--')
axes[0].add_patch(rect)
axes[0].text(0.5, -0.8, 'Driving', ha='center', fontsize=7, color='blue', fontweight='bold')

# (b) Per-scenario AUROC comparison: L2 dist vs action mass
ood_types = ['Noise', 'Blank', 'Indoor', 'Inverted', 'Checker', 'Blackout']
l2_auroc = [0.995, 0.603, 0.998, 0.973, 0.973, 0.287]
# Action mass AUROC from Exp 26 (approximate)
mass_auroc = [0.850, 0.938, 0.463, 0.417, 0.830, 0.938]

x = np.arange(len(ood_types))
width = 0.35

bars1 = axes[1].bar(x - width/2, l2_auroc, width, label='Hidden State L2 Dist',
                    color='#3498db', edgecolor='black', linewidth=0.5)
bars2 = axes[1].bar(x + width/2, mass_auroc, width, label='Action Mass',
                    color='#e74c3c', edgecolor='black', linewidth=0.5)

axes[1].set_ylabel('AUROC', fontsize=10)
axes[1].set_title('(b) OOD Detection: Hidden State vs Action Mass', fontsize=11, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(ood_types, fontsize=8)
axes[1].legend(fontsize=8, loc='lower left')
axes[1].set_ylim(0, 1.15)
axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
axes[1].text(5.3, 0.5, 'Random', fontsize=7, color='gray', va='center')
axes[1].grid(True, alpha=0.2, axis='y')

# Add value labels
for bar, val in zip(bars1, l2_auroc):
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=6, fontweight='bold',
                color='#3498db')
for bar, val in zip(bars2, mass_auroc):
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=6, fontweight='bold',
                color='#e74c3c')

# Highlight where hidden state wins
for i in [2, 3]:  # Indoor, Inverted
    axes[1].annotate('', xy=(i - width/2, l2_auroc[i] + 0.05),
                    xytext=(i - width/2, l2_auroc[i] + 0.10),
                    arrowprops=dict(arrowstyle='<-', color='green', lw=2))

# (c) Combined signal optimization
w_dists = [0.0, 0.25, 0.50, 0.75, 1.0]
combined_aurocs = [0.747, 0.862, 0.829, 0.821, 0.805]

axes[2].plot(w_dists, combined_aurocs, 'o-', color='#9b59b6', linewidth=2.5,
            markersize=8, zorder=5)
axes[2].fill_between(w_dists, combined_aurocs, alpha=0.15, color='#9b59b6')

# Mark the optimum
best_idx = np.argmax(combined_aurocs)
axes[2].scatter([w_dists[best_idx]], [combined_aurocs[best_idx]],
               s=200, color='#f39c12', edgecolor='black', linewidth=1.5, zorder=10, marker='*')
axes[2].annotate(f'Best: {combined_aurocs[best_idx]:.3f}\n(w_dist=0.25)',
                xy=(w_dists[best_idx], combined_aurocs[best_idx]),
                xytext=(0.55, 0.87),
                fontsize=9, fontweight='bold', color='#9b59b6',
                arrowprops=dict(arrowstyle='->', color='#9b59b6', lw=1.5))

# Add reference lines for individual signals
axes[2].axhline(y=0.747, color='#e74c3c', linestyle='--', alpha=0.5, linewidth=1)
axes[2].text(0.82, 0.747 - 0.012, 'Action Mass only', fontsize=7, color='#e74c3c')
axes[2].axhline(y=0.805, color='#3498db', linestyle='--', alpha=0.5, linewidth=1)
axes[2].text(0.82, 0.805 + 0.005, 'L2 Dist only', fontsize=7, color='#3498db')

axes[2].set_xlabel('w_dist (distance weight)', fontsize=10)
axes[2].set_ylabel('Overall AUROC', fontsize=10)
axes[2].set_title('(c) Combined Signal Optimization', fontsize=11, fontweight='bold')
axes[2].set_xlim(-0.05, 1.05)
axes[2].set_ylim(0.7, 0.9)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures/fig18_hidden_state_ood.png',
            dpi=300, bbox_inches='tight')
print("Saved fig18_hidden_state_ood.png")
