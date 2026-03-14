"""Generate Figure 46: Zero-Overhead Latency Profile."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# Data from latency_profile experiment
configs = ['Baseline\n(no extras)', 'Scores\nOnly', 'Hidden\nStates Only', 'Both\n(CalibDrive)']
means = [296.0, 292.2, 294.4, 294.8]
stds = [5.1, 8.8, 9.4, 3.1]
colors = ['#9E9E9E', '#FF9800', '#2196F3', '#4CAF50']

# Panel (a): Latency comparison
ax = axes[0]
bars = ax.bar(range(len(configs)), means, 0.6, yerr=stds, capsize=5,
              color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)
ax.set_xticks(range(len(configs)))
ax.set_xticklabels(configs, fontsize=9)
ax.set_ylabel('Inference Latency (ms)', fontsize=11)
ax.set_title('(a) Generation Latency by Configuration', fontsize=12, fontweight='bold')
ax.set_ylim(250, 330)
ax.grid(True, alpha=0.3, axis='y')
for bar, m, s in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + s + 2,
            f'{m:.0f}ms', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.annotate('Within noise\n(±1σ overlap)', xy=(3, 294.8), xytext=(2.2, 315),
            fontsize=9, color='green', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='green'))

# Panel (b): Overhead percentage
ax = axes[1]
overheads = [-3.9, -1.7, -1.3]
overhead_labels = ['Scores Only', 'Hidden Only', 'Both (CalibDrive)']
overhead_colors = ['#FF9800', '#2196F3', '#4CAF50']

bars = ax.barh(range(len(overheads)), overheads, 0.5, color=overhead_colors,
               edgecolor='black', linewidth=0.5, alpha=0.85)
ax.axvline(x=0, color='black', linewidth=1)
ax.set_yticks(range(len(overheads)))
ax.set_yticklabels(overhead_labels, fontsize=10)
ax.set_xlabel('Overhead vs Baseline (ms)', fontsize=11)
ax.set_title('(b) Overhead: All Within Noise', fontsize=12, fontweight='bold')
ax.set_xlim(-15, 15)
ax.grid(True, alpha=0.3, axis='x')
# Shade noise band
ax.axvspan(-5.1, 5.1, alpha=0.1, color='gray', label='±1σ noise band')
ax.legend(fontsize=8)

for bar, v in zip(bars, overheads):
    x = bar.get_width()
    ax.text(x + 0.5 if x >= 0 else x - 0.5, bar.get_y() + bar.get_height()/2.,
            f'{v:+.1f}ms', ha='left' if x >= 0 else 'right', va='center',
            fontsize=10, fontweight='bold')

# Panel (c): Post-processing cost breakdown
ax = axes[2]
labels = ['Model\nInference', 'Cosine\nDistance', 'Action\nMass']
times_us = [294800, 12.21, 249.30]
times_ms = [t/1000 for t in times_us]
pcts = [t/sum(times_us)*100 for t in times_us]

# Use log scale for the dramatic difference
bars = ax.bar(range(len(labels)), times_us, 0.6,
              color=['#9E9E9E', '#2196F3', '#FF9800'],
              edgecolor='black', linewidth=0.5, alpha=0.85)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel('Time (µs, log scale)', fontsize=11)
ax.set_title('(c) Post-Processing Cost', fontsize=12, fontweight='bold')
ax.set_yscale('log')
ax.grid(True, alpha=0.3, axis='y')
for bar, t, p in zip(bars, times_us, pcts):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() * 1.5,
            f'{p:.2f}%' if p < 1 else f'{p:.0f}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig46_latency.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig46_latency.pdf', dpi=200, bbox_inches='tight')
print("Saved fig46_latency.png/pdf")
