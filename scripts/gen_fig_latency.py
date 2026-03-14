"""Generate Figure 71: Computational Overhead Analysis."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel (a): Inference latency comparison
ax = axes[0]
modes = ['Baseline', 'Hidden\nStates', 'Attention', 'Full\nFeatures']
latencies = [84.1, 84.3, 89.9, 89.5]
stds = [0.5, 0.4, 0.5, 0.4]
colors = ['#607D8B', '#4CAF50', '#FF9800', '#2196F3']

bars = ax.bar(range(len(modes)), latencies, 0.6, yerr=stds, capsize=5,
              color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)
ax.set_xticks(range(len(modes)))
ax.set_xticklabels(modes, fontsize=10)
ax.set_ylabel('Latency (ms)', fontsize=11)
ax.set_title('(a) Inference Latency', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(75, 95)

for bar, v in zip(bars, latencies):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.8,
            f'{v:.1f}ms', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.annotate('Hidden states:\n+0.3% overhead!', xy=(1, 84.5), xytext=(2, 78),
            fontsize=10, color='darkgreen', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='darkgreen'))

# Panel (b): Overhead percentages
ax = axes[1]
overhead_labels = ['Hidden\nStates', 'Attention', 'Full\nFeatures']
overhead_ms = [0.2, 5.8, 5.4]
overhead_pct = [0.3, 6.9, 6.5]
colors_oh = ['#4CAF50', '#FF9800', '#2196F3']

bars = ax.bar(range(len(overhead_labels)), overhead_pct, 0.5, color=colors_oh,
              edgecolor='black', linewidth=0.5, alpha=0.85)
ax.set_xticks(range(len(overhead_labels)))
ax.set_xticklabels(overhead_labels, fontsize=10)
ax.set_ylabel('Overhead (%)', fontsize=11)
ax.set_title('(b) Detection Overhead', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

for bar, v, ms in zip(bars, overhead_pct, overhead_ms):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
            f'{v:.1f}%\n(+{ms:.1f}ms)', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.3, label='1% overhead')
ax.legend(fontsize=8)

# Panel (c): Post-processing time scale comparison
ax = axes[2]
components = ['Model\nForward', 'Hidden State\nOverhead', 'Cosine\nDistance', 'PCA-4\n+ Cosine']
times_us = [84100, 200, 7.6, 7.2]
colors_pp = ['#607D8B', '#4CAF50', '#FF5722', '#E91E63']

bars = ax.barh(range(len(components)), np.log10(times_us), 0.6, color=colors_pp,
               edgecolor='black', linewidth=0.5, alpha=0.85)
ax.set_yticks(range(len(components)))
ax.set_yticklabels(components, fontsize=10)
ax.set_xlabel('Time (log₁₀ μs)', fontsize=11)
ax.set_title('(c) Time Scale Comparison', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

labels_text = ['84.1 ms', '0.2 ms', '7.6 μs', '7.2 μs']
for bar, txt in zip(bars, labels_text):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2.,
            txt, ha='left', va='center', fontsize=10, fontweight='bold')

ax.annotate('Post-processing\n10,000× faster\nthan forward pass', xy=(1, 2.5),
            fontsize=9, color='darkred', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig71_latency.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig71_latency.pdf', dpi=200, bbox_inches='tight')
print("Saved fig71_latency.png/pdf")
