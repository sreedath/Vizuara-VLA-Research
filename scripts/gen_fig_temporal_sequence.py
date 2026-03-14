"""Generate Figure 107: Temporal Sequence Analysis."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/temporal_sequence_20260314_230605.json") as f:
    data = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel (a): Smooth transition — highway to noise and indoor
ax = axes[0]
seq1 = data['sequences']['highway_to_noise']
seq2 = data['sequences']['highway_to_indoor']
alphas1 = [d['alpha'] for d in seq1]
scores1 = [d['score'] for d in seq1]
alphas2 = [d['alpha'] for d in seq2]
scores2 = [d['score'] for d in seq2]

ax.plot(alphas1, scores1, 'o-', color='#F44336', linewidth=2, markersize=5, label='Highway → Noise')
ax.plot(alphas2, scores2, 's-', color='#FF9800', linewidth=2, markersize=5, label='Highway → Indoor')
ax.axhline(y=0.1, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Detection threshold')
ax.set_xlabel('Blend Factor α (0=highway, 1=OOD)', fontsize=11)
ax.set_ylabel('Cosine Distance Score', fontsize=11)
ax.set_title('(a) Smooth OOD Transition', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel (b): Sudden transition
ax = axes[1]
seq3 = data['sequences']['sudden_transition']
frames = [d['frame'] for d in seq3]
scores3 = [d['score'] for d in seq3]
labels3 = [d['label'] for d in seq3]
colors3 = ['#4CAF50' if l == 'highway' else '#F44336' for l in labels3]

ax.bar(frames, scores3, color=colors3, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.axvline(x=9.5, color='blue', linestyle='--', linewidth=2, label='Transition point')
ax.set_xlabel('Frame', fontsize=11)
ax.set_ylabel('Cosine Distance Score', fontsize=11)
ax.set_title('(b) Sudden Transition', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Panel (c): Oscillating
ax = axes[2]
seq4 = data['sequences']['oscillating']
frames4 = [d['frame'] for d in seq4]
scores4 = [d['score'] for d in seq4]
labels4 = [d['label'] for d in seq4]
colors4 = ['#4CAF50' if l == 'highway' else '#F44336' for l in labels4]

ax.bar(frames4, scores4, color=colors4, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.set_xlabel('Frame', fontsize=11)
ax.set_ylabel('Cosine Distance Score', fontsize=11)
ax.set_title('(c) Oscillating Input', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#4CAF50', alpha=0.8, label='Highway'),
                   Patch(facecolor='#F44336', alpha=0.8, label='Noise')]
ax.legend(handles=legend_elements, fontsize=9)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig107_temporal_sequence.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig107_temporal_sequence.pdf', dpi=200, bbox_inches='tight')
print("Saved fig107_temporal_sequence.png/pdf")
