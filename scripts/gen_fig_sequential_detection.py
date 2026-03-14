"""Generate Figure 85: Sequential Inference Detection."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/sequential_detection_20260314_211245.json") as f:
    data = json.load(f)

threshold = data['threshold']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel (a): Gradual transition
ax = axes[0]
gradual = data['gradual_transition']
frames = [s['frame'] for s in gradual]
scores = [s['score'] for s in gradual]
alphas = [s['alpha'] for s in gradual]

colors_g = ['#F44336' if s['detected'] else '#4CAF50' for s in gradual]
ax.bar(frames, scores, color=colors_g, alpha=0.7, edgecolor='black', linewidth=0.3)
ax.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5, label=f'Threshold ({threshold:.4f})')
ax.set_xlabel('Frame', fontsize=11)
ax.set_ylabel('Cosine Distance', fontsize=11)
ax.set_title('(a) Gradual Transition', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# Add alpha labels for key frames
for i in [0, 1, 5, 10, 15, 19]:
    if i < len(gradual):
        ax.text(i, scores[i] + 0.01, f'α={alphas[i]:.2f}', ha='center', va='bottom',
               fontsize=6, rotation=45)

ax.annotate('1st detection\n(α=0.05)', xy=(1, scores[1]), xytext=(5, 0.15),
           fontsize=8, fontweight='bold', color='darkred',
           arrowprops=dict(arrowstyle='->', color='darkred'))

# Panel (b): Abrupt transition
ax = axes[1]
abrupt = data['abrupt_transition']
frames_a = [s['frame'] for s in abrupt]
scores_a = [s['score'] for s in abrupt]
colors_a = ['#4CAF50' if s['scene'] == 'highway' and not s['detected']
            else '#F44336' if s['detected']
            else '#FF9800' for s in abrupt]

ax.bar(frames_a, scores_a, color=colors_a, alpha=0.7, edgecolor='black', linewidth=0.3)
ax.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5, label=f'Threshold')
ax.axvline(x=9.5, color='gray', linestyle=':', linewidth=2, alpha=0.5)
ax.text(4, 0.42, 'Highway', fontsize=10, ha='center', color='#4CAF50', fontweight='bold')
ax.text(14, 0.42, 'Indoor', fontsize=10, ha='center', color='#F44336', fontweight='bold')
ax.set_xlabel('Frame', fontsize=11)
ax.set_ylabel('Cosine Distance', fontsize=11)
ax.set_title('(b) Abrupt Transition', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# Panel (c): Intermittent noise
ax = axes[2]
intermittent = data['intermittent_noise']
frames_i = [s['frame'] for s in intermittent]
scores_i = [s['score'] for s in intermittent]
colors_i = ['#F44336' if s['scene'] == 'noise' else '#4CAF50' for s in intermittent]

ax.bar(frames_i, scores_i, color=colors_i, alpha=0.7, edgecolor='black', linewidth=0.3)
ax.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5, label=f'Threshold')
ax.set_xlabel('Frame', fontsize=11)
ax.set_ylabel('Cosine Distance', fontsize=11)
ax.set_title('(c) Intermittent Noise', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# Legend for scene types
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#4CAF50', label='Highway'),
                   Patch(facecolor='#F44336', label='Noise')]
ax.legend(handles=legend_elements, fontsize=8, loc='center right')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig85_sequential_detection.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig85_sequential_detection.pdf', dpi=200, bbox_inches='tight')
print("Saved fig85_sequential_detection.png/pdf")
