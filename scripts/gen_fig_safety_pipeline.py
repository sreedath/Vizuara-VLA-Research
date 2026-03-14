"""Generate Figure 17: Safety Pipeline Evaluation."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# (a) Per-scenario decision distribution (stacked bar)
scenarios = ['Highway', 'Urban', 'Night', 'Rain', 'OOD\nNoise', 'OOD\nBlank']
proceed = [9/16, 11/14, 0, 0, 0, 0]
caution = [2/16, 3/14, 0, 1/20, 16/25, 2/25]
slow = [1/16, 0, 19/20, 17/20, 1/25, 10/25]
stop = [4/16, 0, 1/20, 2/20, 8/25, 13/25]

x = np.arange(len(scenarios))
width = 0.65

p1 = axes[0].bar(x, proceed, width, label='PROCEED', color='#2ecc71')
p2 = axes[0].bar(x, caution, width, bottom=proceed, label='CAUTION', color='#f1c40f')
p3 = axes[0].bar(x, slow, width, bottom=[p+c for p, c in zip(proceed, caution)], label='SLOW', color='#e67e22')
p4 = axes[0].bar(x, stop, width, bottom=[p+c+s for p, c, s in zip(proceed, caution, slow)], label='STOP', color='#e74c3c')

axes[0].set_ylabel('Fraction', fontsize=10)
axes[0].set_title('(a) Pipeline Decisions by Scenario', fontsize=11, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(scenarios, fontsize=8)
axes[0].legend(fontsize=7, loc='upper right')
axes[0].set_ylim(0, 1.1)

# Add correct action markers
correct_actions = ['PROCEED', 'PROCEED', 'SLOW', 'SLOW', 'STOP', 'STOP']
for i, action in enumerate(correct_actions):
    axes[0].text(i, 1.03, f'Target: {action}', ha='center', fontsize=6, fontstyle='italic')

# (b) Safety rate vs alpha
alphas = [0.05, 0.10, 0.15, 0.20, 0.30]
safety = [66.7, 75.0, 75.8, 89.2, 98.3]
accuracy = [64.2, 68.3, 69.2, 70.8, 58.3]
ood_stop = [22.0, 42.0, 44.0, 76.0, 96.0]
easy_proceed = [90.0, 83.3, 83.3, 73.3, 73.3]

axes[1].plot(alphas, safety, 'o-', color='#e74c3c', linewidth=2, markersize=6, label='Safety Rate')
axes[1].plot(alphas, ood_stop, 's-', color='#9b59b6', linewidth=2, markersize=6, label='OOD→STOP')
axes[1].plot(alphas, easy_proceed, '^-', color='#2ecc71', linewidth=2, markersize=6, label='Easy→PROCEED')
axes[1].plot(alphas, accuracy, 'D-', color='#3498db', linewidth=1.5, markersize=5, label='Accuracy')

axes[1].set_xlabel('α (Conformal Level)', fontsize=10)
axes[1].set_ylabel('Rate (%)', fontsize=10)
axes[1].set_title('(b) Safety vs Throughput Trade-off', fontsize=11, fontweight='bold')
axes[1].legend(fontsize=7, loc='center left')
axes[1].set_xlim(0.03, 0.32)
axes[1].set_ylim(0, 105)
axes[1].grid(True, alpha=0.3)

# Highlight sweet spot
axes[1].axvspan(0.18, 0.22, alpha=0.15, color='yellow')
axes[1].annotate('Sweet spot\nα=0.20', xy=(0.20, 89.2), xytext=(0.25, 80),
                fontsize=8, fontweight='bold', color='#e74c3c',
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5))

# (c) Key metrics summary
metrics = ['Safety\nRate', 'Night\n→SLOW', 'OOD\n→STOP', 'Easy\n→PROCEED', 'Accuracy']
values_020 = [89.2, 95.0, 76.0, 73.3, 70.8]
values_030 = [98.3, 95.0, 96.0, 73.3, 58.3]

x = np.arange(len(metrics))
width = 0.35

bars1 = axes[2].bar(x - width/2, values_020, width, label='α=0.20 (balanced)', color='#3498db', edgecolor='black', linewidth=0.5)
bars2 = axes[2].bar(x + width/2, values_030, width, label='α=0.30 (safety-first)', color='#e74c3c', edgecolor='black', linewidth=0.5)

axes[2].set_ylabel('Rate (%)', fontsize=10)
axes[2].set_title('(c) Pipeline Operating Points', fontsize=11, fontweight='bold')
axes[2].set_xticks(x)
axes[2].set_xticklabels(metrics, fontsize=8)
axes[2].legend(fontsize=8)
axes[2].set_ylim(0, 110)

for bar, val in zip(bars1, values_020):
    axes[2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=7, fontweight='bold')
for bar, val in zip(bars2, values_030):
    axes[2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=7, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures/fig17_safety_pipeline.png',
            dpi=300, bbox_inches='tight')
print("Saved fig17_safety_pipeline.png")
