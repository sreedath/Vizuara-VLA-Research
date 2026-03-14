"""Generate Figure 47: Detection Threshold Sensitivity."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# Panel (a): Threshold vs TPR/FPR for cosine
ax = axes[0]
percentiles = [50, 75, 90, 95, 99]
thresholds = [0.5824, 0.6255, 0.6886, 0.7158, 0.7462]
tprs = [1.000, 0.950, 0.750, 0.725, 0.625]
fprs = [0.500, 0.267, 0.100, 0.067, 0.033]

ax.plot(percentiles, tprs, 'b-o', linewidth=2, markersize=8, label='TPR (recall)')
ax.plot(percentiles, fprs, 'r-s', linewidth=2, markersize=8, label='FPR')
ax.fill_between(percentiles, tprs, fprs, alpha=0.1, color='green')
ax.axhline(y=0.95, color='green', linestyle='--', alpha=0.5, label='95% TPR target')
ax.set_xlabel('ID Percentile Threshold', fontsize=11)
ax.set_ylabel('Rate', fontsize=11)
ax.set_title('(a) TPR/FPR vs Threshold (Cosine)', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='center right')
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.05)
ax.annotate('p75: best\nTPR/FPR tradeoff', xy=(75, 0.950), xytext=(82, 0.75),
            fontsize=8, color='blue', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='blue'))

# Panel (b): Per-scenario catch rates
ax = axes[1]
scenarios = ['Noise', 'Indoor', 'Inverted', 'Blackout']
catch_p90 = [100, 50, 50, 100]
catch_p95 = [100, 50, 40, 100]
colors = ['#F44336', '#FF9800', '#9C27B0', '#333333']

x = np.arange(len(scenarios))
width = 0.35
bars1 = ax.bar(x - width/2, catch_p90, width, label='p90 threshold',
               color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)
bars2 = ax.bar(x + width/2, catch_p95, width, label='p95 threshold',
               color=colors, edgecolor='black', linewidth=0.5, alpha=0.5,
               hatch='///')
ax.set_xticks(x)
ax.set_xticklabels(scenarios, fontsize=10)
ax.set_ylabel('% Caught', fontsize=11)
ax.set_title('(b) Per-OOD Catch Rate at Thresholds', fontsize=12, fontweight='bold')
ax.set_ylim(0, 115)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')
ax.annotate('Indoor/Inverted:\nharder to catch', xy=(1, 50), xytext=(1.5, 70),
            fontsize=8, color='#FF9800',
            arrowprops=dict(arrowstyle='->', color='#FF9800'))

# Panel (c): Conformal prediction
ax = axes[2]
alphas = [0.01, 0.05, 0.10, 0.20]
ood_caught = [85, 95, 95, 98]
false_alarm = [17, 27, 30, 47]

ax.plot(alphas, ood_caught, 'b-o', linewidth=2, markersize=8, label='OOD caught %')
ax.plot(alphas, false_alarm, 'r-s', linewidth=2, markersize=8, label='False alarm %')
ax.fill_between(alphas, ood_caught, false_alarm, alpha=0.1, color='green')
ax.set_xlabel('Conformal α (miscoverage rate)', fontsize=11)
ax.set_ylabel('Percentage', fontsize=11)
ax.set_title('(c) Conformal Prediction Thresholds', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 105)
ax.annotate('α=0.05: 95% OOD\ncaught, 27% FA',
            xy=(0.05, 95), xytext=(0.10, 75),
            fontsize=8, color='blue', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='blue'))

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig47_threshold.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig47_threshold.pdf', dpi=200, bbox_inches='tight')
print("Saved fig47_threshold.png/pdf")
