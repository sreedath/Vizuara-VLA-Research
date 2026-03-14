"""Generate Figure 90: Confidence Calibration."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/confidence_calibration_20260314_213324.json") as f:
    data = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel (a): ECE and Brier comparison
ax = axes[0]
methods = ['raw', 'platt', 'isotonic', 'histogram', 'sigmoid']
method_labels = ['Raw\nMin-Max', 'Platt\nScaling', 'Isotonic\nRegression', 'Histogram\nBinning', 'Temp.\nSigmoid']
ece_vals = [data['methods'][m]['ece'] for m in methods]
brier_vals = [data['methods'][m]['brier'] for m in methods]

x = np.arange(len(methods))
width = 0.35
bars1 = ax.bar(x - width/2, ece_vals, width, label='ECE', color='#F44336', alpha=0.7,
               edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, brier_vals, width, label='Brier Score', color='#2196F3', alpha=0.7,
               edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(method_labels, fontsize=8)
ax.set_ylabel('Score (lower = better)', fontsize=11)
ax.set_title('(a) Calibration Quality', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Annotate perfect calibration
for i, (e, b) in enumerate(zip(ece_vals, brier_vals)):
    if e == 0 and b == 0:
        ax.annotate('Perfect', (i, 0.01), ha='center', fontsize=8, fontweight='bold', color='green')

# Panel (b): Reliability diagram for key methods
ax = axes[1]
colors_rel = {'raw': '#FF9800', 'platt': '#F44336', 'sigmoid': '#9C27B0'}
for method, color in colors_rel.items():
    bins = data['methods'][method]['bins']
    confs = [b['avg_confidence'] for b in bins if b['count'] > 0]
    accs = [b['avg_accuracy'] for b in bins if b['count'] > 0]
    counts = [b['count'] for b in bins if b['count'] > 0]
    ax.scatter(confs, accs, color=color, s=[c*5 for c in counts], alpha=0.7,
               label=f'{method.capitalize()}', edgecolors='black', linewidth=0.5)

ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
ax.set_xlabel('Predicted Probability', fontsize=11)
ax.set_ylabel('Actual Fraction OOD', fontsize=11)
ax.set_title('(b) Reliability Diagram', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.3)

# Panel (c): Temperature sensitivity
ax = axes[2]
temp_data = data['methods']['sigmoid']['temp_search']
temps = sorted(temp_data.keys(), key=lambda k: temp_data[k]['temperature'])
t_vals = [temp_data[t]['temperature'] for t in temps]
ece_t = [temp_data[t]['ece'] for t in temps]
brier_t = [temp_data[t]['brier'] for t in temps]

ax.semilogx(t_vals, ece_t, 'o-', color='#F44336', label='ECE', linewidth=2, markersize=6)
ax.semilogx(t_vals, brier_t, 's-', color='#2196F3', label='Brier', linewidth=2, markersize=6)
ax.axvline(x=data['methods']['sigmoid']['best_temperature'], color='green', linestyle='--',
           alpha=0.5, label=f'Best T={data["methods"]["sigmoid"]["best_temperature"]}')
ax.set_xlabel('Temperature', fontsize=11)
ax.set_ylabel('Score', fontsize=11)
ax.set_title('(c) Sigmoid Temperature Sensitivity', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig90_confidence_calibration.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig90_confidence_calibration.pdf', dpi=200, bbox_inches='tight')
print("Saved fig90_confidence_calibration.png/pdf")
