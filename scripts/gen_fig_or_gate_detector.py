"""Generate Figure 130: OR-Gate Dual-Layer Detector."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/or_gate_detector_20260315_004727.json") as f:
    data = json.load(f)

results = data['results']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel A: Strategy comparison
ax = axes[0]
strats = ['L3_only', 'L32_only', 'OR_gate', 'AND_gate']
labels = ['L3 Only', 'L32 Only', 'OR Gate\n(L3 ∨ L32)', 'AND Gate\n(L3 ∧ L32)']
metrics = ['precision', 'recall', 'f1']
metric_labels = ['Precision', 'Recall', 'F1']
colors = ['#2196F3', '#4CAF50', '#FF9800']

x = np.arange(len(strats))
width = 0.25
for i, (metric, color, label) in enumerate(zip(metrics, colors, metric_labels)):
    vals = [results[s][metric] for s in strats]
    ax.bar(x + i*width, vals, width, color=color, alpha=0.8, label=label)

ax.set_xticks(x + width)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Score")
ax.set_title("(A) Strategy Comparison")
ax.set_ylim(0.88, 1.01)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# Panel B: Per-category recall for OR gate
ax = axes[1]
or_cats = list(results['OR_gate']['per_category'].keys())
or_recalls = [results['OR_gate']['per_category'][c]['recall'] for c in or_cats]
or_colors = ['#4CAF50' if r == 1.0 else '#F44336' for r in or_recalls]
bars = ax.barh(range(len(or_cats)), or_recalls, color=or_colors, alpha=0.8)
ax.axvline(x=1.0, color='green', linestyle='--', alpha=0.5)
ax.set_yticks(range(len(or_cats)))
ax.set_yticklabels([c.replace('_', ' ').title() for c in or_cats], fontsize=9)
ax.set_xlabel("Recall")
ax.set_title("(B) OR-Gate Per-Category Recall")
ax.set_xlim(0.0, 1.1)
ax.grid(True, alpha=0.3, axis='x')
for bar, r in zip(bars, or_recalls):
    ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
            f"{r:.3f}", va='center', fontsize=9)
ax.invert_yaxis()

# Panel C: L3 vs L32 catch comparison
ax = axes[2]
ood_cats = [c for c in or_cats]
l3_recalls = [results['L3_only']['per_category'][c]['recall'] for c in ood_cats]
l32_recalls = [results['L32_only']['per_category'][c]['recall'] for c in ood_cats]
or_recalls_c = [results['OR_gate']['per_category'][c]['recall'] for c in ood_cats]

x = np.arange(len(ood_cats))
width = 0.25
ax.bar(x - width, l3_recalls, width, color='#4CAF50', alpha=0.7, label='L3')
ax.bar(x, l32_recalls, width, color='#F44336', alpha=0.7, label='L32')
ax.bar(x + width, or_recalls_c, width, color='#2196F3', alpha=0.7, label='OR Gate')
ax.set_xticks(x)
ax.set_xticklabels([c.replace('_', ' ') for c in ood_cats], fontsize=7, rotation=45, ha='right')
ax.set_ylabel("Recall")
ax.set_title("(C) Layer Complementarity")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 1.1)

plt.suptitle("OR-Gate Dual-Layer Detector (Exp 144)\nOR gate achieves P=0.994, R=1.000, F1=0.997 — perfect recall on ALL categories including fog_30%",
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/latex/figures/fig130_or_gate_detector.png",
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig130")
