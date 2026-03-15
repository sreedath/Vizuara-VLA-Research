import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/threshold_sensitivity_20260315_010706.json") as f:
    data = json.load(f)

results = data["results"]
sigmas = data["sigma_levels"]
cats = data["ood_categories"]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel A: Recall-FPR tradeoff across σ for all strategies
ax = axes[0]
strategies = ["L3_only", "L32_only", "OR_gate", "AND_gate"]
colors = {"L3_only": "#2196F3", "L32_only": "#FF9800", "OR_gate": "#4CAF50", "AND_gate": "#F44336"}
labels = {"L3_only": "L3 Only", "L32_only": "L32 Only", "OR_gate": "OR Gate (L3∨L32)", "AND_gate": "AND Gate (L3∧L32)"}

for strat in strategies:
    recalls = [results[f"sigma_{s}"][strat]["recall"] for s in sigmas]
    fprs = [results[f"sigma_{s}"][strat]["fpr"] for s in sigmas]
    ax.plot(fprs, recalls, 'o-', color=colors[strat], label=labels[strat], markersize=6, linewidth=2)

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("Recall")
ax.set_title("(A) ROC Curve Across Threshold Levels")
ax.legend(fontsize=9)
ax.set_xlim(-0.02, 0.4)
ax.set_ylim(0.75, 1.02)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
ax.axvline(x=0.0, color='gray', linestyle='--', alpha=0.3)

# Panel B: F1 vs σ for all strategies
ax = axes[1]
for strat in strategies:
    f1s = [results[f"sigma_{s}"][strat]["f1"] for s in sigmas]
    ax.plot(sigmas, f1s, 'o-', color=colors[strat], label=labels[strat], markersize=6, linewidth=2)

ax.set_xlabel("Threshold (σ)")
ax.set_ylabel("F1 Score")
ax.set_title("(B) F1 Score vs Threshold Level")
ax.legend(fontsize=9)
ax.set_ylim(0.8, 1.0)
ax.axhline(y=0.96, color='gray', linestyle='--', alpha=0.3)
# Mark optimal
best_sigma_idx = max(range(len(sigmas)), key=lambda i: results[f"sigma_{sigmas[i]}"]["OR_gate"]["f1"])
best_sigma = sigmas[best_sigma_idx]
best_f1 = results[f"sigma_{best_sigma}"]["OR_gate"]["f1"]
ax.annotate(f'σ={best_sigma}\nF1={best_f1:.3f}', xy=(best_sigma, best_f1),
            xytext=(best_sigma + 0.8, best_f1 - 0.03), fontsize=9,
            arrowprops=dict(arrowstyle='->', color='black'))

# Panel C: Per-category recall heatmap at σ=3.0
ax = axes[2]
sigma_key = "sigma_3.0"
heatmap_data = []
strat_labels = ["L3 Only", "L32 Only", "OR Gate", "AND Gate"]
strat_keys = ["L3_only", "L32_only", "OR_gate", "AND_gate"]
for sk in strat_keys:
    row = [results[sigma_key][sk]["per_category"][c]["recall"] for c in cats]
    heatmap_data.append(row)
heatmap_data = np.array(heatmap_data)
im = ax.imshow(heatmap_data, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
ax.set_xticks(range(len(cats)))
ax.set_xticklabels([c.replace("_", "\n") for c in cats], fontsize=8, rotation=45, ha='right')
ax.set_yticks(range(len(strat_keys)))
ax.set_yticklabels(strat_labels, fontsize=9)
for i in range(len(strat_keys)):
    for j in range(len(cats)):
        val = heatmap_data[i, j]
        color = 'white' if val < 0.5 else 'black'
        ax.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=8, color=color)
plt.colorbar(im, ax=ax, shrink=0.8, label='Recall')
ax.set_title("(C) Per-Category Recall at σ=3.0")

plt.tight_layout()
plt.savefig("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/latex/figures/fig132_threshold_sensitivity.png", dpi=150, bbox_inches='tight')
print("Saved fig132")
