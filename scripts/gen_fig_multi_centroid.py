import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/multi_centroid_20260315_012448.json") as f:
    data = json.load(f)

results = data["results"]
prompts = data["prompts"]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel A: F1 comparison across strategies
ax = axes[0]
strategies = ["same_prompt", "wrong_prompt", "nearest_centroid", "min_distance"]
strat_labels = ["Same Prompt\n(oracle)", "Wrong Prompt\n(worst case)", "Nearest\nCentroid", "Min\nDistance"]
colors = ["#4CAF50", "#F44336", "#2196F3", "#FF9800"]

for i, (strat, label, color) in enumerate(zip(strategies, strat_labels, colors)):
    f1s = [results[p][strat]["f1"] for p in prompts]
    ax.bar(i, np.mean(f1s), yerr=np.std(f1s), color=color, alpha=0.85, capsize=5)
    ax.text(i, np.mean(f1s) + np.std(f1s) + 0.01, f"{np.mean(f1s):.3f}", ha='center', fontsize=10, fontweight='bold')

ax.set_xticks(range(len(strategies)))
ax.set_xticklabels(strat_labels, fontsize=9)
ax.set_ylabel("F1 Score")
ax.set_ylim(0.8, 1.05)
ax.set_title("(A) Strategy Comparison: Mean F1")
ax.axhline(y=0.933, color='gray', linestyle='--', alpha=0.3)

# Panel B: FPR comparison
ax = axes[1]
for i, (strat, label, color) in enumerate(zip(strategies, strat_labels, colors)):
    fprs = [results[p][strat]["fpr"] for p in prompts]
    ax.bar(i, np.mean(fprs), yerr=np.std(fprs), color=color, alpha=0.85, capsize=5)
    ax.text(i, np.mean(fprs) + np.std(fprs) + 0.02, f"{np.mean(fprs):.3f}", ha='center', fontsize=10, fontweight='bold')

ax.set_xticks(range(len(strategies)))
ax.set_xticklabels(strat_labels, fontsize=9)
ax.set_ylabel("False Positive Rate")
ax.set_ylim(-0.05, 1.2)
ax.set_title("(B) FPR: Cross-Prompt Collapse vs Multi-Centroid Fix")

# Panel C: Per-prompt breakdown for nearest centroid
ax = axes[2]
short_names = [p[:10] for p in prompts]
x = np.arange(len(prompts))
w = 0.25
for i, (strat, label, color) in enumerate(zip(["same_prompt", "nearest_centroid", "min_distance"],
                                                ["Same (oracle)", "Nearest Centroid", "Min Distance"],
                                                ["#4CAF50", "#2196F3", "#FF9800"])):
    f1s = [results[p][strat]["f1"] for p in prompts]
    ax.bar(x + (i-1)*w, f1s, w, label=label, color=color, alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(short_names, fontsize=8, rotation=45, ha='right')
ax.set_ylabel("F1 Score")
ax.set_ylim(0.85, 1.0)
ax.legend(fontsize=9)
ax.set_title("(C) Per-Prompt F1: Nearest Centroid Matches Oracle")

plt.tight_layout()
plt.savefig("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/latex/figures/fig135_multi_centroid.png", dpi=150, bbox_inches='tight')
print("Saved fig135")
