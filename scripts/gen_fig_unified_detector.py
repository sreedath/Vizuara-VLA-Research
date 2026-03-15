import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/unified_detector_20260315_013014.json") as f:
    data = json.load(f)

results = data["results"]
prompts = data["prompts"]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel A: F1 comparison
ax = axes[0]
detectors = ["cosine_same", "recon_same", "cosine_nearest", "recon_nearest"]
det_labels = ["Cosine\nSame-Prompt", "PCA-Recon\nSame-Prompt", "Cosine\nNearest-Centroid", "PCA-Recon\nNearest-Centroid"]
colors = ["#4CAF50", "#F44336", "#2196F3", "#FF9800"]

for i, (det, label, color) in enumerate(zip(detectors, det_labels, colors)):
    f1s = [results[p][det]["f1"] for p in prompts]
    ax.bar(i, np.mean(f1s), yerr=np.std(f1s), color=color, alpha=0.85, capsize=5)
    ax.text(i, np.mean(f1s) + np.std(f1s) + 0.01, f"{np.mean(f1s):.3f}", ha='center', fontsize=9, fontweight='bold')

ax.set_xticks(range(len(detectors)))
ax.set_xticklabels(det_labels, fontsize=8)
ax.set_ylabel("F1 Score")
ax.set_ylim(0.8, 1.05)
ax.set_title("(A) Detector F1 Comparison")

# Panel B: Recall vs FPR scatter
ax = axes[1]
for det, label, color, marker in zip(detectors, det_labels, colors, ['o', 's', '^', 'D']):
    for p in prompts:
        r = results[p][det]
        ax.scatter(r["fpr"], r["recall"], color=color, marker=marker, s=80, alpha=0.7)
    # Mean point
    mean_fpr = np.mean([results[p][det]["fpr"] for p in prompts])
    mean_rec = np.mean([results[p][det]["recall"] for p in prompts])
    ax.scatter(mean_fpr, mean_rec, color=color, marker=marker, s=200, edgecolors='black', linewidth=2,
               label=label.replace('\n', ' '))

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("Recall")
ax.set_title("(B) Recall vs FPR Per Prompt")
ax.legend(fontsize=7, loc='lower right')
ax.set_xlim(-0.05, 1.1)

# Panel C: Per-prompt F1 for top detectors
ax = axes[2]
short_names = [p[:10] for p in prompts]
x = np.arange(len(prompts))
w = 0.35
for i, (det, label, color) in enumerate(zip(["cosine_same", "cosine_nearest"],
                                              ["Cosine Same", "Cosine Nearest"],
                                              ["#4CAF50", "#2196F3"])):
    f1s = [results[p][det]["f1"] for p in prompts]
    ax.bar(x + (i-0.5)*w, f1s, w, label=label, color=color, alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(short_names, fontsize=8, rotation=45, ha='right')
ax.set_ylabel("F1 Score")
ax.set_ylim(0.9, 1.0)
ax.legend()
ax.set_title("(C) Best Detectors Per Prompt")

plt.tight_layout()
plt.savefig("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/latex/figures/fig137_unified_detector.png", dpi=150, bbox_inches='tight')
print("Saved fig137")
