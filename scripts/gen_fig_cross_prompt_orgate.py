import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/cross_prompt_orgate_20260315_011956.json") as f:
    data = json.load(f)

results = data["results"]
prompts = data["prompts"]
short_names = [p[:10] for p in prompts]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel A: F1 transfer matrix heatmap
ax = axes[0]
f1_matrix = np.zeros((len(prompts), len(prompts)))
for i, cal_p in enumerate(prompts):
    for j, inf_p in enumerate(prompts):
        f1_matrix[i, j] = results[cal_p][inf_p]["f1"]

im = ax.imshow(f1_matrix, cmap='RdYlGn', vmin=0.8, vmax=1.0, aspect='auto')
ax.set_xticks(range(len(prompts)))
ax.set_xticklabels(short_names, fontsize=8, rotation=45, ha='right')
ax.set_yticks(range(len(prompts)))
ax.set_yticklabels(short_names, fontsize=8)
ax.set_xlabel("Inference Prompt")
ax.set_ylabel("Calibration Prompt")
for i in range(len(prompts)):
    for j in range(len(prompts)):
        val = f1_matrix[i, j]
        color = 'white' if val < 0.9 else 'black'
        bold = 'bold' if i == j else 'normal'
        ax.text(j, i, f"{val:.3f}", ha='center', va='center', fontsize=9, color=color, fontweight=bold)
plt.colorbar(im, ax=ax, shrink=0.8, label='F1 Score')
ax.set_title("(A) Cross-Prompt F1 Transfer Matrix")

# Panel B: FPR matrix
ax = axes[1]
fpr_matrix = np.zeros((len(prompts), len(prompts)))
for i, cal_p in enumerate(prompts):
    for j, inf_p in enumerate(prompts):
        fpr_matrix[i, j] = results[cal_p][inf_p]["fpr"]

im = ax.imshow(fpr_matrix, cmap='RdYlGn_r', vmin=0, vmax=1, aspect='auto')
ax.set_xticks(range(len(prompts)))
ax.set_xticklabels(short_names, fontsize=8, rotation=45, ha='right')
ax.set_yticks(range(len(prompts)))
ax.set_yticklabels(short_names, fontsize=8)
ax.set_xlabel("Inference Prompt")
ax.set_ylabel("Calibration Prompt")
for i in range(len(prompts)):
    for j in range(len(prompts)):
        val = fpr_matrix[i, j]
        color = 'white' if val > 0.5 else 'black'
        bold = 'bold' if i == j else 'normal'
        ax.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=9, color=color, fontweight=bold)
plt.colorbar(im, ax=ax, shrink=0.8, label='False Positive Rate')
ax.set_title("(B) Cross-Prompt FPR Matrix")

# Panel C: Bar chart comparing same vs cross prompt
ax = axes[2]
same_f1 = [results[p][p]["f1"] for p in prompts]
cross_f1_mean = []
for cal_p in prompts:
    cross_vals = [results[cal_p][inf_p]["f1"] for inf_p in prompts if inf_p != cal_p]
    cross_f1_mean.append(np.mean(cross_vals))

x = np.arange(len(prompts))
w = 0.35
bars1 = ax.bar(x - w/2, same_f1, w, label='Same Prompt', color='#4CAF50', alpha=0.85)
bars2 = ax.bar(x + w/2, cross_f1_mean, w, label='Cross Prompt (mean)', color='#F44336', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(short_names, fontsize=8, rotation=45, ha='right')
ax.set_ylabel("F1 Score")
ax.set_ylim(0.8, 1.0)
ax.legend()
ax.set_title("(C) Same vs Cross-Prompt F1")

# Add annotations
for bar, val in zip(bars1, same_f1):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003, f"{val:.3f}",
            ha='center', fontsize=8)
for bar, val in zip(bars2, cross_f1_mean):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003, f"{val:.3f}",
            ha='center', fontsize=8)

plt.tight_layout()
plt.savefig("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/latex/figures/fig134_cross_prompt_orgate.png", dpi=150, bbox_inches='tight')
print("Saved fig134")
