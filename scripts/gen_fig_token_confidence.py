"""Generate Figure 113: Token Confidence Analysis."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/token_confidence_20260314_234219.json") as f:
    data = json.load(f)

cats = list(data['per_category'].keys())
groups = [data['per_category'][c]['group'] for c in cats]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel A: Confidence by category
ax = axes[0]
confs = [data['per_category'][c]['mean_confidence'] for c in cats]
conf_stds = [data['per_category'][c]['std_confidence'] for c in cats]
colors = ['#4CAF50' if g == 'ID' else '#F44336' for g in groups]
bars = ax.bar(range(len(cats)), confs, color=colors, alpha=0.7)
ax.errorbar(range(len(cats)), confs, yerr=conf_stds, fmt='none', color='black', capsize=3)
ax.set_xticks(range(len(cats)))
ax.set_xticklabels(cats, rotation=45, ha='right', fontsize=8)
ax.set_ylabel("Mean Token Confidence")
ax.set_title("(A) Confidence by Category")
ax.grid(True, alpha=0.3, axis='y')
# Add ID/OOD means
id_mean = data['detection']['id_conf_mean']
ood_mean = data['detection']['ood_conf_mean']
ax.axhline(y=id_mean, color='green', linestyle='--', alpha=0.5, label=f'ID mean={id_mean:.3f}')
ax.axhline(y=ood_mean, color='red', linestyle='--', alpha=0.5, label=f'OOD mean={ood_mean:.3f}')
ax.legend(fontsize=7)

# Panel B: Entropy by category
ax = axes[1]
ents = [data['per_category'][c]['mean_entropy'] for c in cats]
ent_stds = [data['per_category'][c]['std_entropy'] for c in cats]
bars = ax.bar(range(len(cats)), ents, color=colors, alpha=0.7)
ax.errorbar(range(len(cats)), ents, yerr=ent_stds, fmt='none', color='black', capsize=3)
ax.set_xticks(range(len(cats)))
ax.set_xticklabels(cats, rotation=45, ha='right', fontsize=8)
ax.set_ylabel("Mean Token Entropy")
ax.set_title("(B) Entropy by Category")
ax.grid(True, alpha=0.3, axis='y')
id_ent = data['detection']['id_ent_mean']
ood_ent = data['detection']['ood_ent_mean']
ax.axhline(y=id_ent, color='green', linestyle='--', alpha=0.5, label=f'ID mean={id_ent:.3f}')
ax.axhline(y=ood_ent, color='red', linestyle='--', alpha=0.5, label=f'OOD mean={ood_ent:.3f}')
ax.legend(fontsize=7)

# Panel C: Detection comparison
ax = axes[2]
methods = ['Cosine Distance\n(Embedding)', 'Confidence\n(Token)', 'Entropy\n(Token)']
aurocs = [1.0, data['detection']['confidence_auroc'], data['detection']['entropy_auroc']]
ds = [44.7, data['detection']['confidence_d'], data['detection']['entropy_d']]

x = np.arange(len(methods))
width = 0.35
bars1 = ax.bar(x - width/2, aurocs, width, color='#2196F3', alpha=0.7, label='AUROC')
ax2 = ax.twinx()
bars2 = ax2.bar(x + width/2, ds, width, color='#FF9800', alpha=0.7, label="D-prime")

ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=8)
ax.set_ylabel("AUROC", color='#2196F3')
ax2.set_ylabel("D-prime", color='#FF9800')
ax.set_ylim(0, 1.1)
ax.set_title("(C) Detection Method Comparison")

# Add values
for bar, v in zip(bars1, aurocs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{v:.3f}", ha='center', fontsize=8, color='#2196F3')
for bar, v in zip(bars2, ds):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{v:.1f}", ha='center', fontsize=8, color='#FF9800')

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

plt.suptitle("Token Confidence Analysis (Exp 127)\nConfidence/entropy are NOT effective OOD detectors (AUROC~0.56)",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/latex/figures/fig113_token_confidence.png",
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig113")
