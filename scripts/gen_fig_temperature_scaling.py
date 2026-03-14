"""Generate Figure 79: Temperature Scaling Effect on OOD Detection."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/temperature_scaling_20260314_204959.json") as f:
    data = json.load(f)

temps = data['temperatures']
tr = data['results']['temperature_results']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel (a): AUROC vs temperature for each output feature
ax = axes[0]
for feat, color, marker, label in [('entropy', '#F44336', 'o', 'Entropy'),
                                     ('top_prob', '#2196F3', 's', 'Top-1 Prob'),
                                     ('top5_prob', '#4CAF50', '^', 'Top-5 Prob')]:
    aurocs = [tr[f'T_{T}_{feat}']['auroc'] for T in temps]
    ax.plot(temps, aurocs, f'{marker}-', color=color, linewidth=2, markersize=6, label=label, alpha=0.8)

ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label='Hidden state (1.000)')
ax.axhline(y=0.5, color='black', linestyle=':', alpha=0.3, linewidth=1)
ax.set_xscale('log')
ax.set_xlabel('Temperature', fontsize=11)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(a) AUROC vs Temperature', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.35, 1.05)

# Panel (b): ID vs OOD entropy at different temperatures
ax = axes[1]
id_means = [tr[f'T_{T}_entropy']['id_mean'] for T in temps]
ood_means = [tr[f'T_{T}_entropy']['ood_mean'] for T in temps]
id_stds = [tr[f'T_{T}_entropy']['id_std'] for T in temps]
ood_stds = [tr[f'T_{T}_entropy']['ood_std'] for T in temps]

ax.errorbar(temps, id_means, yerr=id_stds, fmt='o-', color='#4CAF50',
            linewidth=2, markersize=6, label='ID', alpha=0.8, capsize=3)
ax.errorbar(temps, ood_means, yerr=ood_stds, fmt='s-', color='#F44336',
            linewidth=2, markersize=6, label='OOD', alpha=0.8, capsize=3)
ax.set_xscale('log')
ax.set_xlabel('Temperature', fontsize=11)
ax.set_ylabel('Output Entropy', fontsize=11)
ax.set_title('(b) ID vs OOD Entropy', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel (c): Summary comparison — hidden state vs best output
ax = axes[2]
methods = ['Hidden\nState', 'Entropy\n(T=0.1)', 'Top-1\n(T=0.25)', 'Top-5\n(T=0.5)']
aurocs = [1.0, 0.747, 0.729, 0.661]
colors = ['#4CAF50', '#F44336', '#FF9800', '#FF9800']
bars = ax.bar(range(len(methods)), aurocs, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods, fontsize=10)
ax.set_ylabel('AUROC', fontsize=11)
ax.set_title('(c) Best Feature per Method', fontsize=12, fontweight='bold')
ax.set_ylim(0, 1.1)
ax.axhline(y=0.5, color='black', linestyle=':', alpha=0.3, linewidth=1)
ax.grid(True, alpha=0.3, axis='y')

for i, a in enumerate(aurocs):
    ax.text(i, a + 0.02, f'{a:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig79_temperature_scaling.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig79_temperature_scaling.pdf', dpi=200, bbox_inches='tight')
print("Saved fig79_temperature_scaling.png/pdf")
