"""Generate Figure 21: Cosine Distance Safety Pipeline Comparison."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# (a) Safety rate comparison across alpha levels
alphas = [0.05, 0.10, 0.15, 0.20, 0.30]
cos_safety = [100.0, 100.0, 100.0, 100.0, 100.0]
mass_safety = [60.8, 69.1, 75.3, 76.3, 81.4]

axes[0].plot(alphas, cos_safety, 'o-', color='#2ecc71', linewidth=3, markersize=10,
            label='Cosine Pipeline', zorder=5)
axes[0].plot(alphas, mass_safety, 's--', color='#e74c3c', linewidth=2.5, markersize=8,
            label='Mass Pipeline', zorder=4)

axes[0].fill_between(alphas, cos_safety, mass_safety, alpha=0.15, color='#2ecc71')

# Add easy throughput as secondary info
cos_throughput = [72.0, 52.0, 44.0, 32.0, 16.0]
ax_twin = axes[0].twinx()
ax_twin.plot(alphas, cos_throughput, 'D:', color='#3498db', linewidth=1.5, markersize=5,
            label='Cosine Easy→PROCEED')
ax_twin.set_ylabel('Easy Throughput (%)', fontsize=8, color='#3498db')
ax_twin.tick_params(axis='y', labelcolor='#3498db')
ax_twin.set_ylim(0, 110)

axes[0].set_xlabel('α (Conformal Level)', fontsize=10)
axes[0].set_ylabel('Safety Rate (%)', fontsize=10)
axes[0].set_title('(a) Safety Rate: Cosine vs Mass', fontsize=11, fontweight='bold')
axes[0].legend(fontsize=9, loc='center right')
axes[0].set_ylim(50, 105)
axes[0].set_xlim(0.03, 0.32)
axes[0].grid(True, alpha=0.3)

# Annotate the gap
axes[0].annotate('', xy=(0.15, 100), xytext=(0.15, 75.3),
                arrowprops=dict(arrowstyle='<->', color='black', lw=2))
axes[0].text(0.16, 87, 'Δ=24.7%', fontsize=9, fontweight='bold')

# Highlight sweet spot
axes[0].axvspan(0.04, 0.06, alpha=0.2, color='#f1c40f')
axes[0].annotate('Sweet spot\nα=0.05', xy=(0.05, 100), xytext=(0.12, 97),
                fontsize=8, fontweight='bold', color='#2ecc71',
                arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=1.5))

# (b) OOD→STOP comparison
cos_ood_stop = [100.0, 100.0, 100.0, 100.0, 100.0]
mass_ood_stop = [47.2, 58.3, 66.7, 68.1, 75.0]

axes[1].plot(alphas, cos_ood_stop, 'o-', color='#2ecc71', linewidth=3, markersize=10,
            label='Cosine Pipeline')
axes[1].plot(alphas, mass_ood_stop, 's--', color='#e74c3c', linewidth=2.5, markersize=8,
            label='Mass Pipeline')
axes[1].fill_between(alphas, cos_ood_stop, mass_ood_stop, alpha=0.15, color='#2ecc71')

axes[1].set_xlabel('α (Conformal Level)', fontsize=10)
axes[1].set_ylabel('OOD→STOP Rate (%)', fontsize=10)
axes[1].set_title('(b) OOD Detection Rate', fontsize=11, fontweight='bold')
axes[1].legend(fontsize=9, loc='center right')
axes[1].set_ylim(35, 105)
axes[1].set_xlim(0.03, 0.32)
axes[1].grid(True, alpha=0.3)

# Annotate
axes[1].annotate('Mass misses\n>50% of OOD\nat α=0.05', xy=(0.05, 47.2), xytext=(0.12, 50),
                fontsize=8, color='#e74c3c',
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5))

# (c) Per-scenario comparison at alpha=0.20
scenarios = ['Noise', 'Blank', 'Indoor', 'Inverted', 'Checker', 'Blackout']
cos_correct = [12, 12, 12, 12, 12, 12]  # All perfect
mass_correct = [11, 12, 3, 2, 9, 12]
n_per = 12

x = np.arange(len(scenarios))
width = 0.35

b1 = axes[2].bar(x - width/2, [c/n_per*100 for c in cos_correct], width,
                label='Cosine Pipeline', color='#2ecc71', edgecolor='black', linewidth=0.5)
b2 = axes[2].bar(x + width/2, [c/n_per*100 for c in mass_correct], width,
                label='Mass Pipeline', color='#e74c3c', edgecolor='black', linewidth=0.5)

axes[2].set_ylabel('Correct OOD→STOP (%)', fontsize=10)
axes[2].set_title('(c) Per-OOD-Type at α=0.20', fontsize=11, fontweight='bold')
axes[2].set_xticks(x)
axes[2].set_xticklabels(scenarios, fontsize=8)
axes[2].legend(fontsize=8)
axes[2].set_ylim(0, 115)
axes[2].grid(True, alpha=0.2, axis='y')

# Add value labels
for bar, val in zip(b1, cos_correct):
    axes[2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{val}/{n_per}', ha='center', va='bottom', fontsize=7, fontweight='bold',
                color='#2ecc71')
for bar, val in zip(b2, mass_correct):
    axes[2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{val}/{n_per}', ha='center', va='bottom', fontsize=7, fontweight='bold',
                color='#e74c3c')

# Highlight indoor and inverted
for i in [2, 3]:  # Indoor, Inverted
    axes[2].annotate('', xy=(i + width/2, mass_correct[i]/n_per*100 + 5),
                    xytext=(i + width/2, mass_correct[i]/n_per*100 + 20),
                    arrowprops=dict(arrowstyle='<-', color='#e74c3c', lw=2))

plt.tight_layout()
plt.savefig('/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures/fig21_cosine_safety_pipeline.png',
            dpi=300, bbox_inches='tight')
print("Saved fig21_cosine_safety_pipeline.png")
