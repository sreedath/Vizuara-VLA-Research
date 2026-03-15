import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/adversarial_detector_20260315_014756.json") as f:
    data = json.load(f)

results = data["results"]
cal_stats = data["cal_stats"]
threshold_l32 = cal_stats["L32"]["mean"] + 3 * cal_stats["L32"]["std"]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel A: Pixel MSE vs L32 distance for all attacks
ax = axes[0]
attacks = list(results.keys())
colors = ["#2196F3", "#FF9800", "#4CAF50", "#F44336", "#9C27B0"]
for attack, color in zip(attacks, colors):
    ar = results[attack]["results"]
    mses = [r["pixel_mse"] for r in ar]
    l32s = [r["L32_distance"] for r in ar]
    ax.scatter(mses, l32s, color=color, s=80, alpha=0.7, label=attack.replace("_", " "))
    ax.plot(mses, l32s, color=color, alpha=0.3)

ax.axhline(y=threshold_l32, color='red', linestyle='--', linewidth=2, label=f'L32 3σ threshold')
ax.set_xlabel("Pixel MSE (corruption severity)")
ax.set_ylabel("L32 Cosine Distance")
ax.set_title("(A) Pixel Corruption vs Embedding Distance")
ax.legend(fontsize=8)
# Shade evasion zone
ax.axhspan(0, threshold_l32, xmin=0.3, alpha=0.1, color='red')
ax.text(2000, threshold_l32/2, "Evasion\nZone", ha='center', fontsize=10, color='red', alpha=0.5)

# Panel B: Detection rate by attack type
ax = axes[1]
attack_labels = [a.replace("_", "\n") for a in attacks]
det_rates = []
for attack in attacks:
    ar = results[attack]["results"]
    total = len(ar)
    detected = sum(1 for r in ar if r["L3_flagged"] or r["L32_flagged"])
    det_rates.append(detected / total)

bars = ax.bar(range(len(attacks)), det_rates, color=colors, alpha=0.85)
ax.set_xticks(range(len(attacks)))
ax.set_xticklabels(attack_labels, fontsize=8)
ax.set_ylabel("Detection Rate")
ax.set_ylim(0, 1.1)
ax.set_title("(B) Detection Rate by Attack Type")
for bar, rate in zip(bars, det_rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{rate:.0%}", ha='center', fontsize=10, fontweight='bold')

# Panel C: High-freq attack detail (the one that evades)
ax = axes[2]
hf = results["high_freq"]["results"]
strengths = [r["strength"] for r in hf]
mses = [r["pixel_mse"] for r in hf]
l32s = [r["L32_distance"] for r in hf]

ax2 = ax.twinx()
bars1 = ax.bar([s - 2 for s in strengths], mses, 4, color='#90CAF9', alpha=0.7, label='Pixel MSE')
line1, = ax2.plot(strengths, l32s, 'o-', color='#F44336', markersize=8, linewidth=2, label='L32 distance')
ax2.axhline(y=threshold_l32, color='red', linestyle='--', alpha=0.5)

ax.set_xlabel("High-freq amplitude")
ax.set_ylabel("Pixel MSE", color='#2196F3')
ax2.set_ylabel("L32 Cosine Distance", color='#F44336')
ax.set_title("(C) High-Freq Attack: MSE Grows While Embedding Stays")

# Mark evasion region
for r in hf:
    if r["pixel_mse"] > 500 and not r["L32_flagged"]:
        ax.annotate("EVADES!", xy=(r["strength"], r["pixel_mse"]),
                    xytext=(r["strength"]+10, r["pixel_mse"]+500),
                    fontsize=9, color='red', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='red'))

lines = [bars1, line1]
labels = ['Pixel MSE', 'L32 distance']
ax.legend(lines, labels, loc='upper left', fontsize=8)

plt.tight_layout()
plt.savefig("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/latex/figures/fig140_adversarial.png", dpi=150, bbox_inches='tight')
print("Saved fig140")
