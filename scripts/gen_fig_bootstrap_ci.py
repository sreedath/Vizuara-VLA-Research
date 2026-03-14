"""Generate Figure 54: Bootstrap Confidence Intervals."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# Panel (a): Method AUROCs with 95% CIs — all OOD
ax = axes[0]
methods = ['Cosine', 'Attn Max', 'Attn Entropy', 'MSP', 'Energy']
aurocs = [1.000, 0.896, 0.873, 0.749, 0.379]
ci_lows = [1.000, 0.831, 0.803, 0.654, 0.273]
ci_highs = [1.000, 0.951, 0.932, 0.837, 0.488]

colors = ['#FF5722', '#4CAF50', '#2196F3', '#9E9E9E', '#9E9E9E']
y_pos = np.arange(len(methods))

for i, (m, a, lo, hi) in enumerate(zip(methods, aurocs, ci_lows, ci_highs)):
    ax.barh(i, a, 0.5, color=colors[i], alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.plot([lo, hi], [i, i], 'k-', linewidth=2)
    ax.plot([lo, lo], [i-0.15, i+0.15], 'k-', linewidth=2)
    ax.plot([hi, hi], [i-0.15, i+0.15], 'k-', linewidth=2)
    ax.text(max(hi, a) + 0.01, i, f'{a:.3f}\n[{lo:.3f}, {hi:.3f}]', va='center', fontsize=8)

ax.set_yticks(y_pos)
ax.set_yticklabels(methods, fontsize=10)
ax.set_xlabel('AUROC', fontsize=11)
ax.set_title('(a) Method AUROCs with 95% CIs\n(All OOD, N=10000 bootstrap)', fontsize=11, fontweight='bold')
ax.set_xlim(0.2, 1.18)
ax.axvline(x=0.5, color='red', linestyle=':', alpha=0.3)
ax.grid(True, alpha=0.3, axis='x')

# Panel (b): Pairwise comparison CIs
ax = axes[1]
pairs = ['Attn vs Cos\n(Far-OOD)', 'Attn vs Cos\n(Near-OOD)', 'Attn vs Cos\n(All OOD)',
         'Cos vs MSP\n(All OOD)', 'Attn vs MSP\n(All OOD)', 'Cos vs Energy\n(All OOD)']
diffs = [0.000, -0.166, -0.104, 0.251, 0.148, 0.621]
ci_lo_d = [0.000, -0.266, -0.169, 0.163, 0.047, 0.512]
ci_hi_d = [0.000, -0.079, -0.049, 0.346, 0.248, 0.727]
p_vals = [0.955, 0.000, 0.000, 0.000, 0.001, 0.000]

for i, (p, d, lo, hi, pv) in enumerate(zip(pairs, diffs, ci_lo_d, ci_hi_d, p_vals)):
    color = '#4CAF50' if lo > 0.001 else '#F44336' if hi < -0.001 else '#FF9800'
    ax.plot([lo, hi], [i, i], color=color, linewidth=3)
    ax.plot(d, i, 'ko', markersize=8)
    ax.plot([lo, lo], [i-0.15, i+0.15], color=color, linewidth=2)
    ax.plot([hi, hi], [i-0.15, i+0.15], color=color, linewidth=2)
    sig = '***' if pv < 0.001 else '**' if pv < 0.01 else '*' if pv < 0.05 else 'ns'
    ax.text(max(hi, 0) + 0.02, i, f'p={pv:.3f} {sig}', va='center', fontsize=8, fontweight='bold')

ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.set_yticks(range(len(pairs)))
ax.set_yticklabels(pairs, fontsize=9)
ax.set_xlabel('AUROC Difference (A − B)', fontsize=11)
ax.set_title('(b) Pairwise Comparisons\n(95% Bootstrap CIs)', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

from matplotlib.patches import Patch
legend_els = [Patch(facecolor='#4CAF50', label='A > B (sig.)'),
              Patch(facecolor='#F44336', label='B > A (sig.)'),
              Patch(facecolor='#FF9800', label='Not significant')]
ax.legend(handles=legend_els, fontsize=7, loc='lower right')

# Panel (c): Effect sizes
ax = axes[2]
signals = ['Cosine\nDistance', 'Attn\nMax', 'Attn\nEntropy']
cohens_d = [5.18, 1.20, 1.34]
colors_d = ['#FF5722', '#4CAF50', '#2196F3']

bars = ax.bar(range(len(signals)), cohens_d, 0.5, color=colors_d,
              edgecolor='black', linewidth=0.5, alpha=0.85)
ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Large effect (d=0.8)')
ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium effect (d=0.5)')
ax.set_xticks(range(len(signals)))
ax.set_xticklabels(signals, fontsize=10)
ax.set_ylabel("Cohen's d", fontsize=11)
ax.set_title("(c) Effect Sizes (ID vs OOD)", fontsize=11, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')
for bar, v in zip(bars, cohens_d):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
            f'd={v:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.annotate('All signals show\nlarge effect sizes', xy=(1, 1.2), xytext=(1.5, 3.5),
            fontsize=8, color='green', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='green'))

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig54_bootstrap_ci.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig54_bootstrap_ci.pdf', dpi=200, bbox_inches='tight')
print("Saved fig54_bootstrap_ci.png/pdf")
