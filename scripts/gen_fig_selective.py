"""
Generate Fig 9: Real-model selective prediction — OOD rejection rate by uncertainty signal.
"""
import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_config_sel"
os.makedirs("/tmp/matplotlib_config_sel", exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

COLORS = {
    'blue': '#2171b5',
    'orange': '#e6550d',
    'green': '#31a354',
    'red': '#de2d26',
    'purple': '#756bb1',
    'gray': '#969696',
}


def fig9_selective_prediction():
    """Fig 9: OOD and Hard rejection rates by uncertainty signal at different coverage levels."""
    coverage = [100, 90, 80, 70, 60, 50, 40, 30]

    # OOD rejection rates from experiment
    ood_entropy = [0, 0, 16, 32, 44, 60, 80, 84]
    ood_confidence = [0, 0, 8, 24, 36, 48, 72, 88]
    ood_top5 = [0, 0, 8, 24, 40, 60, 68, 76]
    ood_margin = [0, 8, 8, 24, 48, 68, 80, 88]

    # Hard rejection rates
    hard_entropy = [0, 26.7, 42.2, 57.8, 66.7, 77.8, 82.2, 91.1]
    hard_max_entropy = [0, 26.7, 53.3, 77.8, 93.3, 95.6, 97.8, 97.8]
    hard_top5 = [0, 26.7, 48.9, 62.2, 80.0, 86.7, 93.3, 97.8]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: OOD rejection
    ax1.plot(coverage, ood_entropy, 'o-', color=COLORS['blue'], linewidth=2.5, markersize=7, label='Mean Entropy')
    ax1.plot(coverage, ood_confidence, 's--', color=COLORS['orange'], linewidth=2, markersize=6, label='Confidence')
    ax1.plot(coverage, ood_top5, '^-', color=COLORS['green'], linewidth=2, markersize=6, label='Top-5 Mass')
    ax1.plot(coverage, ood_margin, 'D:', color=COLORS['purple'], linewidth=2, markersize=6, label='Margin')

    ax1.set_xlabel('Coverage (%)', fontsize=12)
    ax1.set_ylabel('OOD Rejection Rate (%)', fontsize=12)
    ax1.set_title('(a) OOD Sample Rejection', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.set_xlim(25, 105)
    ax1.set_ylim(-5, 100)
    ax1.invert_xaxis()
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(alpha=0.3)

    # Annotate 50% coverage point
    ax1.axvline(x=50, color='gray', linestyle=':', alpha=0.5)
    ax1.annotate('60% at 50% cov.', xy=(50, 60), xytext=(65, 75),
                 arrowprops=dict(arrowstyle='->', color=COLORS['blue']),
                 fontsize=9, color=COLORS['blue'], fontweight='bold')

    # Right: Hard scenario rejection
    ax2.plot(coverage, hard_max_entropy, 'o-', color=COLORS['red'], linewidth=2.5, markersize=7, label='Max Dim Entropy')
    ax2.plot(coverage, hard_entropy, 's-', color=COLORS['blue'], linewidth=2, markersize=6, label='Mean Entropy')
    ax2.plot(coverage, hard_top5, '^-', color=COLORS['green'], linewidth=2, markersize=6, label='Top-5 Mass')

    ax2.set_xlabel('Coverage (%)', fontsize=12)
    ax2.set_ylabel('Hard Scenario Rejection Rate (%)', fontsize=12)
    ax2.set_title('(b) Hard Scenario Rejection', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.set_xlim(25, 105)
    ax2.set_ylim(-5, 105)
    ax2.invert_xaxis()
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(alpha=0.3)

    ax2.axvline(x=50, color='gray', linestyle=':', alpha=0.5)
    ax2.annotate('96% at 50% cov.', xy=(50, 95.6), xytext=(68, 80),
                 arrowprops=dict(arrowstyle='->', color=COLORS['red']),
                 fontsize=9, color=COLORS['red'], fontweight='bold')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig9_real_selective_prediction.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


def fig10_auroc_comparison():
    """Fig 10: AUROC comparison bar chart for different uncertainty signals."""
    signals = ['Mean\nEntropy', 'Neg Top-5\nMass', 'Neg\nMargin', 'Neg\nConfidence', 'Max Dim\nEntropy']
    aurocs = [0.786, 0.782, 0.741, 0.722, 0.618]
    colors = [COLORS['blue'], COLORS['green'], COLORS['purple'], COLORS['orange'], COLORS['red']]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    bars = ax.bar(range(len(signals)), aurocs, 0.6, color=colors, alpha=0.85,
                  edgecolor='black', linewidth=0.5)

    ax.set_ylabel('AUROC (Easy vs OOD)', fontsize=12)
    ax.set_xticks(range(len(signals)))
    ax.set_xticklabels(signals, fontsize=10)
    ax.set_ylim(0.5, 0.85)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    # Dashed line at 0.5 (random)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random (0.5)')

    # Value labels
    for bar, val in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.legend(fontsize=9, loc='upper right')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig10_auroc_signals.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


if __name__ == "__main__":
    fig9_selective_prediction()
    fig10_auroc_comparison()
    print("Selective prediction figures generated!")
