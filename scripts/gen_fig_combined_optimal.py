"""
Generate Fig 12: Combined optimal UQ — AUROC comparison and selective prediction with optimal dropout.
"""
import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_config_co"
os.makedirs("/tmp/matplotlib_config_co", exist_ok=True)

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
    'teal': '#1b9e77',
}


def fig12_combined_results():
    """Fig 12: Combined UQ AUROC comparison and selective prediction."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: AUROC bar chart comparing signals (Easy vs OOD and Easy vs Hard)
    signals = ['Raw\nEntropy', 'Neg\nConfidence', 'Cal\nEntropy', 'MC Conf\nStd',
               'Prompt\nDisagree', 'Combined\n(RMS)']
    auroc_ood = [0.873, 0.823, 0.793, 0.576, 0.560, 0.502]
    auroc_hard = [0.627, 0.637, 0.487, 0.559, 0.541, 0.530]

    x = np.arange(len(signals))
    width = 0.35

    bars1 = ax1.bar(x - width/2, auroc_ood, width, color=COLORS['blue'], alpha=0.85,
                    edgecolor='black', linewidth=0.5, label='Easy vs OOD')
    bars2 = ax1.bar(x + width/2, auroc_hard, width, color=COLORS['orange'], alpha=0.85,
                    edgecolor='black', linewidth=0.5, label='Easy vs Hard')

    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax1.set_ylabel('AUROC', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(signals, fontsize=9)
    ax1.set_ylim(0.35, 0.95)
    ax1.legend(fontsize=10, loc='upper right')
    ax1.set_title('(a) Signal Comparison (p=0.20, per-dim T)', fontsize=13, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='y', alpha=0.3)

    # Value labels for OOD
    for bar, val in zip(bars1, auroc_ood):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.008,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold',
                 color=COLORS['blue'])

    # Right: Selective prediction — OOD rejection curves
    coverage = [100, 90, 80, 70, 60, 50]

    raw_entropy_ood = [0, 26.7, 40.0, 63.3, 80.0, 86.7]
    cal_entropy_ood = [0, 20.0, 50.0, 60.0, 76.7, 83.3]
    neg_conf_ood = [0, 23.3, 40.0, 56.7, 66.7, 83.3]
    mc_std_ood = [0, 13.3, 16.7, 33.3, 46.7, 50.0]
    combined_ood = [0, 6.7, 16.7, 26.7, 33.3, 53.3]

    ax2.plot(coverage, raw_entropy_ood, 'o-', color=COLORS['blue'], linewidth=2.5,
             markersize=7, label='Raw Entropy')
    ax2.plot(coverage, cal_entropy_ood, 's-', color=COLORS['teal'], linewidth=2,
             markersize=6, label='Cal Entropy')
    ax2.plot(coverage, neg_conf_ood, '^-', color=COLORS['orange'], linewidth=2,
             markersize=6, label='Neg Confidence')
    ax2.plot(coverage, mc_std_ood, 'D--', color=COLORS['green'], linewidth=2,
             markersize=6, label='MC Conf Std')
    ax2.plot(coverage, combined_ood, 'v:', color=COLORS['gray'], linewidth=2,
             markersize=6, label='Combined (RMS)')

    ax2.set_xlabel('Coverage (%)', fontsize=12)
    ax2.set_ylabel('OOD Rejection Rate (%)', fontsize=12)
    ax2.set_title('(b) Selective Prediction (Optimal p=0.20)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9, loc='lower left')
    ax2.set_xlim(45, 105)
    ax2.set_ylim(-5, 100)
    ax2.invert_xaxis()
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(alpha=0.3)

    # Annotate 50% coverage
    ax2.axvline(x=50, color='gray', linestyle=':', alpha=0.5)
    ax2.annotate('87% OOD rejected\nat 50% coverage',
                 xy=(50, 86.7), xytext=(70, 70),
                 arrowprops=dict(arrowstyle='->', color=COLORS['blue'], lw=1.5),
                 fontsize=9, color=COLORS['blue'], fontweight='bold')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig12_combined_optimal.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


if __name__ == "__main__":
    fig12_combined_results()
    print("Combined optimal figure generated!")
