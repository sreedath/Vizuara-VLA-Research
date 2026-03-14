"""
Generate Fig 11: Dropout rate sensitivity — AUROC vs dropout rate and token agreement.
"""
import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_config_ds"
os.makedirs("/tmp/matplotlib_config_ds", exist_ok=True)

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


def fig11_dropout_sensitivity():
    """Fig 11: AUROC and MC variance vs dropout rate."""
    dropout_rates = [0.0, 0.01, 0.05, 0.10, 0.15, 0.20, 0.30]

    # From experiment results
    auroc_entropy = [0.590, 0.733, 0.675, 0.810, 0.877, 0.932, 0.725]
    auroc_mc_std = [0.500, 0.598, 0.555, 0.643, 0.675, 0.575, 0.533]
    conf_std = [0.0000, 0.0821, 0.0832, 0.0838, 0.0885, 0.0867, 0.0833]
    tok_agree = [1.00, 0.002, 0.00, 0.00, 0.00, 0.00, 0.00]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: AUROC vs dropout rate
    ax1.plot(dropout_rates, auroc_entropy, 'o-', color=COLORS['blue'], linewidth=2.5,
             markersize=8, label='AUROC (Entropy)', zorder=5)
    ax1.plot(dropout_rates, auroc_mc_std, 's--', color=COLORS['orange'], linewidth=2,
             markersize=7, label='AUROC (MC Conf Std)', zorder=4)

    # Highlight optimal
    best_idx = np.argmax(auroc_entropy)
    ax1.scatter([dropout_rates[best_idx]], [auroc_entropy[best_idx]], s=200,
                facecolors='none', edgecolors=COLORS['red'], linewidths=2.5, zorder=6)
    ax1.annotate(f'Best: p={dropout_rates[best_idx]}\nAUROC={auroc_entropy[best_idx]:.3f}',
                 xy=(dropout_rates[best_idx], auroc_entropy[best_idx]),
                 xytext=(0.25, 0.85),
                 arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=1.5),
                 fontsize=10, color=COLORS['red'], fontweight='bold')

    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random (0.5)')
    ax1.set_xlabel('Dropout Rate (p)', fontsize=12)
    ax1.set_ylabel('AUROC (Easy vs OOD)', fontsize=12)
    ax1.set_title('(a) OOD Detection vs Dropout Rate', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='lower left')
    ax1.set_xlim(-0.02, 0.32)
    ax1.set_ylim(0.45, 1.0)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(alpha=0.3)

    # Right: MC variance and token agreement
    ax2_twin = ax2.twinx()

    bars = ax2.bar(np.array(dropout_rates) - 0.008, conf_std, width=0.016,
                   color=COLORS['green'], alpha=0.7, label='MC Conf Std', zorder=3)
    line = ax2_twin.plot(dropout_rates, tok_agree, 'D-', color=COLORS['red'],
                         linewidth=2.5, markersize=8, label='Token Agreement', zorder=5)

    ax2.set_xlabel('Dropout Rate (p)', fontsize=12)
    ax2.set_ylabel('MC Confidence Std', fontsize=12, color=COLORS['green'])
    ax2_twin.set_ylabel('Token Agreement', fontsize=12, color=COLORS['red'])
    ax2.set_title('(b) MC Variance & Prediction Stability', fontsize=13, fontweight='bold')

    ax2.set_xlim(-0.02, 0.32)
    ax2.set_ylim(0, 0.12)
    ax2_twin.set_ylim(-0.05, 1.1)

    # Annotate the cliff
    ax2.annotate('Token agreement\ndrops to 0 at p=0.01',
                 xy=(0.01, 0.082), xytext=(0.10, 0.105),
                 arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=1.5),
                 fontsize=9, color=COLORS['red'], fontweight='bold')

    ax2.spines['top'].set_visible(False)
    ax2_twin.spines['top'].set_visible(False)
    ax2.grid(alpha=0.3)

    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='center right')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig11_dropout_sensitivity.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


if __name__ == "__main__":
    fig11_dropout_sensitivity()
    print("Dropout sensitivity figure generated!")
