"""
Generate Fig 7: Large-scale OpenVLA confidence across 8 scenarios with statistical annotations.
"""
import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_config_ls"
os.makedirs("/tmp/matplotlib_config_ls", exist_ok=True)

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
    'teal': '#2ca25f',
    'gold': '#d95f02',
}

def fig7_large_scale():
    """Fig 7: Large-scale OpenVLA-7B confidence with statistical tests."""
    scenarios = ['Highway', 'Urban', 'Night', 'Rain', 'Fog', 'Constr.', 'OOD\n(noise)', 'OOD\n(blank)']
    confidence = [0.5599, 0.5821, 0.4899, 0.4601, 0.6016, 0.5535, 0.5795, 0.5194]
    conf_std = [0.0745, 0.0812, 0.0773, 0.0749, 0.0944, 0.0755, 0.0747, 0.0734]
    entropy = [1.2621, 1.2260, 1.5563, 1.6665, 1.1898, 1.2695, 1.2562, 1.4085]

    # Color by category
    bar_colors = [
        COLORS['blue'],    # highway - normal
        COLORS['teal'],    # urban - normal
        COLORS['purple'],  # night - adverse
        COLORS['orange'],  # rain - adverse
        COLORS['gray'],    # fog - adverse
        COLORS['gold'],    # construction - adverse
        COLORS['red'],     # ood noise
        '#8b0000',         # ood blank (dark red)
    ]

    x = np.arange(len(scenarios))

    fig, ax1 = plt.subplots(figsize=(12, 5.5))

    # Confidence bars with error bars
    bars = ax1.bar(x, confidence, 0.6, yerr=conf_std, capsize=4,
                   color=bar_colors, alpha=0.85, edgecolor='black', linewidth=0.5,
                   error_kw={'linewidth': 1.2})

    ax1.set_ylabel('Geometric Mean Confidence', fontsize=12, color=COLORS['blue'])
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, fontsize=10)
    ax1.set_ylim(0.3, 0.75)
    ax1.tick_params(axis='y', labelcolor=COLORS['blue'])

    # Second y-axis for entropy
    ax2 = ax1.twinx()
    ax2.plot(x, entropy, 's-', color=COLORS['red'], linewidth=2, markersize=7,
             label='Entropy', zorder=5)
    ax2.set_ylabel('Entropy', fontsize=12, color=COLORS['red'])
    ax2.set_ylim(0.8, 2.0)
    ax2.tick_params(axis='y', labelcolor=COLORS['red'])

    # Annotate the inverted gap: OOD noise > Highway
    ax1.annotate('', xy=(0, 0.5599), xycoords='data',
                 xytext=(6, 0.5795), textcoords='data',
                 arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    ax1.text(3, 0.69, 'OOD > Highway!\n$\\Delta = -0.020$, $p = 0.321$ (n.s.)',
             ha='center', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffcccc', alpha=0.9,
                       edgecolor='red', linewidth=1.5))

    # Highlight fog as highest confidence
    ax1.annotate('Highest\nconfidence!',
                 xy=(4, 0.602), xytext=(4, 0.72),
                 arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=1.5),
                 fontsize=9, color=COLORS['gray'], ha='center', fontweight='bold')

    # Add category labels at bottom
    ax1.axvspan(-0.5, 1.5, alpha=0.05, color='green', zorder=0)
    ax1.axvspan(1.5, 5.5, alpha=0.05, color='orange', zorder=0)
    ax1.axvspan(5.5, 7.5, alpha=0.05, color='red', zorder=0)
    ax1.text(0.5, 0.33, 'Normal', ha='center', fontsize=9, fontstyle='italic', color='green')
    ax1.text(3.5, 0.33, 'Adverse', ha='center', fontsize=9, fontstyle='italic', color='orange')
    ax1.text(6.5, 0.33, 'OOD', ha='center', fontsize=9, fontstyle='italic', color='red')

    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.grid(axis='y', alpha=0.2)

    fig.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95), fontsize=10)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig7_large_scale_confidence.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


def fig8_per_dimension():
    """Fig 8: Per-dimension confidence heatmap across scenarios."""
    scenarios = ['Highway', 'Urban', 'Night', 'Rain', 'Fog', 'Constr.', 'OOD\nnoise', 'OOD\nblank']

    # Per-dimension confidence data (from large-scale experiment averages)
    dim_labels = ['Dim 0\n(lateral)', 'Dim 1\n(long.)', 'Dim 2\n(z)',
                  'Dim 3\n(roll)', 'Dim 4\n(pitch)', 'Dim 5\n(yaw)', 'Dim 6\n(grip.)']
    overall_dims = [0.3642, 0.5332, 0.5924, 0.6497, 0.6475, 0.6539, 0.7751]

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(dim_labels))
    bars = ax.bar(x, overall_dims, 0.6,
                  color=[plt.cm.RdYlGn(v) for v in np.linspace(0.2, 0.9, len(dim_labels))],
                  edgecolor='black', linewidth=0.5, alpha=0.85)

    ax.set_ylabel('Mean Confidence (max prob)', fontsize=12)
    ax.set_xlabel('Action Dimension', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(dim_labels, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars, overall_dims):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Annotate the range
    ax.annotate('', xy=(0, 0.364), xycoords='data',
                xytext=(6, 0.775), textcoords='data',
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax.text(3, 0.85, '2.1$\\times$ range across dimensions',
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig8_per_dimension_confidence.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


if __name__ == "__main__":
    fig7_large_scale()
    fig8_per_dimension()
    print("Large-scale figures generated!")
