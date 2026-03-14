"""
Generate Fig 14: Input perturbation sensitivity — token agreement and confidence shift.
"""
import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_config_pt"
os.makedirs("/tmp/matplotlib_config_pt", exist_ok=True)

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


def fig14_perturbation():
    """Fig 14: Perturbation sensitivity."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    sigma = [0, 1, 5, 10, 25, 50]

    # Token agreement vs noise
    highway_agree = [1.000, 0.257, 0.143, 0.086, 0.057, 0.043]
    urban_agree = [1.000, 0.543, 0.557, 0.329, 0.171, 0.157]
    night_agree = [1.000, 0.343, 0.114, 0.100, 0.029, 0.043]
    ood_agree = [1.000, 0.829, 0.443, 0.471, 0.214, 0.114]

    ax1.plot(sigma, highway_agree, 'o-', color=COLORS['blue'], linewidth=2.5, markersize=7, label='Highway')
    ax1.plot(sigma, urban_agree, 's-', color=COLORS['green'], linewidth=2, markersize=6, label='Urban')
    ax1.plot(sigma, night_agree, '^-', color=COLORS['red'], linewidth=2, markersize=6, label='Night')
    ax1.plot(sigma, ood_agree, 'D-', color=COLORS['purple'], linewidth=2, markersize=6, label='OOD Noise')

    ax1.set_xlabel('Gaussian Noise σ', fontsize=12)
    ax1.set_ylabel('Token Agreement with Clean', fontsize=12)
    ax1.set_title('(a) Action Stability vs Noise', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.set_ylim(-0.05, 1.1)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(alpha=0.3)

    # Annotate σ=1 collapse
    ax1.annotate('σ=1: 49% tokens\nalready changed!',
                 xy=(1, 0.493), xytext=(15, 0.7),
                 arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=1.5),
                 fontsize=9, color=COLORS['red'], fontweight='bold')

    # Confidence shift vs noise
    highway_conf = [0.493, 0.500, 0.564, 0.565, 0.581, 0.592]
    night_conf = [0.196, 0.218, 0.254, 0.369, 0.507, 0.560]
    urban_conf = [0.707, 0.692, 0.710, 0.590, 0.643, 0.625]
    ood_conf = [0.558, 0.558, 0.614, 0.602, 0.606, 0.571]

    ax2.plot(sigma, highway_conf, 'o-', color=COLORS['blue'], linewidth=2.5, markersize=7, label='Highway')
    ax2.plot(sigma, urban_conf, 's-', color=COLORS['green'], linewidth=2, markersize=6, label='Urban')
    ax2.plot(sigma, night_conf, '^-', color=COLORS['red'], linewidth=2, markersize=6, label='Night')
    ax2.plot(sigma, ood_conf, 'D-', color=COLORS['purple'], linewidth=2, markersize=6, label='OOD Noise')

    ax2.set_xlabel('Gaussian Noise σ', fontsize=12)
    ax2.set_ylabel('Geometric Mean Confidence', fontsize=12)
    ax2.set_title('(b) Confidence Shift with Noise', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.set_ylim(0.1, 0.8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(alpha=0.3)

    # Annotate night confidence increase
    ax2.annotate('Night: noise INCREASES\nconfidence 3×!',
                 xy=(50, 0.560), xytext=(20, 0.35),
                 arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=1.5),
                 fontsize=9, color=COLORS['red'], fontweight='bold')

    # Per-dimension stability at σ=10
    dims = ['Dim 0\n(lateral)', 'Dim 1', 'Dim 2', 'Dim 3', 'Dim 4', 'Dim 5', 'Dim 6\n(gripper)']
    dim_agree = [0.600, 0.250, 0.200, 0.200, 0.150, 0.125, 0.200]
    dim_colors = [COLORS['green']] + [COLORS['orange']] * 5 + [COLORS['orange']]

    bars = ax3.bar(range(7), dim_agree, 0.6, color=dim_colors, alpha=0.8,
                   edgecolor='black', linewidth=0.5)

    ax3.set_xticks(range(7))
    ax3.set_xticklabels(dims, fontsize=8)
    ax3.set_ylabel('Token Agreement at σ=10', fontsize=12)
    ax3.set_title('(c) Per-Dimension Stability', fontsize=13, fontweight='bold')
    ax3.set_ylim(0, 0.75)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, dim_agree):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax3.axhline(y=1/7, color='gray', linestyle='--', alpha=0.5, label='Random (1/7)')
    ax3.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig14_perturbation_sensitivity.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


if __name__ == "__main__":
    fig14_perturbation()
    print("Perturbation figure generated!")
