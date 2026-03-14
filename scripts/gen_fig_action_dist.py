"""
Generate Fig 13: Action distribution analysis — bin utilization and KL divergence.
"""
import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_config_ad"
os.makedirs("/tmp/matplotlib_config_ad", exist_ok=True)

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
    'brown': '#8c510a',
}


def fig13_action_distribution():
    """Fig 13: Action distribution analysis."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Left: Per-scenario perplexity and effective support
    scenarios = ['Highway', 'Urban', 'Night', 'Rain', 'Fog', 'Constr.', 'OOD\nNoise', 'OOD\nBlank']
    perplexities = [5.6, 2.8, 25.8, 10.4, 5.7, 3.5, 4.6, 7.8]
    eff_support = [17.4, 10.4, 53.7, 29.6, 17.7, 12.2, 16.1, 22.7]
    colors_scenario = [COLORS['blue'], COLORS['blue'], COLORS['red'], COLORS['red'],
                       COLORS['red'], COLORS['red'], COLORS['purple'], COLORS['purple']]

    ax1 = axes[0]
    x = np.arange(len(scenarios))
    bars = ax1.bar(x, eff_support, 0.6, color=colors_scenario, alpha=0.8,
                   edgecolor='black', linewidth=0.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, fontsize=9)
    ax1.set_ylabel('Effective Support (bins > 0.1%)', fontsize=11)
    ax1.set_title('(a) Bin Utilization by Scenario', fontsize=13, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='y', alpha=0.3)

    # Value labels
    for bar, val in zip(bars, eff_support):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                 f'{val:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Annotate night
    ax1.annotate('Night: 6× more bins\nthan Urban', xy=(2, 53.7), xytext=(4.5, 50),
                 arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=1.5),
                 fontsize=9, color=COLORS['red'], fontweight='bold')

    # Middle: Per-dimension perplexity
    ax2 = axes[1]
    dims = ['Dim 0\n(lateral)', 'Dim 1', 'Dim 2', 'Dim 3', 'Dim 4', 'Dim 5', 'Dim 6\n(gripper)']
    dim_perp = [24.5, 7.6, 4.5, 4.4, 6.2, 4.5, 2.9]
    dim_support = [54.4, 23.3, 16.2, 15.1, 19.4, 13.7, 9.5]
    dim_colors = [COLORS['red']] + [COLORS['blue']] * 5 + [COLORS['green']]

    bars2 = ax2.bar(range(7), dim_perp, 0.6, color=dim_colors, alpha=0.8,
                    edgecolor='black', linewidth=0.5)

    ax2.set_xticks(range(7))
    ax2.set_xticklabels(dims, fontsize=8)
    ax2.set_ylabel('Mean Perplexity', fontsize=11)
    ax2.set_title('(b) Per-Dimension Distribution Width', fontsize=13, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars2, dim_perp):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2.annotate('8.5× wider than\ngripper', xy=(0, 24.5), xytext=(2, 22),
                 arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=1.5),
                 fontsize=9, color=COLORS['red'], fontweight='bold')

    # Right: KL divergence heatmap (simplified)
    ax3 = axes[2]
    scenario_short = ['Hwy', 'Urban', 'Night', 'Rain', 'Fog', 'Constr', 'OOD_N', 'OOD_B']
    kl_matrix = np.array([
        [0.00, 5.61, 3.01, 3.22, 4.02, 5.02, 4.37, 3.24],
        [3.18, 0.00, 4.58, 4.63, 5.58, 4.47, 5.80, 4.56],
        [4.19, 7.50, 0.00, 2.54, 3.90, 5.33, 2.88, 3.15],
        [3.95, 7.53, 2.04, 0.00, 3.92, 5.33, 3.74, 2.75],
        [3.84, 8.27, 2.13, 3.54, 0.00, 5.22, 3.27, 3.22],
        [3.55, 6.47, 3.68, 3.62, 4.53, 0.00, 4.48, 3.90],
        [4.36, 7.90, 2.09, 3.02, 4.00, 4.69, 0.00, 2.93],
        [3.75, 7.36, 2.33, 2.82, 3.49, 4.99, 3.38, 0.00],
    ])

    im = ax3.imshow(kl_matrix, cmap='YlOrRd', vmin=0, vmax=8)
    ax3.set_xticks(range(8))
    ax3.set_yticks(range(8))
    ax3.set_xticklabels(scenario_short, fontsize=8, rotation=45, ha='right')
    ax3.set_yticklabels(scenario_short, fontsize=8)
    ax3.set_title('(c) KL Divergence Between\nScenario Distributions', fontsize=13, fontweight='bold')

    # Add text annotations for key values
    for i in range(8):
        for j in range(8):
            color = 'white' if kl_matrix[i, j] > 5 else 'black'
            ax3.text(j, i, f'{kl_matrix[i, j]:.1f}', ha='center', va='center',
                     fontsize=6.5, color=color)

    plt.colorbar(im, ax=ax3, shrink=0.8, label='KL(row || col)')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig13_action_distribution.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


if __name__ == "__main__":
    fig13_action_distribution()
    print("Action distribution figure generated!")
