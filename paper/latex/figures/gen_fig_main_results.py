"""
Generate publication-quality grouped bar chart comparing OOD detection
AUROC across methods and corruption types for NeurIPS paper.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
methods = [
    "MSP",
    "Energy",
    "Entropy",
    "Top-k\nProb",
    "Feature\nNorm",
    "Cosine Dist.\n(Ours)",
]

corruption_types = ["Fog", "Night", "Noise", "Blur", "Mean"]

# rows = methods, cols = corruption types (same order as above)
data = np.array([
    [0.36, 0.72, 0.54, 0.84, 0.62],  # MSP
    [0.00, 0.72, 0.30, 1.00, 0.51],  # Energy
    [0.28, 0.76, 0.54, 0.98, 0.64],  # Entropy
    [0.28, 0.78, 0.60, 0.96, 0.66],  # Top-k Prob
    [1.00, 1.00, 0.84, 1.00, 0.96],  # Feature Norm
    [1.00, 1.00, 1.00, 1.00, 1.00],  # Cosine Distance (Ours)
])

n_methods = len(methods)
n_groups = len(corruption_types)

# ---------------------------------------------------------------------------
# Colorblind-friendly palette (Wong 2011 / IBM Design Library)
# ---------------------------------------------------------------------------
colors = [
    "#648FFF",  # Fog  – blue
    "#785EF0",  # Night – violet
    "#DC267F",  # Noise – magenta
    "#FE6100",  # Blur  – orange
    "#222222",  # Mean  – near-black
]

# ---------------------------------------------------------------------------
# Figure setup
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "mathtext.default": "regular",
})

fig, ax = plt.subplots(figsize=(10, 4))

bar_width = 0.14
group_gap = 0.12  # extra space between method groups
x_positions = np.arange(n_methods) * (n_groups * bar_width + group_gap)

for j, (corr, color) in enumerate(zip(corruption_types, colors)):
    offsets = x_positions + j * bar_width
    values = data[:, j]

    # Make "Cosine Distance (Ours)" bars visually distinct with hatching
    for i, (xpos, val) in enumerate(zip(offsets, values)):
        is_ours = (i == n_methods - 1)
        bar = ax.bar(
            xpos,
            val,
            width=bar_width,
            color=color,
            edgecolor="white" if not is_ours else "#333333",
            linewidth=0.5 if not is_ours else 1.2,
            hatch="//" if is_ours else None,
            label=corr if i == 0 else None,
            zorder=3,
        )

# ---------------------------------------------------------------------------
# Reference lines
# ---------------------------------------------------------------------------
ax.axhline(y=1.0, color="#888888", linestyle="--", linewidth=0.8, zorder=1)
ax.axhline(
    y=0.5, color="#AAAAAA", linestyle=":", linewidth=0.8, zorder=1,
)
ax.text(
    x_positions[-1] + n_groups * bar_width + 0.05,
    0.5,
    "Random",
    va="center",
    ha="left",
    fontsize=9,
    color="#888888",
    style="italic",
)

# ---------------------------------------------------------------------------
# Axes formatting
# ---------------------------------------------------------------------------
ax.set_ylabel("AUROC", fontsize=12, labelpad=8)
ax.set_ylim(0, 1.08)
ax.set_yticks(np.arange(0, 1.1, 0.2))
ax.tick_params(axis="y", labelsize=10)

# Center x-ticks under each method group
tick_positions = x_positions + (n_groups - 1) * bar_width / 2
ax.set_xticks(tick_positions)
ax.set_xticklabels(methods, fontsize=11, ha="center")
ax.tick_params(axis="x", length=0, pad=6)

# Subtle grid on y-axis
ax.yaxis.grid(True, linestyle="-", linewidth=0.3, color="#DDDDDD", zorder=0)
ax.set_axisbelow(True)

# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles,
    labels,
    loc="upper left",
    ncol=5,
    fontsize=9.5,
    frameon=True,
    framealpha=0.9,
    edgecolor="#CCCCCC",
    handlelength=1.4,
    handletextpad=0.5,
    columnspacing=1.0,
    borderpad=0.4,
)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
fig.tight_layout()
out_path = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/latex/figures/fig_main_results.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out_path}")
