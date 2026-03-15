"""
Generate publication-quality calibration figure for NeurIPS paper.
Two-panel figure: (a) Calibration Efficiency, (b) Cross-Scene Transfer.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------------------------------------------------------------------------
# Global style: publication-quality, colorblind-friendly
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.8,
    "lines.markersize": 7,
    "text.usetex": False,
})

# Colorblind-friendly palette (Wong 2011 / Tol bright)
COLORS = {
    "Fog":   "#0077BB",  # blue
    "Night": "#EE7733",  # orange
    "Noise": "#009988",  # teal
    "Blur":  "#CC3311",  # red
}

MARKERS = {
    "Fog":   "o",
    "Night": "s",
    "Noise": "D",
    "Blur":  "^",
}

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
cal_sizes = np.array([1, 2, 5, 10, 20])

auroc_data = {
    "Fog":   np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
    "Night": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
    "Noise": np.array([0.96, 0.98, 0.99, 0.99, 1.0]),
    "Blur":  np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
}

noise_std = np.array([0.06, 0.02, 0.02, 0.02, 0.0])

# Cross-scene transfer heatmap (3x3, all 1.0)
transfer_matrix = np.ones((3, 3))
scene_labels = ["A", "B", "C"]

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5),
                                gridspec_kw={"width_ratios": [1.4, 1]})

# ---- Panel (a): AUROC vs Calibration Set Size ----
for name in ["Fog", "Night", "Noise", "Blur"]:
    y = auroc_data[name]
    color = COLORS[name]
    marker = MARKERS[name]

    if name == "Noise":
        ax1.errorbar(
            cal_sizes, y, yerr=noise_std,
            color=color, marker=marker, label=name,
            capsize=4, capthick=1.2, elinewidth=1.0,
            markeredgecolor="white", markeredgewidth=0.6,
            zorder=5,
        )
        # Light shaded error band
        ax1.fill_between(
            cal_sizes,
            y - noise_std,
            np.minimum(y + noise_std, 1.01),
            color=color, alpha=0.12, zorder=2,
        )
    else:
        ax1.plot(
            cal_sizes, y,
            color=color, marker=marker, label=name,
            markeredgecolor="white", markeredgewidth=0.6,
            zorder=5,
        )

ax1.set_xlabel("Number of Calibration Images")
ax1.set_ylabel("AUROC")
ax1.set_title("(a) AUROC vs Calibration Set Size", fontweight="bold", pad=10)
ax1.set_xticks(cal_sizes)
ax1.set_xticklabels([str(x) for x in cal_sizes])
ax1.set_ylim(0.9, 1.01)
ax1.set_xlim(-0.5, 21)

# Subtle grid
ax1.grid(True, linestyle="--", alpha=0.35, zorder=0)
ax1.set_axisbelow(True)

# Legend
ax1.legend(
    loc="lower right",
    frameon=True,
    framealpha=0.9,
    edgecolor="#cccccc",
    fancybox=False,
    borderpad=0.6,
    handlelength=2.2,
)

# Minor ticks
ax1.minorticks_on()
ax1.tick_params(which="minor", length=2, width=0.5)

# ---- Panel (b): Cross-Scene Transfer Heatmap ----
# Use a sequential colormap clipped so 1.0 appears as a strong, distinct color
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    "custom_green",
    ["#f7fcf5", "#c7e9c0", "#74c476", "#238b45", "#005a32"],
    N=256,
)

im = ax2.imshow(
    transfer_matrix, cmap=cmap, vmin=0.8, vmax=1.0,
    aspect="equal",
)

# Annotate cells
for i in range(3):
    for j in range(3):
        val = transfer_matrix[i, j]
        ax2.text(
            j, i, f"{val:.2f}",
            ha="center", va="center",
            fontsize=14, fontweight="bold",
            color="white",
        )

ax2.set_xticks(range(3))
ax2.set_xticklabels(scene_labels)
ax2.set_yticks(range(3))
ax2.set_yticklabels(scene_labels)
ax2.set_xlabel("Test Set")
ax2.set_ylabel("Calibration Set")
ax2.set_title("(b) Cross-Scene Transfer AUROC", fontweight="bold", pad=10)

# Colorbar
cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.06, shrink=0.85)
cbar.set_label("AUROC", fontsize=10)
cbar.ax.tick_params(labelsize=9)

# Remove minor ticks on heatmap
ax2.minorticks_off()

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
fig.tight_layout(pad=2.0)

out_path = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/latex/figures/fig_calibration.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out_path}")
