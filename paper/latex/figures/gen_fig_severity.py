"""
Generate publication-quality figure: Cosine Distance vs Corruption Severity.
NeurIPS 2026 -- CalibDrive project.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Data ──────────────────────────────────────────────────────────────────────
severity = np.linspace(0.0, 1.0, 21)

fog = np.array([
    1.2e-07, 1.597e-05, 1.961e-05, 2.193e-05, 2.605e-05, 3.242e-05,
    3.791e-05, 4.774e-05, 5.615e-05, 5.358e-05, 6.896e-05, 8.744e-05,
    0.00010532, 0.00012589, 0.00016069, 0.0001685, 0.00020617, 0.00024736,
    0.0003078, 0.00036699, 0.00042725,
])

night = np.array([
    1.2e-07, 2.599e-05, 7.48e-05, 0.00015169, 0.00026125, 0.00041586,
    0.00058192, 0.00081503, 0.00104231, 0.00130606, 0.00158399, 0.00193137,
    0.00231928, 0.00279582, 0.00317627, 0.00357294, 0.00405121, 0.00442547,
    0.00503224, 0.00572252, 0.00644326,
])

noise = np.array([
    1.2e-07, 8.64e-06, 1.276e-05, 1.293e-05, 1.407e-05, 1.407e-05,
    1.872e-05, 2.05e-05, 2.378e-05, 2.438e-05, 2.939e-05, 3.237e-05,
    3.546e-05, 4.256e-05, 5.561e-05, 5.686e-05, 4.029e-05, 4.512e-05,
    5.341e-05, 5.078e-05, 5.168e-05,
])

blur = np.array([
    1.2e-07, 1.436e-05, 0.00013071, 0.00047398, 0.00077778, 0.00095052,
    0.00123042, 0.00156802, 0.0018881, 0.00228959, 0.00259537, 0.00289559,
    0.00315768, 0.00342864, 0.00362015, 0.00380701, 0.00401127, 0.00421846,
    0.00431752, 0.00442004, 0.00454754,
])

threshold = 8.81e-05

# ── Style ─────────────────────────────────────────────────────────────────────
# Colorblind-friendly palette (Wong 2011 / IBM Design Library)
colors = {
    "fog":   "#0072B2",   # blue
    "night": "#D55E00",   # vermillion
    "noise": "#009E73",   # bluish green
    "blur":  "#CC79A7",   # reddish purple
}

markers = {
    "fog":   "o",
    "night": "s",
    "noise": "^",
    "blur":  "D",
}

r2 = {"fog": 0.83, "night": 0.93, "noise": 0.91, "blur": 0.99}

# ── Figure ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.8,
    "lines.markersize": 5,
    "figure.dpi": 300,
})

fig, ax = plt.subplots(figsize=(8, 5))

# Plot each corruption type
for name, data in [("fog", fog), ("night", night), ("noise", noise), ("blur", blur)]:
    label = f"{name.capitalize()} ($R^2$={r2[name]:.2f})"
    ax.plot(
        severity, data,
        color=colors[name],
        marker=markers[name],
        markeredgecolor="white",
        markeredgewidth=0.5,
        markersize=5,
        label=label,
        zorder=3,
    )

# 3-sigma threshold line
ax.axhline(
    y=threshold,
    color="#555555",
    linestyle="--",
    linewidth=1.2,
    zorder=2,
    label=r"3$\sigma$ threshold",
)

# Annotate the threshold value on the right side
ax.text(
    1.01, threshold,
    f"  {threshold:.2e}",
    va="center", ha="left",
    fontsize=8.5,
    color="#555555",
    transform=ax.get_yaxis_transform(),
)

# Shaded noise-floor region below threshold
ax.axhspan(
    ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 1e-08,
    threshold,
    color="#DDDDDD",
    alpha=0.45,
    zorder=0,
    label="Noise floor",
)

# Log scale
ax.set_yscale("log")

# Axes labels
ax.set_xlabel("Severity")
ax.set_ylabel("Cosine Distance (log scale)")

# Limits
ax.set_xlim(-0.02, 1.02)
# Set y-limits explicitly to give room for annotation and noise floor
ax.set_ylim(5e-08, 1.5e-02)

# Re-draw the shaded region now that y-limits are set
# (remove the previous axhspan and redo)
for patch in ax.patches:
    patch.remove()
ax.axhspan(
    5e-08, threshold,
    color="#DDDDDD",
    alpha=0.45,
    zorder=0,
)

# Add "Noise floor" text inside the shaded region
ax.text(
    0.50, 2.5e-06,
    "Noise floor",
    fontsize=9,
    color="#888888",
    ha="center",
    va="center",
    fontstyle="italic",
    zorder=1,
)

# Grid
ax.grid(True, which="major", linestyle="-", alpha=0.3, zorder=0)
ax.grid(True, which="minor", linestyle=":", alpha=0.15, zorder=0)

# Tick formatting
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))

# Legend -- outside the data area
legend = ax.legend(
    loc="upper left",
    frameon=True,
    framealpha=0.9,
    edgecolor="#CCCCCC",
    fancybox=False,
    borderpad=0.6,
    handlelength=2.2,
)
legend.get_frame().set_linewidth(0.6)

# Title (optional for NeurIPS -- usually the caption carries context)
# Uncomment if desired:
# ax.set_title("Cosine Distance vs Corruption Severity", pad=12)

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/latex/figures/fig_severity.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.12)
plt.close(fig)
print(f"Saved to {out_path}")
