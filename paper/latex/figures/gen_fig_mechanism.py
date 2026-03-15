"""
Generate publication-quality 2-panel figure for NeurIPS paper:
  (a) Attention Pattern Stability — bar chart
  (b) Separation Ratio by Layer   — line plot
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Global style — NeurIPS-friendly, colorblind-safe
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth": 1.8,
    "axes.grid": False,
})

# Colorblind-friendly palette (Wong, 2011 — Nature Methods)
COLOR_BLUE   = "#0072B2"
COLOR_ORANGE = "#E69F00"
COLOR_GREEN  = "#009E73"
COLOR_RED    = "#D55E00"
COLOR_PURPLE = "#CC79A7"
COLOR_GRAY   = "#999999"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
corruptions = ["Fog", "Night", "Noise", "Blur"]

layer3_vals  = [0.9999, 0.9998, 0.9999, 0.9999]
layer31_vals = [0.94,   0.82,   0.95,   0.82]

sep_layers = np.arange(0, 33)
sep_values = [
    0.0, 256.92, 88.56, 85.12, 76.83, 73.89, 70.40, 81.01,
    52.18, 42.00, 24.07, 29.71, 27.07, 20.84, 18.10, 14.48,
    14.12, 13.45, 13.49, 13.08, 12.73, 11.32, 10.34, 8.87,
    8.16, 7.61, 7.10, 6.51, 6.30, 6.00, 5.84, 5.77, 5.75,
]

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 4.5))

# ── Panel (a): Attention Stability bar chart ──────────────────────────────
x = np.arange(len(corruptions))
bar_width = 0.32

bars_l3  = ax_a.bar(x - bar_width / 2, layer3_vals,  bar_width,
                     label="Layer 3",  color=COLOR_BLUE, edgecolor="white",
                     linewidth=0.5, zorder=3)
bars_l31 = ax_a.bar(x + bar_width / 2, layer31_vals, bar_width,
                     label="Layer 31", color=COLOR_ORANGE, edgecolor="white",
                     linewidth=0.5, zorder=3)

ax_a.set_xticks(x)
ax_a.set_xticklabels(corruptions)
ax_a.set_ylabel("Cosine Similarity")
ax_a.set_ylim(0.75, 1.01)
ax_a.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
ax_a.yaxis.set_minor_locator(ticker.MultipleLocator(0.025))
ax_a.set_title("(a) Attention Pattern Stability", fontweight="bold", pad=10)
ax_a.legend(loc="lower left", frameon=True, fancybox=False, edgecolor="#cccccc")

# Light horizontal grid behind bars
ax_a.yaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.6, zorder=0)
ax_a.set_axisbelow(True)

# Annotation: "Detection layer" pointing to Layer-3 bars
# Place text just above the Fog L3 bar with a short downward arrow
ax_a.annotate(
    "Detection layer",
    xy=(x[0], 1.003),
    xytext=(x[0], 1.008),
    fontsize=9,
    fontstyle="italic",
    color=COLOR_BLUE,
    ha="center",
    va="bottom",
    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=COLOR_BLUE,
              lw=0.8, alpha=0.9),
    clip_on=False,
)
# Small downward arrow below the annotation box
ax_a.annotate(
    "",
    xy=(x[0] - bar_width / 2, layer3_vals[0] + 0.002),
    xytext=(x[0], 1.007),
    arrowprops=dict(
        arrowstyle="->,head_width=0.2,head_length=0.12",
        color=COLOR_BLUE,
        lw=1.2,
    ),
    clip_on=False,
)

# ── Panel (b): Separation Ratio line plot ─────────────────────────────────
ax_b.plot(sep_layers, sep_values, color=COLOR_RED, marker="o", markersize=3.5,
          markerfacecolor="white", markeredgewidth=1.2, zorder=3)
ax_b.fill_between(sep_layers, sep_values, alpha=0.08, color=COLOR_RED, zorder=1)

ax_b.set_xlabel("Layer")
ax_b.set_ylabel("Separation Ratio")
ax_b.set_title("(b) Separation Ratio by Layer", fontweight="bold", pad=10)
ax_b.set_xlim(-0.5, 32.5)
ax_b.xaxis.set_major_locator(ticker.MultipleLocator(4))
ax_b.xaxis.set_minor_locator(ticker.MultipleLocator(1))

# Light horizontal grid
ax_b.yaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.6, zorder=0)
ax_b.set_axisbelow(True)

# Vertical dashed line at layer 3
ax_b.axvline(x=3, color=COLOR_BLUE, linestyle="--", linewidth=1.2, alpha=0.8,
             zorder=2)
# Label for the vertical dashed line — place text at top of plot
ax_b.text(
    3.6, 245, "Detection\nlayer",
    fontsize=9,
    fontstyle="italic",
    color=COLOR_BLUE,
    ha="left",
    va="top",
    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=COLOR_BLUE,
              lw=0.8, alpha=0.9),
)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
fig.tight_layout(w_pad=3.0)
out_path = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/latex/figures/fig_mechanism.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out_path}")
