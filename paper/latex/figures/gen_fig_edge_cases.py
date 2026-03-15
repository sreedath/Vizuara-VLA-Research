"""
Generate publication-quality horizontal bar chart of cosine distances
for edge-case inputs, with 3-sigma detection threshold.

Output: fig_edge_cases.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────
detected_labels = [
    "All black",
    "All white",
    "Pure blue",
    "Gradient",
    "Pure green",
    "Pure red",
    "Checkerboard",
    r"32$\times$32 resolution",
    "Left half fog",
    "Bottom half fog",
    "Top half fog",
    "20 px border fog",
    "Non-square",
    r"50$\times$50 patch fog",
    r"512$\times$512",
]
detected_values = [
    8.71e-3, 6.23e-3, 5.46e-3, 5.32e-3, 4.95e-3,
    4.45e-3, 4.24e-3, 3.34e-3, 1.58e-3, 1.54e-3,
    1.17e-3, 1.05e-3, 1.02e-3, 6.07e-4, 2.86e-4,
]

not_detected_labels = [
    "1 pixel changed",
    "1000 pixels changed",
    r"Rotate 90$^\circ$",
    "Channel swap",
    "HF noise",
    "Horizontal flip",
]
not_detected_values = [
    5.90e-5, 5.55e-5, 5.37e-5, 4.55e-5, 2.98e-5, 2.96e-5,
]

THRESHOLD = 8.03e-5

# ── Colours ───────────────────────────────────────────────────────────
GREEN = "#2ca02c"
GREEN_LIGHT = "#e6f5e6"
RED = "#d62728"
RED_LIGHT = "#fde8e8"
THRESH_COL = "#333333"

# ── Build ordered lists (bottom → top in chart) ──────────────────────
labels, values, colors = [], [], []

# Not-detected group (bottom)
for lbl, val in zip(reversed(not_detected_labels), reversed(not_detected_values)):
    labels.append(lbl)
    values.append(val)
    colors.append(RED)

# Spacer row
labels.append("")
values.append(0)
colors.append("none")

# Detected group (top)
for lbl, val in zip(reversed(detected_labels), reversed(detected_values)):
    labels.append(lbl)
    values.append(val)
    colors.append(GREEN)

y_pos = np.arange(len(labels))
spacer_idx = len(not_detected_labels)  # index of spacer row

# ── Figure ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9.5,
    "figure.dpi": 300,
})

fig, ax = plt.subplots(figsize=(8, 8))

# Background shading for each group
ax.axhspan(-0.5, spacer_idx - 0.5, color=RED_LIGHT, alpha=0.5, zorder=0)
ax.axhspan(spacer_idx + 0.5, len(labels) - 0.5, color=GREEN_LIGHT, alpha=0.5, zorder=0)

# Bars
ax.barh(
    y_pos, values,
    color=colors,
    edgecolor=[c if c != "none" else "none" for c in colors],
    linewidth=0.5, height=0.7, alpha=0.88, zorder=3,
)

# Log scale
ax.set_xscale("log")
ax.set_xlim(left=1e-5, right=2.5e-2)

# Threshold line
ax.axvline(
    THRESHOLD, color=THRESH_COL, linestyle="--", linewidth=1.5, zorder=5,
)

# Threshold annotation (place beside the dashed line, in the middle region)
ax.annotate(
    rf"$3\sigma$ = {THRESHOLD:.2e}",
    xy=(THRESHOLD, spacer_idx),
    xytext=(THRESHOLD * 6, spacer_idx),
    fontsize=9, fontweight="bold", color=THRESH_COL,
    arrowprops=dict(arrowstyle="->", color=THRESH_COL, lw=1.0),
    ha="left", va="center",
)

# Group heading text (placed inside the shaded regions, top-right corner)
trans = mtransforms.blended_transform_factory(ax.transData, ax.transData)

ax.text(
    1.8e-2, len(labels) - 1.2,
    "DETECTED",
    fontsize=10, fontweight="bold", color=GREEN,
    va="top", ha="right", alpha=0.7, zorder=2,
)
ax.text(
    1.8e-2, spacer_idx - 1.2,
    "NOT DETECTED",
    fontsize=10, fontweight="bold", color=RED,
    va="top", ha="right", alpha=0.7, zorder=2,
)

# Separator line between groups
ax.axhline(spacer_idx, color="#999999", linestyle="-", linewidth=0.8, zorder=1)

# Labels and ticks
ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.set_xlabel("Cosine Distance (log scale)")
ax.set_title(
    "Edge-Case Input Detection via Cosine Distance",
    fontweight="bold", pad=14,
)

# Value annotations on bars
for i, (val, col) in enumerate(zip(values, colors)):
    if val == 0:
        continue
    ax.text(
        val * 1.18, i,
        f"{val:.2e}",
        va="center", ha="left", fontsize=7.5, color="#444444", zorder=6,
    )

# Colour y-tick labels to match group
for tick_label, col in zip(ax.get_yticklabels(), colors):
    if col == "none":
        continue
    tick_label.set_color(col)

# Clean up
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", alpha=0.2, which="both", zorder=0)
ax.set_ylim(-0.6, len(labels) - 0.2)

plt.tight_layout()
fig.savefig(
    "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/latex/figures/fig_edge_cases.png",
    dpi=300, bbox_inches="tight", facecolor="white",
)
plt.close(fig)
print("Saved fig_edge_cases.png")
