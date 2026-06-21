"""Publication figure: significance frontier for DHCD benchmark saturation."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT = Path("paper/figures/significance_frontier.pdf")

# ---------------------------------------------------------------------------
# Data (hard-coded — no JSON needed)
# ---------------------------------------------------------------------------

# (label, mean_errors, std_errors or None, group, excluded)
CONFIGS = [
    # Students
    ("Student distilled TTA",   36.6, 2.1,  "student",   False),
    ("Student distilled clean", 38.2, 2.2,  "student",   False),
    ("Student supervised",      39.8, 2.8,  "student",   False),
    # Ensembles
    ("9-teacher ensemble",      30.0, None, "ensemble",  False),
    ("15-teacher ensemble",     30.0, None, "ensemble",  False),
    ("9-teacher + flip-TTA",    28.0, None, "ensemble",  True),   # excluded
    # References
    ("MallaNet (baseline)",     40.0, None, "reference", False),
    ("Intrinsic floor",         11.0, None, "reference", False),
]

# Significance thresholds (from exact McNemar vs MallaNet at 40 errors)
THRESH_05  = 25   # p < 0.05  (99.82% accuracy)
THRESH_01  = 20   # p < 0.01  (99.855% accuracy)
FLOOR_LINE = 11   # label-noise floor
MALLANET   = 40   # baseline

# Group visual properties
GROUP_META = {
    "student":   {"color": "#0072B2", "band_alpha": 0.07, "band_color": "#0072B2"},
    "ensemble":  {"color": "#009E73", "band_alpha": 0.07, "band_color": "#009E73"},
    "reference": {"color": "#555555", "band_alpha": 0.07, "band_color": "#555555"},
}

# Y positions: top to bottom in display order (students first, then ensembles, then refs)
GROUP_ORDER = ["student", "ensemble", "reference"]
GROUP_LABELS = {"student": "Students", "ensemble": "Ensembles", "reference": "References"}


def main():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})

    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    # Assign y positions by group order, reversed so top group is at top of plot
    y_pos = {}
    y = len(CONFIGS) - 1
    for group in GROUP_ORDER:
        for label, mean, std, grp, excl in CONFIGS:
            if grp == group:
                y_pos[label] = y
                y -= 1

    # Draw background bands per group
    # Collect y extent of each group
    group_y = {g: [] for g in GROUP_ORDER}
    for label, mean, std, grp, excl in CONFIGS:
        group_y[grp].append(y_pos[label])

    for grp in GROUP_ORDER:
        ys = group_y[grp]
        y_min = min(ys) - 0.45
        y_max = max(ys) + 0.45
        meta = GROUP_META[grp]
        ax.axhspan(y_min, y_max,
                   color=meta["band_color"],
                   alpha=meta["band_alpha"],
                   zorder=0)
        # Group label on the right outside the band
        mid_y = (y_min + y_max) / 2.0
        ax.text(51.5, mid_y,
                GROUP_LABELS[grp],
                fontsize=8, color=meta["color"],
                va="center", ha="left",
                fontweight="bold",
                clip_on=False)

    # Draw significance threshold lines (behind data)
    ax.axvline(THRESH_05, color="#CC0000", linestyle="--", linewidth=1.2, zorder=1)
    ax.axvline(THRESH_01, color="#CC0000", linestyle=":",  linewidth=1.2, zorder=1)
    ax.axvline(MALLANET,  color="#888888", linestyle="-",  linewidth=1.0, zorder=1)
    ax.axvline(FLOOR_LINE, color="#888888", linestyle="-", linewidth=1.0, zorder=1)

    # Threshold label positions (top of plot)
    y_top = len(CONFIGS) - 0.2
    ax.text(THRESH_05 - 0.4, y_top, "p<0.05\nthreshold",
            color="#CC0000", fontsize=7.5, ha="right", va="top", linespacing=1.3)
    ax.text(THRESH_01 - 0.4, y_top, "p<0.01\nthreshold",
            color="#CC0000", fontsize=7.5, ha="right", va="top", linespacing=1.3)
    ax.text(MALLANET + 0.4, y_top, "MallaNet\n(40)",
            color="#888888", fontsize=7.5, ha="left", va="top", linespacing=1.3)
    ax.text(FLOOR_LINE + 0.4, y_top, "label-noise\nfloor",
            color="#888888", fontsize=7.5, ha="left", va="top", linespacing=1.3)

    # Plot data points
    for label, mean, std, grp, excl in CONFIGS:
        yp = y_pos[label]
        color = GROUP_META[grp]["color"]

        if excl:
            # Hollow marker for excluded point
            ax.scatter(mean, yp,
                       marker="o", s=55,
                       facecolors="none",
                       edgecolors=color,
                       linewidths=1.4,
                       zorder=3)
            # Strikethrough annotation
            ax.annotate(
                "excluded\n(semantically invalid)",
                xy=(mean, yp),
                xytext=(mean - 8, yp - 0.55),
                fontsize=7.5,
                color=color,
                ha="right",
                va="top",
                style="italic",
                arrowprops=dict(arrowstyle="-", color=color, lw=0.6),
            )
        else:
            ax.scatter(mean, yp,
                       marker="o", s=55,
                       color=color,
                       edgecolors="white",
                       linewidths=0.7,
                       zorder=3)
            if std is not None:
                ax.errorbar(mean, yp,
                            xerr=std,
                            fmt="none",
                            ecolor=color,
                            elinewidth=1.1,
                            capsize=3.5,
                            capthick=1.1,
                            zorder=2)

        # Config label on left
        ax.text(-0.6, yp, label,
                ha="right", va="center",
                fontsize=8.5,
                color="#333333" if not excl else color)

    # Axes formatting
    ax.set_xlim(0, 50)
    ax.set_ylim(-0.7, len(CONFIGS) - 0.3)
    ax.set_xlabel(
        "Errors on DHCD test set (n=13,800) — fewer is better →",
        fontsize=9
    )
    ax.set_title(
        "Significance Frontier: No Configuration Crosses p<0.05",
        fontsize=10, pad=8
    )

    # Hide y-axis ticks (labels on the left instead)
    ax.set_yticks([])
    ax.yaxis.set_visible(False)

    # Keep only bottom and top spines for clean look
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # X-axis ticks at clean multiples of 5
    ax.set_xticks(range(0, 51, 5))

    # Legend for marker styles
    solid_patch  = mpatches.Patch(color="#0072B2", label="Mean ± 1 std (student configs)")
    hollow_patch = plt.Line2D(
        [], [], marker="o", color="#009E73",
        markerfacecolor="none", markeredgecolor="#009E73",
        markeredgewidth=1.4, markersize=6,
        linestyle="none", label="Excluded point"
    )
    thresh_line  = plt.Line2D([], [], color="#CC0000", linestyle="--",
                              linewidth=1.2, label="Significance thresholds")
    ax.legend(
        handles=[solid_patch, hollow_patch, thresh_line],
        loc="lower right",
        fontsize=8,
        framealpha=0.92,
        edgecolor="#CCCCCC",
    )

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
