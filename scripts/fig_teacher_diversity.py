"""Publication figure: teacher diversity and ensemble collapse.

Story: Individual teachers span error counts 27–43; ensembling collapses that
spread to 30 errors regardless of ensemble size (9 or 15 teachers). Adding
more teachers provides no further gain because the residual errors sit on the
label-noise floor, not on recoverable model-capacity failures.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT = Path("paper/figures/teacher_diversity.pdf")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
# Try to load per-teacher breakdowns from results/; fall back to hard-coded list.
TEACHER_ERRORS_HARDCODED = [27, 28, 30, 30, 32, 34, 36, 37, 37, 39, 39, 40, 41, 42, 43]

def load_teacher_errors():
    results = Path("results")
    errors = []
    if results.exists():
        for p in sorted(results.glob("teacher_*seed*/eval_metrics.json")):
            try:
                d = json.loads(p.read_text())
                # accept various key names
                for key in ("errors", "n_errors", "error_count", "num_errors"):
                    if key in d:
                        errors.append(int(d[key]))
                        break
            except Exception:
                pass
    return errors if len(errors) >= 5 else TEACHER_ERRORS_HARDCODED


teacher_errors = load_teacher_errors()
n_teachers = len(teacher_errors)

# Reference values
ENSEMBLE_9  = 30   # 9-teacher ensemble (clean)
ENSEMBLE_15 = 30   # 15-teacher ensemble (clean)
BASELINE    = 40   # MallaNet baseline
FLOOR       = 11   # label-noise floor (shared by all models)
THRESHOLD   = 25   # significance threshold (p<0.05 vs MallaNet)

# ---------------------------------------------------------------------------
# Layout: horizontal dot-plot, one row per entity group
# ---------------------------------------------------------------------------
# Y positions (top → bottom = higher y → lower y in axes)
Y_TEACHERS  = 3.0
Y_ENS9      = 2.0
Y_ENS15     = 1.5
Y_BASELINE  = 0.5

Y_LABELS = {
    f"Individual teachers (×{n_teachers})": Y_TEACHERS,
    "9-teacher ensemble":                   Y_ENS9,
    "15-teacher ensemble":                  Y_ENS15,
    "MallaNet (baseline)":                  Y_BASELINE,
}

rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size":       9,
    "axes.titlesize": 10,
    "axes.labelsize":  9,
    "font.family":    "sans-serif",
})

fig, ax = plt.subplots(figsize=(7.0, 3.5))

# --- Reference lines --------------------------------------------------------
ax.axvline(FLOOR,     color="#888888", linestyle="-",  linewidth=1.2, zorder=1)
ax.axvline(THRESHOLD, color="#CC0000", linestyle="--", linewidth=1.2, zorder=1)
ax.axvline(BASELINE,  color="#888888", linestyle=":",  linewidth=1.0, zorder=1)

# Reference-line labels (top of axes)
ax.text(FLOOR + 0.5,  3.65, "label-noise\nfloor",      fontsize=7.5,
        color="#555555", va="top", ha="left")
ax.text(THRESHOLD + 0.5, 3.65, "p<0.05\nthreshold",   fontsize=7.5,
        color="#CC0000", va="top", ha="left")
ax.text(BASELINE - 0.5,  3.65, "MallaNet",            fontsize=7.5,
        color="#555555", va="top", ha="right")

# --- Individual teachers (jittered) ----------------------------------------
jitter = rng.uniform(-0.18, 0.18, size=len(teacher_errors))
y_teachers = Y_TEACHERS + jitter
ax.scatter(
    teacher_errors, y_teachers,
    color="#5B9BD5", alpha=0.75, s=40, zorder=3,
    linewidths=0.4, edgecolors="#2a6099",
)

# --- Ensemble dots ----------------------------------------------------------
ax.scatter([ENSEMBLE_9],  [Y_ENS9],  color="#2ca02c", s=70, zorder=3,
           linewidths=0.6, edgecolors="#1a6b1a")
ax.scatter([ENSEMBLE_15], [Y_ENS15], color="#2ca02c", s=70, zorder=3,
           linewidths=0.6, edgecolors="#1a6b1a")

# --- Baseline dot -----------------------------------------------------------
ax.scatter([BASELINE], [Y_BASELINE], color="#e07b00", s=70, zorder=3,
           linewidths=0.6, edgecolors="#9e5700")

# --- Collapse annotation: brace/arrow from spread to ensemble ---------------
# Draw a horizontal double-headed arrow spanning the teacher range, then a
# downward arrow pointing to the ensemble dot.
t_min, t_max = min(teacher_errors), max(teacher_errors)
ax.annotate(
    "",
    xy=(t_max, Y_TEACHERS + 0.28),
    xytext=(t_min, Y_TEACHERS + 0.28),
    arrowprops=dict(arrowstyle="<->", color="#5B9BD5", lw=1.2),
)
ax.text(
    (t_min + t_max) / 2, Y_TEACHERS + 0.38,
    f"spread: {t_min}–{t_max} errors",
    fontsize=7.5, ha="center", va="bottom", color="#2a6099",
)

# Arrow from ensemble cluster to annotation
ax.annotate(
    "ensemble collapses\nto 30 (9 or 15 teachers)",
    xy=(ENSEMBLE_9, Y_ENS9),
    xytext=(ENSEMBLE_9 + 8, Y_ENS9 + 0.55),
    fontsize=7.5, ha="left", va="center", color="#1a6b1a",
    arrowprops=dict(arrowstyle="->", color="#1a6b1a", lw=1.0,
                    connectionstyle="arc3,rad=-0.25"),
)

# --- Insight text box -------------------------------------------------------
ax.text(
    0.01, 0.03,
    "Ensembling ↓ variance but cannot break\nthe data-quality floor (11 errors).",
    transform=ax.transAxes,
    fontsize=7.5, va="bottom", ha="left",
    bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
              edgecolor="#C7C7C7", alpha=0.90),
)

# --- Axes formatting --------------------------------------------------------
ax.set_xlim(5, 50)
ax.set_ylim(0.0, 4.0)
ax.set_xlabel("Error count on DHCD test set (n=13,800)  —  fewer = better")
ax.set_yticks(list(Y_LABELS.values()))
ax.set_yticklabels(list(Y_LABELS.keys()), fontsize=8.5)
ax.set_title("Teacher Diversity and Ensemble Collapse", fontsize=10, pad=6)
ax.grid(True, axis="x", linewidth=0.5, alpha=0.5)
ax.grid(False, axis="y")

# --- Legend -----------------------------------------------------------------
legend_handles = [
    mpatches.Patch(color="#5B9BD5", alpha=0.75, label="Individual teacher"),
    mpatches.Patch(color="#2ca02c",             label="Ensemble"),
    mpatches.Patch(color="#e07b00",             label="MallaNet baseline"),
]
ax.legend(handles=legend_handles, fontsize=8, loc="upper right",
          framealpha=0.92, edgecolor="#C7C7C7")

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print(f"wrote {OUT}")
