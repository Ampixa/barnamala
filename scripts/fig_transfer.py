"""Publication figure: CMATERdb transfer progression (knowledge distillation vs supervised).

Generates paper/figures/transfer_progression.pdf — two-panel figure showing:
  LEFT:  Horizontal bar chart comparing supervised vs distilled zero-shot accuracy
         plus linear probe and fine-tuned adaptation results.
  RIGHT: Step/progression chart showing the adaptation trajectory with shaded
         KD advantage gap.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results" / "digit_transfer.json"
OUT = ROOT / "paper" / "figures" / "transfer_progression.pdf"

# ---------------------------------------------------------------------------
# Colors  (colorblind-safe)
# ---------------------------------------------------------------------------
C_SUPERVISED = "#999999"   # gray
C_DISTILLED  = "#0072B2"   # blue
C_PROBE      = "#009E73"   # teal
C_FINETUNE   = "#E69F00"   # gold/amber

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
d = json.load(open(DATA))

sup_mean  = d["zero_shot"]["supervised_mean"] * 100
sup_seeds = [v * 100 for v in d["zero_shot"]["supervised_seeds"]]
# supervised_std not present in file — compute from seeds
sup_std   = float(np.std(sup_seeds, ddof=1))

dist_mean  = d["zero_shot"]["distilled_mean"] * 100
dist_seeds = [v * 100 for v in d["zero_shot"]["distilled_seeds"]]
dist_std   = d["zero_shot"]["distilled_std"] * 100

probe_acc    = d["adaptation_distilled_seed0"]["linear_probe"]["acc"] * 100
finetune_acc = d["adaptation_distilled_seed0"]["full_finetune"]["acc"] * 100

kd_gap = dist_mean - sup_mean   # should be ~14 pp

XMIN, XMAX = 55, 100

# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(8.0, 3.2))
# left panel ~40%, right panel ~60%
gs = fig.add_gridspec(1, 2, width_ratios=[0.40, 0.60], wspace=0.38)
ax_bar  = fig.add_subplot(gs[0])
ax_prog = fig.add_subplot(gs[1])

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
})

# ===========================================================================
# LEFT PANEL — horizontal bar chart
# ===========================================================================
bar_labels = [
    "Supervised\nzero-shot",
    "Distilled\nzero-shot",
    "Linear probe\n(distilled)",
    "Fine-tuned\n(distilled)",
]
bar_values = [sup_mean, dist_mean, probe_acc, finetune_acc]
bar_colors = [C_SUPERVISED, C_DISTILLED, C_PROBE, C_FINETUNE]
bar_stds   = [sup_std, dist_std, 0.0, 0.0]

y_pos = np.arange(len(bar_labels))[::-1]   # top-to-bottom order
bar_height = 0.52

for i, (y, val, color, std) in enumerate(zip(y_pos, bar_values, bar_colors, bar_stds)):
    # Draw bar from XMIN to val
    ax_bar.barh(y, val - XMIN, left=XMIN, height=bar_height,
                color=color, alpha=0.85, zorder=3)
    # Error bar
    if std > 0:
        ax_bar.errorbar(val, y, xerr=std, fmt="none",
                        color="black", capsize=3, linewidth=1.0, zorder=5)
    # Value label
    ax_bar.text(val + 0.6, y, f"{val:.1f}%",
                va="center", ha="left", fontsize=7.5, zorder=6)

# Overlay individual seed dots for zero-shot bars
for seed_val in sup_seeds:
    ax_bar.plot(seed_val, y_pos[0], "o", color="white", markeredgecolor=C_SUPERVISED,
                markersize=4, zorder=7, markeredgewidth=0.8)
for seed_val in dist_seeds:
    ax_bar.plot(seed_val, y_pos[1], "o", color="white", markeredgecolor=C_DISTILLED,
                markersize=4, zorder=7, markeredgewidth=0.8)

# Vertical dashed reference line at supervised zero-shot mean
ax_bar.axvline(sup_mean, color=C_SUPERVISED, linestyle="--", linewidth=1.0,
               zorder=2, alpha=0.7)

ax_bar.set_xlim(XMIN, XMAX + 6)
ax_bar.set_ylim(-0.6, len(bar_labels) - 0.4)
ax_bar.set_yticks(y_pos)
ax_bar.set_yticklabels(bar_labels, fontsize=7.5)
ax_bar.set_xlabel("Accuracy (%)", fontsize=8)
ax_bar.set_title("CMATERdb Transfer\n(DHCD → digits)", fontsize=9, pad=6)
ax_bar.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))
ax_bar.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.7, zorder=1)
ax_bar.set_axisbelow(True)
# Remove y-axis spine clutter
ax_bar.spines["top"].set_visible(False)
ax_bar.spines["right"].set_visible(False)

# ===========================================================================
# RIGHT PANEL — adaptation trajectory
# ===========================================================================
stage_labels = [
    "Zero-shot\n(supervised)",
    "Zero-shot\n(distilled)",
    "Linear\nprobe",
    "Fine-\ntuned",
]
x_stages = [0, 1, 2, 3]

# Distilled path: all four values
dist_path = [sup_mean, dist_mean, probe_acc, finetune_acc]
# We'll draw two series:
#   supervised: only stage 0 (dot)
#   distilled: stages 0–3, but starting from stage 1 (its own zero-shot)

# Distilled trajectory line: stages 1,2,3
dist_x = [1, 2, 3]
dist_y = [dist_mean, probe_acc, finetune_acc]

ax_prog.plot(dist_x, dist_y, color=C_DISTILLED, linewidth=1.8,
             marker="o", markersize=6, zorder=5, label="Distilled path")

# Supervised baseline dot at stage 0
ax_prog.plot(0, sup_mean, "o", color=C_SUPERVISED, markersize=7, zorder=6,
             label="Supervised baseline")

# Shade the gap between distilled and supervised at zero-shot (stages 0 and 1)
# Vertical shaded band between sup_mean and dist_mean at x=1, with a connector
ax_prog.fill_betweenx([sup_mean, dist_mean], 0.85, 1.15,
                      color=C_DISTILLED, alpha=0.18, zorder=2)
ax_prog.vlines(1, sup_mean, dist_mean, colors=C_DISTILLED,
               linewidth=1.0, linestyle=":", zorder=3)

# Annotate the gap
mid_gap = (sup_mean + dist_mean) / 2
ax_prog.annotate(
    f"+{kd_gap:.0f} pp\n(KD advantage)",
    xy=(1, mid_gap),
    xytext=(1.55, mid_gap - 5),
    fontsize=7,
    color=C_DISTILLED,
    ha="left",
    arrowprops=dict(arrowstyle="-", color=C_DISTILLED, lw=0.8),
    zorder=8,
)

# Dashed horizontal line at supervised baseline across full plot
ax_prog.axhline(sup_mean, color=C_SUPERVISED, linestyle="--",
                linewidth=1.0, alpha=0.6, zorder=1)
ax_prog.text(3.08, sup_mean + 0.5, f"{sup_mean:.1f}%", color=C_SUPERVISED,
             fontsize=6.5, va="bottom", zorder=8)

# Value annotations on distilled path
for xi, yi in zip(dist_x, dist_y):
    offset = 1.5 if xi < 3 else -1.5
    va = "bottom" if xi < 3 else "top"
    ax_prog.text(xi, yi + offset, f"{yi:.1f}%",
                 ha="center", va=va, fontsize=7, color=C_DISTILLED, zorder=8)

ax_prog.set_xlim(-0.4, 3.6)
ax_prog.set_ylim(XMIN, XMAX + 3)
ax_prog.set_xticks(x_stages)
ax_prog.set_xticklabels(stage_labels, fontsize=7.5)
ax_prog.set_ylabel("Accuracy (%)", fontsize=8)
ax_prog.set_title("Adaptation Trajectory", fontsize=9, pad=6)
ax_prog.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7, zorder=0)
ax_prog.set_axisbelow(True)
ax_prog.spines["top"].set_visible(False)
ax_prog.spines["right"].set_visible(False)

# Legend
sup_patch = mpatches.Patch(color=C_SUPERVISED, label="Supervised baseline")
dist_patch = mpatches.Patch(color=C_DISTILLED, label="Distilled (KD)")
probe_patch = mpatches.Patch(color=C_PROBE, label="Linear probe")
ft_patch = mpatches.Patch(color=C_FINETUNE, label="Fine-tuned")
ax_prog.legend(handles=[sup_patch, dist_patch, probe_patch, ft_patch],
               fontsize=6.5, loc="upper left", framealpha=0.85)

# ===========================================================================
# Save
# ===========================================================================
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"Saved: {OUT}")
