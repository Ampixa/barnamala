"""Publication figure: DHCD accuracy versus model size."""
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT = Path("paper/figures/efficiency_frontier.pdf")
N_TEST = 13_800
Z95 = 1.96
CEILING = (N_TEST - 11) / N_TEST * 100


def ci95_pct(accuracy_pct):
    p = accuracy_pct / 100.0
    return Z95 * math.sqrt(p * (1.0 - p) / N_TEST) * 100.0


def main():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9})

    eff = json.load(open("results/efficiency_metrics.json"))
    student = eff["models"][0]
    mallanet = eff["models"][1]

    points = [
        {
            "label": "Ours\n2027",
            "params_m": student["params_millions"],
            "accuracy": 99.735,
            "marker": "*",
            "size": 150,
            "color": "#D4A017",
            "edge": "#6B4E00",
            "zorder": 5,
            "ci": True,
            "offset": (8, 8),
        },
        {
            "label": "MallaNet\n2025",
            "params_m": mallanet["params_millions"],
            "accuracy": 99.710,
            "marker": "o",
            "size": 54,
            "color": "#4C78A8",
            "edge": "white",
            "zorder": 4,
            "ci": True,
            "offset": (-66, -18),
        },
        {
            "label": "Mishra et al.\n2021",
            "params_m": 39.0,
            "accuracy": 99.72,
            "marker": "o",
            "size": 48,
            "color": "#72B7B2",
            "edge": "white",
            "zorder": 3,
            "ci": False,
            "offset": (-76, 8),
        },
        {
            "label": "Yadav et al.\n2024",
            "params_m": 0.4,
            "accuracy": 99.21,
            "marker": "o",
            "size": 48,
            "color": "#F58518",
            "edge": "white",
            "zorder": 3,
            "ci": False,
            "offset": (8, -6),
        },
        {
            "label": "Acharya et al.\n2015",
            "params_m": 0.03,
            "accuracy": 98.47,
            "marker": "o",
            "size": 48,
            "color": "#B279A2",
            "edge": "white",
            "zorder": 3,
            "ci": False,
            "offset": (8, 4),
        },
    ]

    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    for pt in points:
        yerr = ci95_pct(pt["accuracy"]) if pt["ci"] else None
        if yerr is not None:
            ax.errorbar(
                pt["params_m"],
                pt["accuracy"],
                yerr=yerr,
                fmt="none",
                ecolor="#333333",
                elinewidth=0.8,
                capsize=2.5,
                capthick=0.8,
                zorder=pt["zorder"] - 1,
            )
        ax.scatter(
            pt["params_m"],
            pt["accuracy"],
            marker=pt["marker"],
            s=pt["size"],
            c=pt["color"],
            edgecolors=pt["edge"],
            linewidths=0.7,
            zorder=pt["zorder"],
        )
        ax.annotate(
            pt["label"],
            (pt["params_m"], pt["accuracy"]),
            xytext=pt["offset"],
            textcoords="offset points",
            fontsize=8,
            ha="left",
            va="center",
        )

    ax.axhline(CEILING, color="#B2182B", linestyle="--", linewidth=1.1)
    ax.annotate(
        "label-noise ceiling (11-error floor)",
        xy=(0.032, CEILING),
        xytext=(4, -10),
        textcoords="offset points",
        color="#B2182B",
        fontsize=8,
        ha="left",
        va="top",
    )

    ax.set_xscale("log")
    ax.set_xlim(0.02, 60)
    ax.set_ylim(98.25, 100.02)
    ax.set_title("Accuracy vs. Parameters on DHCD", fontsize=11, pad=8)
    ax.set_xlabel("Parameters (millions, log scale)")
    ax.set_ylabel("DHCD test accuracy (%)")
    ax.grid(True, which="major", linewidth=0.6, alpha=0.6)
    ax.grid(True, which="minor", linewidth=0.35, alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
