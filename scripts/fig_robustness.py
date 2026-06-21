"""Publication figure: corruption robustness for Barnamala and MallaNet."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT = Path("paper/figures/robustness_curves.pdf")
BARNAMALA = "#0072B2"
MALLANET = "#D55E00"


def main():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})

    rob = json.load(open("results/robustness.json"))
    kinds = [("noise", "Noise"), ("blur", "Blur"), ("contrast", "Contrast")]
    severities = list(range(6))

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0), sharey=True)
    for ax, (kind, title) in zip(axes, kinds):
        student = rob["student"]["per_corruption"][kind]
        mallanet = rob["mallanet"]["per_corruption"][kind]

        ax.plot(
            severities,
            [v * 100 for v in student["by_severity"]],
            color=BARNAMALA,
            marker="o",
            markersize=4,
            linewidth=1.8,
            label="Ours (1.11M)",
        )
        ax.plot(
            severities,
            [v * 100 for v in mallanet["by_severity"]],
            color=MALLANET,
            linestyle="--",
            marker="o",
            markerfacecolor="white",
            markeredgecolor=MALLANET,
            markeredgewidth=1.0,
            markersize=4,
            linewidth=1.6,
            label="MallaNet (17.32M)",
        )

        ax.set_title(
            f"{title} (mCA B/M: {student['mCA'] * 100:.1f}/{mallanet['mCA'] * 100:.1f}%)",
            fontsize=10,
            pad=7,
        )
        ax.set_xlabel("Corruption severity (0=clean)")
        ax.set_xticks(severities)
        ax.set_xlim(-0.15, 5.15)
        ax.grid(True, linewidth=0.6, alpha=0.55)

    axes[0].set_ylabel("DHCD test accuracy (%)")
    axes[0].set_ylim(0, 103)
    axes[0].legend(loc="lower left", fontsize=8, frameon=True)

    fig.tight_layout()
    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
