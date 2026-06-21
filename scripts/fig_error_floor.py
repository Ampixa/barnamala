"""Publication figure: agreement structure of DHCD errors across 16 models."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


OUT = Path("paper/figures/error_floor.pdf")


def short_class_name(name):
    if name.startswith("character_"):
        return name.split("_", 2)[2]
    if name.startswith("digit_"):
        return f"digit {name.split('_', 1)[1]}"
    return name.replace("_", " ")


def main():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})

    es = json.load(open("results/error_structure.json"))
    n_models = es["n_models"]
    shared_by_k = es["shared_by_k"]
    ks = list(range(1, n_models + 1))
    vals = shared_by_k[1 : n_models + 1]

    cmap = LinearSegmentedColormap.from_list(
        "floor_gradient", ["#D9ECFF", "#F7B267", "#B2182B"]
    )
    colors = [cmap(i / (len(ks) - 1)) for i in range(len(ks))]

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.bar(ks, vals, color=colors, edgecolor="white", linewidth=0.6)

    ax.axvline(n_models, color="#B2182B", linestyle="--", linewidth=1.1)
    ax.annotate(
        f"All-model floor: {es['floor_all_models']} images",
        xy=(n_models, es["floor_all_models"]),
        xytext=(-86, 28),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="-", color="#B2182B", lw=0.8),
        color="#B2182B",
        fontsize=8,
        ha="left",
        va="bottom",
    )

    confusion_lines = ["Top confused pairs"]
    for src, dst, count in es["top_confusions"][:3]:
        confusion_lines.append(f"{short_class_name(src)} -> {short_class_name(dst)}: {count}")
    ax.text(
        6.3,
        23.8,
        "\n".join(confusion_lines),
        fontsize=8,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#C7C7C7", alpha=0.92),
    )

    ax.set_xlabel(f"Models misclassifying the same test image (of {n_models})")
    ax.set_ylabel("Test images")
    ax.set_xticks(ks)
    ax.set_xlim(0.4, n_models + 0.6)
    ax.set_ylim(0, 25)
    ax.grid(True, axis="y", linewidth=0.6, alpha=0.55)
    ax.grid(False, axis="x")

    fig.tight_layout()
    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
