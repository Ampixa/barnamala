"""F4: corruption robustness, 3 panels (noise/blur/contrast), student vs MallaNet."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rob = json.load(open("results/robustness.json"))
kinds = ["noise", "blur", "contrast"]
sev = list(range(6))
fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.8), sharey=True)
for ax, kind in zip(axes, kinds):
    for name, lab in [("student", "Barnamala/CDN"), ("mallanet", "MallaNet")]:
        ys = [v * 100 for v in rob[name]["per_corruption"][kind]["by_severity"]]
        ax.plot(sev, ys, marker="o", ms=3, label=lab)
    ax.set_title(kind)
    ax.set_xlabel("severity")
    ax.grid(True, ls=":", alpha=0.5)
axes[0].set_ylabel("accuracy (\\%)")
axes[0].legend(fontsize=7, loc="lower left")
fig.savefig("paper/figures/robustness_curves.pdf", bbox_inches="tight")
print("wrote paper/figures/robustness_curves.pdf")
