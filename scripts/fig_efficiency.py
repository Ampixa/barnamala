"""F2: parameter/accuracy efficiency frontier. Ours + MallaNet from artifacts;
Mishra/Saini from the literature (cited in-text)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

eff = json.load(open("results/efficiency_metrics.json"))
s = eff["models"][0]
# (label, params_millions, accuracy_pct, marker)
pts = [
    ("CDN (ours)", s["params_millions"], 99.735, "*"),
    ("MallaNet [Malla 2025]", eff["models"][1]["params_millions"], 99.710, "o"),
    # Mishra et al. 2021 (INDISCON) accuracy 99.72% confirmed; parameter count
    # (~39 M) could not be verified from the primary source — point omitted.
]
fig, ax = plt.subplots(figsize=(4.2, 3.0))
for label, p, acc, m in pts:
    ax.scatter(p, acc, marker=m, s=90)
    ax.annotate(label, (p, acc), fontsize=7,
                xytext=(5, 4), textcoords="offset points")
ax.set_xscale("log")
ax.set_xlabel("Parameters (millions, log scale)")
ax.set_ylabel("DHCD test accuracy (\\%)")
ax.set_ylim(99.6, 99.85)
ax.grid(True, which="both", ls=":", alpha=0.5)
fig.savefig("paper/figures/efficiency_frontier.pdf", bbox_inches="tight")
print("wrote paper/figures/efficiency_frontier.pdf")
