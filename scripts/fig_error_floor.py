"""F3: how many test images are misclassified by exactly k of the 16 models.
The spike at k = n_models is the intrinsic floor (11)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

es = json.load(open("results/error_structure.json"))
sbk = es["shared_by_k"]
ks = list(range(1, len(sbk)))          # drop k=0 (correctly classified by all)
vals = sbk[1:]
fig, ax = plt.subplots(figsize=(4.2, 3.0))
ax.bar(ks, vals)
ax.axvline(es["n_models"], color="C3", ls="--", lw=1)
ax.annotate(f"intrinsic floor = {es['floor_all_models']}",
            (es["n_models"], max(vals) * 0.6), fontsize=7, ha="right", color="C3")
ax.set_xlabel("number of models (of %d) misclassifying an image" % es["n_models"])
ax.set_ylabel("test images")
ax.grid(True, axis="y", ls=":", alpha=0.5)
fig.savefig("paper/figures/error_floor.pdf", bbox_inches="tight")
print("wrote paper/figures/error_floor.pdf")
