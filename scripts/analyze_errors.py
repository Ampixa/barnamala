"""Error-structure analysis: what separates our students from MallaNet, and
what it would take to beat it with McNemar significance.

Loads the local test-prediction dumps and reports overlap structure, the
student soft-vote ensemble, the intrinsic-difficulty floor, per-class error
concentration, and the significance arithmetic. Read-only; writes nothing.
"""
import glob
import numpy as np
from scipy.stats import binomtest

from devnet.data import load_split
from devnet.evaluate import mcnemar_exact

ROOT = "results"
DATA = "data/extracted/DevanagariHandwrittenCharacterDataset"
_, _, CLASSES = load_split(DATA, "Test")


def load(path):
    d = np.load(path)
    return d["logits"], d["preds"], d["labels"]


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


# --- load baseline + the completed distilled students ---
mb_logits, mb_preds, labels = load(f"{ROOT}/mallanet_baseline/test_predictions.npz")
students = {}
for p in sorted(glob.glob(f"{ROOT}/student_distilled_seed*/test_predictions.npz")):
    seed = p.split("seed")[1].split("/")[0]
    students[f"distilled_s{seed}"] = load(p)
mb_ok = mb_preds == labels


def mcnemar(cand_preds):
    c_ok = cand_preds == labels
    n10 = int((~mb_ok & c_ok).sum())   # MallaNet wrong, candidate right
    n01 = int((mb_ok & ~c_ok).sum())   # candidate wrong, MallaNet right
    return n10, n01, mcnemar_exact(n10, n01), int((~c_ok).sum())


print(f"Test set n={len(labels)}.  MallaNet baseline: {(~mb_ok).sum()} errors\n")
print(f"{'model':16s} {'errs':>4} {'n10':>4} {'n01':>4} {'concord':>7} {'McNemar p':>10}")
print("-" * 52)
ens_prob = np.zeros((len(labels), 46))
for name, (lg, pr, _) in students.items():
    n10, n01, p, errs = mcnemar(pr)
    concord = int(((~mb_ok) & (pr != labels)).sum())  # both wrong
    print(f"{name:16s} {errs:>4} {n10:>4} {n01:>4} {concord:>7} {p:>10.4f}")
    ens_prob += softmax(lg)

# --- student soft-vote ensemble (concrete improvement #1) ---
ens_preds = ens_prob.argmax(1)
n10, n01, p, errs = mcnemar(ens_preds)
concord = int(((~mb_ok) & (ens_preds != labels)).sum())
print(f"{'ENSEMBLE(soft)':16s} {errs:>4} {n10:>4} {n01:>4} {concord:>7} {p:>10.4f}")

# --- teacher ensemble (TTA dumps; informative, TTA known to hurt) ---
t_paths = sorted(glob.glob(f"{ROOT}/teacher_*_seed*/test_predictions_tta.npz"))
if t_paths:
    tprob = np.zeros((len(labels), 46))
    for tp in t_paths:
        s, _, _ = load(tp)
        tprob += s / s.sum(1, keepdims=True)  # stored scores are TTA probs
    tpreds = tprob.argmax(1)
    n10, n01, p, errs = mcnemar(tpreds)
    print(f"{'TEACHERS(9,TTA)':16s} {errs:>4} {n10:>4} {n01:>4} "
          f"{int(((~mb_ok)&(tpreds!=labels)).sum()):>7} {p:>10.4f}")

# --- intrinsic-difficulty floor: samples ALL students get wrong ---
all_wrong = np.ones(len(labels), bool)
for _, pr, _ in students.values():
    all_wrong &= (pr != labels)
core = all_wrong
core_and_mb = core & ~mb_ok
print(f"\nConsensus errors (all {len(students)} students wrong): {core.sum()}")
print(f"  ...also wrong for MallaNet (intrinsic/label-noise floor): {core_and_mb.sum()}")

# --- significance arithmetic: what n01 budget is allowed ---
print("\nMcNemar significance frontier (two-sided exact, alpha=0.05):")
print("  if we FIX k of MallaNet's 40 errors (n10=k), max n01 to keep p<0.05:")
for k in (12, 15, 18, 20, 25):
    best = 0
    for n01 in range(0, k + 1):
        if mcnemar_exact(k, n01) < 0.05:
            best = n01
    cand_err = 40 - k + best  # baseline 40 = concord + n10
    print(f"    n10={k:2d} -> n01<={best:2d}  (=> candidate would have ~{cand_err} errors)")

# --- where the ensemble's errors live (per-class + confusion) ---
ens_err = ens_preds != labels
print(f"\nEnsemble error breakdown ({ens_err.sum()} errors):")
own_goals = ens_err & mb_ok          # we wrong, MallaNet right  (the n01 to kill)
print(f"  own-goals (we wrong, MallaNet right): {own_goals.sum()}")
from collections import Counter
conf = Counter()
for i in np.where(ens_err)[0]:
    conf[(CLASSES[labels[i]], CLASSES[ens_preds[i]])] += 1
print("  top confusions (true -> predicted):")
for (t, pdc), c in conf.most_common(10):
    tag = ""
    print(f"    {t:24s} -> {pdc:24s}  x{c}")
print("  own-goal confusions (highest-leverage to fix):")
og = Counter()
for i in np.where(own_goals)[0]:
    og[(CLASSES[labels[i]], CLASSES[ens_preds[i]])] += 1
for (t, pdc), c in og.most_common(8):
    print(f"    {t:24s} -> {pdc:24s}  x{c}")
