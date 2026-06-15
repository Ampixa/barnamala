# Barnamala — Experiment Summary

> "Barnamala" (वर्णमाला, "the alphabet") is the model/paper name; the student
> class is `DevNet` in code.


Handwritten Devanagari (DHCD, 46 classes). Test set evaluated once per final
configuration; model selection on a 10% stratified validation carve-out.
Test n = 13,800. All figures computed from the prediction dumps in this folder
(`scripts/analyze_errors.py`, `scripts/eval_ensemble.py`, `scripts/run_mcnemar.py`).

Baseline: **MallaNet** (Malla, Sci. Reports 2025) reproduced exactly —
**40 errors, 99.7101%**, 17.32M params. Paired significance uses the exact
two-sided McNemar test against this reproduced checkpoint (unpaired CIs always
overlap at this accuracy ceiling, so only the paired test is informative).

## Headline

A **1.11M-parameter** student reaches **99.73%** on DHCD — statistically
indistinguishable from the **17.32M-parameter** MallaNet SOTA (**15.6× fewer
parameters**), *even without distillation*. No configuration beats MallaNet
with McNemar significance: the benchmark is saturated at an **11-error
intrinsic floor** that every model — ours and MallaNet — shares.

## Main results (test, n = 13,800)

| Configuration | Params | Errors (5 seeds) | Mean±std | Acc % | McNemar vs MallaNet (seed 0) |
|---|---|---|---|---|---|
| Student, **distilled** (TTA targets) | 1.11M | 34, 39, 35, 39, 36 | **36.6 ± 2.1** | 99.735 | n10=17, n01=11, **p=0.345** |
| Student, **distilled** (clean targets) | 1.11M | 39, 40, 34, 40, 38 | 38.2 ± 2.2 | 99.723 | n10=16, n01=15, p=1.000 |
| Student, **supervised** (control) | 1.11M | 43, 40, 42, 35, 39 | 39.8 ± 2.8 | 99.712 | n10=12, n01=15, p=0.701 |
| MallaNet (reproduced baseline) | 17.32M | 40 | — | 99.710 | — |

Best single student: distilled seed 0 / clean seed 2, **34 errors (99.754%)**,
Wilson 95% CI [99.656, 99.824].

## Teacher ensembles (test, n = 13,800)

Teachers: 15 models (3 architectures × 5 seeds), 5.98M–9.90M params each.
The ensemble is used to generate distillation targets; it is *also* scored
here as a predictor (the original program never did this).

| Ensemble | Errors | Acc % | McNemar vs MallaNet | Significant? |
|---|---|---|---|---|
| 9 teachers, clean | 30 | 99.783 | n10=18, n01=8, p=0.0755 | no |
| **15 teachers, clean** | **30** | **99.783** | n10=18, n01=8, **p=0.0755** | no |
| 9 teachers, +flip-TTA | 28 | 99.797 | n10=19, n01=7, p=0.029 | yes — but fragile (see below) |

Per-model clean errors (15): 27, 28, 30, 30, 32, 34, 36, 37, 37, 39, 39, 40, 41, 42, 43.
The ensemble **plateaus at 30 errors**: scaling 9 → 15 teachers changed nothing.

## Two negative results (both phase-2 hypotheses)

1. **Clean distillation targets did not help.** Re-distilling from clean
   (no-TTA) ensemble logits instead of flip-TTA logits gave 38.2 ± 2.2 errors —
   no better than the TTA-target students (36.6 ± 2.1). The TTA corruption in
   the teaching signal was never the bottleneck.
2. **More teachers did not help.** The 15-teacher clean ensemble (30 errors,
   p=0.0755) is identical to the 9-teacher clean ensemble. The ceiling is not
   ensemble capacity.

## Test-time augmentation is harmful

Horizontal-flip TTA is semantically invalid for a script, and the data agree:

| | no-TTA | +flip-TTA |
|---|---|---|
| Distilled student, mean errors | 36.6 | 36.8 (seed 0: 34 → 40) |
| Supervised student, mean errors | 39.8 | 40.2 (variance ↑, std 2.8 → 4.6) |
| Individual teachers (worst cases) | ~30 | up to **88, 102** errors |

The lone "significant" result (9-teacher TTA ensemble, p=0.029) depends entirely
on this invalid augmentation and on a 1–2 sample margin, while TTA breaks several
individual teachers catastrophically. It is not a defensible claim.

## Error structure — why the ceiling exists

- **11 images** are misclassified by **all 15 of our students *and* MallaNet** —
  an intrinsic / label-noise floor no model in this study escapes.
- Students distilled from the same targets are highly correlated (pairwise error
  Jaccard ≈ 0.5), so adding seeds reduces variance, not the shared bias.
- Errors concentrate on visually confusable Devanagari pairs:
  dha/gha, ba/waw, da/dhaa, tra/ba.
- Students are over-confident (ECE ≈ 0.13–0.16): the few "own-goal" errors are
  low-margin flips, not systematic blind spots.

## Defensible claims for the paper

1. **Extreme parameter efficiency at SOTA parity.** 1.11M params match a 17.32M
   SOTA (statistical parity, McNemar p > 0.3) — 15.6× smaller — and the
   supervised control alone (no distillation) already ties MallaNet.
2. **Knowledge distillation gives a small, non-significant gain** (~3 errors)
   that is insensitive to teacher-signal cleanliness — distillation is optional
   here, not the source of the result.
3. **Flip-TTA is harmful for Devanagari** — a clean, evidence-backed ablation
   and a critique of its use in prior work.
4. **DHCD is saturated.** An 11-error shared floor across architectures bounds
   achievable accuracy; reported gains at this ceiling are within noise.

## What is NOT claimed

We do **not** claim to beat MallaNet with statistical significance. No lever
tried — more seeds, clean distillation targets, or 9 → 15 teachers — produced a
robust McNemar-significant win.

## Reproducibility

- Per-model / ensemble eval: `scripts/evaluate_checkpoint.py`, `scripts/eval_ensemble.py`
- Paired test: `scripts/run_mcnemar.py`
- Error-structure analysis: `scripts/analyze_errors.py`
- Baseline reproduction: `scripts/reproduce_mallanet.py`
- Run drivers: `~/run_program.sh` (phase 1), `scripts/run_phase2.sh` (15-teacher
  clean re-distill). Logs: `results/program.log`, `results/program2.log`.

Hardware: AWS EC2 g6.4xlarge (NVIDIA L4), ~60 GPU-hours total.
