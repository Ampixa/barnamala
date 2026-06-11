# Design: Statistically-Defended SOTA for Handwritten Devanagari Character Recognition

**Date:** 2026-06-11
**Status:** Approved
**Working model name:** DevNet (placeholder — final name chosen before manuscript submission)
**Target venue:** Scientific Reports (same venue as MallaNet)

## 1. Background and motivation

MallaNet (Malla, Scientific Reports 2025, DOI 10.1038/s41598-025-30871-z) reports 99.71%
test accuracy on the Devanagari Handwritten Character Dataset (DHCD) with 17.32M
parameters. The standing SOTA is 99.72% (Mishra et al., IEEE INDISCON 2021, ResNet-85,
~39M parameters), unbeaten since 2021. A literature sweep (2026-06-11; OpenAlex citations
of the DHCD dataset paper and of Mishra et al., plus keyword searches) confirms no
published result above 99.72% on the full 46-class DHCD test set. Claims above that
number are digits-only (10 classes), numerals-only, or training-accuracy figures.

Three exploitable gaps in the existing literature:

1. **Accuracy ceiling is soft.** DHCD is CIFAR-shaped (32×32, balanced, 78,200 train
   images), yet no prior work imports the modern small-image training playbook
   (strong augmentation policies, cosine schedules, weight EMA, ensemble distillation)
   that routinely cuts error 2–3× relative to 2015–2021-era training.
2. **Pareto frontier hole.** Nothing exists between Saini et al. (0.4M params, 99.21%)
   and MallaNet (17.3M, 99.71%). A ~1–1.5M-parameter model at ≥99.70% dominates the
   efficiency frontier.
3. **Zero statistical rigor in the field.** No prior DHCD work reports multi-seed means,
   confidence intervals, or significance tests. At this ceiling, published results
   differ by 1–2 test images out of 13,800.

## 2. Claims the paper will make

| # | Claim | Success criterion |
|---|-------|-------------------|
| C1 | New SOTA on 46-class DHCD | Ensemble+TTA ≥ 99.82% (≤25 errors), McNemar p < 0.05 vs reproduced MallaNet |
| C2 | Pareto dominance | Distilled student ~1–1.5M params at ≥ 99.70% accuracy, single forward pass (no TTA, no ensemble) |
| C3 | First statistically rigorous evaluation on this benchmark | 5-seed mean±std, Wilson 95% CIs, exact McNemar paired tests |
| C4 | Robustness and ceiling analysis | Degradation curves (noise/blur/contrast) vs reproduced MallaNet; manual audit of all residual test errors |

### Statistical grounding for C1 (computed 2026-06-11, n = 13,800)

- Wilson 95% CIs overlap for any realistic pair of published accuracies at this ceiling
  (e.g., 99.82% → [99.733, 99.877] vs MallaNet 99.71% → [99.606, 99.787]). Unpaired
  comparison cannot establish superiority; **paired McNemar is the only rigorous path.**
- McNemar exact test vs a 40-error baseline reaches p < 0.05 at ≈25 errors (99.82%)
  given moderate error overlap, and p < 0.01 at ≈20 errors (99.855%). Hence the C1
  target of 99.82–99.86%: roughly halving MallaNet's error count.

### Fallback framings (paper survives all outcomes)

- Ensemble lands 99.73–99.81% (not significant): reframe C1 as "statistically
  indistinguishable from SOTA at 12–17× fewer parameters"; C2–C4 carry the paper.
- Residual errors are dominated by ambiguous/mislabeled images: the manual audit (C4)
  becomes an "effective ceiling of DHCD" contribution in its own right.
- Student cannot hold 99.70% at 1.5M: relax to 2–3M parameters (still 6–8× smaller
  than MallaNet); claim adjusted, not dropped.
- MallaNet checkpoint does not reproduce 99.71%: report the reproduction gap as a
  finding; McNemar comparison falls back to our own strong baseline (ResNet-18 trained
  with their recipe) plus published-number comparison.

## 3. Data pipeline

- **Dataset:** DHCD, standard split: 78,200 train / 13,800 test, 32×32 grayscale,
  46 classes (36 consonants + 10 digits), balanced (1,700 train + 300 test per class).
  Source: original release (UCI ML Repository mirror acceptable).
- **Validation:** 10% stratified carve-out from train (7,820 images) for early stopping
  and model selection. **The test set is evaluated exactly once per final model
  configuration.** This protocol is stated explicitly in the paper.
- **Normalization:** per-dataset mean/std (computed on train split only).
- **Script-aware augmentation:** rotation ±12°, translate/scale affine, elastic
  deformation, random erasing, mixup/cutmix. **Horizontal flips are excluded** —
  mirrored Devanagari characters are not valid characters. (MallaNet used horizontal
  flips; noted in the paper as a semantically incorrect augmentation for scripts.)
- Augmentation policy intensity is a swept hyperparameter (light/medium/heavy tiers).

## 4. Student architecture (headline model, C2)

Compact pre-activation residual ConvNet for 32×32×1 input, ~1–1.5M parameters:

- Stem: 3×3 conv → 48 channels.
- Three stages of pre-activation residual blocks with Squeeze-Excitation attention;
  widths 48/96/192; depth 2–3 blocks per stage; stride-2 downsample between stages.
  Exact width/depth fixed by a small sweep constrained to ≤1.5M parameters.
- Head: global average pooling → dropout → linear(46).
- No capsule layers. An ablation demonstrates SE + multi-scale residual features match
  or beat the HFC capsule mechanism at a fraction of the parameters.

## 5. Training recipe

- AdamW, cosine annealing with linear warmup, ~300 epochs, batch 256, label smoothing
  0.1, weight EMA, mixed precision (AMP).
- Every final configuration trained with **5 seeds**; report mean ± std.
- TTA (averaged small rotations/shifts) reported separately from single-pass numbers.
- Smoke test gate: a 2-minute CPU run on a data subset must exceed 90% val accuracy
  before any rented-GPU run is launched.

## 6. Teacher ensemble and distillation (C1 → C2)

- **Teachers:** 5–7 diverse models (varying seed, width up to ~8–10M params, depth,
  augmentation tier). Logit-averaged ensemble + TTA produces the C1 headline number.
- **Distillation:** KL divergence on temperature-softened ensemble logits + CE on hard
  labels, distilling into the compact student (C2 headline).
- **Reporting structure (pre-empts "brute force" objection):** three numbers reported
  separately — (a) single compact model, (b) distilled compact model, (c) full
  ensemble. Significance claimed only where McNemar grants it.

## 7. Evaluation and rigor layer (C3, C4)

- Metrics: accuracy, macro-F1, per-class F1, confusion analysis focused on confusable
  pairs (e.g., क/ख, ब/भ).
- Wilson 95% CI on accuracy; exact McNemar paired tests vs baselines.
- **MallaNet baseline reproduction:** run the author's published checkpoint
  (github.com/sahajrajmalla/MallaNet, `models/best_model.pth` — verified present) on
  the test set to obtain per-image predictions for paired testing. **Comparison
  baseline only — zero architectural inheritance.**
- Robustness: accuracy vs Gaussian noise σ, blur radius, contrast reduction — ours vs
  reproduced MallaNet.
- Calibration: expected calibration error (ECE).
- Efficiency: parameters, FLOPs, inference latency (CPU and GPU) vs prior work.
- Error-ceiling audit: manual inspection of every residual test error of the ensemble,
  categorized as model error / ambiguous / mislabeled.

## 8. Repository structure and engineering

```
mallanet/                      # project dir name; model renamed before submission
├── configs/                   # YAML per experiment
├── src/devnet/
│   ├── data.py                # dataset, splits, transforms
│   ├── models/                # student, teachers, baseline reimplementations
│   ├── train.py               # single-run trainer
│   ├── distill.py             # ensemble distillation
│   ├── evaluate.py            # test-set evaluation, prediction dumps
│   └── stats.py               # Wilson CI, McNemar, ECE
├── scripts/                   # download_data.sh, run_seeds.sh, reproduce_mallanet.py
├── tests/                     # see Testing below
├── results/                   # per-run config + git SHA + metrics CSV + test predictions
├── docs/superpowers/specs/
└── paper/                     # manuscript (later)
```

- PyTorch. Deterministic seeding. Every run logs config, git SHA, and metrics CSV.
- Per-image test predictions saved for every evaluated model so all statistics are
  recomputable without re-running training.

### Testing

- Data pipeline: split integrity (stratified, disjoint), no train/test leakage, no
  horizontal flip anywhere in the transform stack, normalization range checks.
- Models: parameter-count assertions (student ≤1.5M), output-shape checks.
- Training: 2-minute smoke train on subset reaching >90% val accuracy.

### Compute plan

- Develop and debug locally on CPU via the smoke test.
- Full runs on a rented single GPU (RTX 4090 class is sufficient; well under 1 hour
  per training run at 32×32). Estimated total program: 30–50 GPU-hours.

## 9. Out of scope (this paper)

- Word-level / continuous text recognition.
- Cross-script transfer (a second-dataset generalization section, e.g. CMATERdb, is an
  optional stretch goal — decided after C1/C2 results are in).
- Pretrained weights or external data: **everything trains from scratch on DHCD** for
  clean comparison with all prior work.
