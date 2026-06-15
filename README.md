# Barnamala — A 1.1M-Parameter Student at SOTA Parity on Handwritten Devanagari

**वर्णमाला** ("the alphabet"). A 1.11M-parameter distilled SE-ResNet student that
matches the 17.32M-parameter MallaNet SOTA on the 46-class DHCD benchmark —
**15.6× fewer parameters at statistical parity** (McNemar p > 0.3) — *even
without distillation*.

This repository ships the full pipeline **and the prediction dumps + checkpoints**
so every figure in [`results/SUMMARY.md`](results/SUMMARY.md) is recomputable
with no GPU.

## Headline result

| Model | Params | Test acc (DHCD, n=13,800) | Significantly beats MallaNet? |
|---|---|---|---|
| MallaNet (Malla, Sci. Reports 2025), reproduced | 17.32M | 99.710% (40 errors) | — |
| **Barnamala** student, distilled (mean of 5 seeds) | **1.11M** | **99.735%** (36.6 errors) | no (McNemar p = 0.345) |
| Barnamala student, supervised control | 1.11M | 99.712% | no (p = 0.701) |

**Honest framing:** we do *not* claim a statistically-significant win over
MallaNet. The benchmark is saturated at an ~11-error intrinsic/label-noise floor
that every model in this study — ours and MallaNet — shares. The defensible
contribution is **extreme parameter efficiency at parity** plus a clean
flip-TTA-is-harmful ablation and a characterization of the benchmark ceiling.
Full tables, negative results, and error analysis: [`results/SUMMARY.md`](results/SUMMARY.md).

> Implementation note: the student model class is named `DevNet` in code
> (`devnet` Python package) for historical reasons; "Barnamala" is the
> paper/model name.

## Setup

    python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
    ./scripts/download_data.sh
    .venv/bin/pytest                      # all unit tests + real-data smoke gate

## Reproduce the numbers (no GPU needed)

All prediction dumps are committed under `results/`. Recompute any figure:

    .venv/bin/python scripts/analyze_errors.py            # error structure + floor
    .venv/bin/python scripts/run_mcnemar.py \
        results/student_distilled_seed0/test_predictions.npz \
        results/mallanet_baseline/test_predictions.npz    # paired test vs MallaNet
    .venv/bin/python scripts/eval_ensemble.py results/teacher_*_seed*/best.pth

## Full experiment program (rented GPU)

    # 1. Teachers: 3 configs x 5 seeds (15 runs)
    ./scripts/run_seeds.sh configs/teacher_a.yaml
    ./scripts/run_seeds.sh configs/teacher_b.yaml
    ./scripts/run_seeds.sh configs/teacher_c.yaml

    # 2. Evaluate each teacher on test (predictions saved per run)
    for d in results/teacher_*_seed*/; do
        .venv/bin/python scripts/evaluate_checkpoint.py "$d/best.pth"
    done

    # 3. Ensemble + dump train logits for distillation (use --no-tta: flip-TTA is harmful)
    .venv/bin/python scripts/dump_teacher_logits.py results/teacher_*_seed*/best.pth --no-tta

    # 4. Distill the student (5 seeds) + supervised control
    ./scripts/run_seeds.sh configs/student_distilled.yaml
    ./scripts/run_seeds.sh configs/student.yaml

    # 5. Baseline + paired stats
    git clone --depth 1 https://github.com/sahajrajmalla/MallaNet /tmp/mallanet_repo
    .venv/bin/python scripts/reproduce_mallanet.py

## Statistical protocol

The test set is evaluated once per final configuration. Model selection uses a
10% stratified validation carve-out. Reported: mean±std over 5 seeds, Wilson
95% CIs, and exact two-sided McNemar paired tests vs the reproduced MallaNet
checkpoint (unpaired CIs always overlap at this accuracy ceiling, so only the
paired test is informative).
