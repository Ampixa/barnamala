# DevNet — Statistically-Defended SOTA for Handwritten Devanagari Recognition

Target: beat 99.72% (Mishra et al. 2021) on the 46-class DHCD benchmark with a
McNemar-significant margin, and dominate the parameter-efficiency frontier with
a ~1.1M-parameter distilled student. Spec: `docs/superpowers/specs/`.

## Setup

    python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
    ./scripts/download_data.sh
    .venv/bin/pytest                      # all unit tests + real-data smoke gate

## Full experiment program (rented GPU)

    # 1. Teachers: 3 configs x 5 seeds (~15 runs)
    ./scripts/run_seeds.sh configs/teacher_a.yaml
    ./scripts/run_seeds.sh configs/teacher_b.yaml
    ./scripts/run_seeds.sh configs/teacher_c.yaml

    # 2. Evaluate each teacher on test (predictions saved per run)
    for d in results/teacher_*_seed*/; do
        .venv/bin/python scripts/evaluate_checkpoint.py "$d/best.pth" --tta
    done

    # 3. Ensemble + dump train logits for distillation
    .venv/bin/python scripts/dump_teacher_logits.py results/teacher_*_seed*/best.pth

    # 4. Distill the student (5 seeds) + supervised control
    ./scripts/run_seeds.sh configs/student_distilled.yaml
    ./scripts/run_seeds.sh configs/student.yaml

    # 5. Baseline + paired stats
    git clone --depth 1 https://github.com/sahajrajmalla/MallaNet /tmp/mallanet_repo
    .venv/bin/python scripts/reproduce_mallanet.py
    .venv/bin/python scripts/run_mcnemar.py \
        results/student_distilled_seed0/test_predictions.npz \
        results/mallanet_baseline/test_predictions.npz

## Statistical protocol

The test set is evaluated once per final configuration. Model selection uses a
10% stratified validation carve-out. Reported: mean±std over 5 seeds, Wilson
95% CIs, exact McNemar paired tests vs the reproduced MallaNet checkpoint.
