#!/usr/bin/env bash
# Phase 2 (queued after the phase-1 program): scale to 15 teachers, re-dump
# CLEAN (no-TTA) ensemble logits, re-distil the student from the clean signal,
# and score it + the 15-teacher clean ensemble vs MallaNet. Writes its own
# markers to ~/program2.log; PROGRAM2_COMPLETE gates the money watchdog.
set -uo pipefail
cd ~/mallanet
LOG=~/program2.log

done_run() { [ -f "$1/best.pth" ] && [ "$(wc -l < "$1/log.csv" 2>/dev/null || echo 0)" -gt "$2" ]; }

# wait for phase 1 to finish before touching the GPU
until grep -q PROGRAM_COMPLETE ~/program.log 2>/dev/null; do sleep 120; done
echo "=== PHASE2 START $(date -u) ===" >> "$LOG"

# 1. extra teachers -> 15 total (seeds 3,4 of each config)
for cfg in teacher_a teacher_b teacher_c; do
  for seed in 3 4; do
    out=results/${cfg}_seed${seed}
    if done_run "$out" 300; then echo "skip $out (complete)" >> "$LOG"; continue; fi
    echo "=== $cfg seed $seed $(date -u) ===" >> "$LOG"
    .venv/bin/python -m devnet.train --config configs/$cfg.yaml --seed $seed >> "$LOG" 2>&1
  done
done

# 2. clean (no-TTA) logit re-dump from all 15 teachers
echo "=== CLEAN LOGIT DUMP $(date -u) ===" >> "$LOG"
.venv/bin/python scripts/dump_teacher_logits.py results/teacher_*_seed*/best.pth \
    --no-tta --out results/teacher_logits_clean.npz >> "$LOG" 2>&1

# 3. re-distil 5 students from the clean targets
echo "=== CLEAN STUDENTS (5 seeds) $(date -u) ===" >> "$LOG"
./scripts/run_seeds.sh configs/student_distilled_clean.yaml >> "$LOG" 2>&1

# 4. score clean students (no-TTA, the better setting) + 15-teacher ensemble
echo "=== CLEAN STUDENT EVAL $(date -u) ===" >> "$LOG"
for d in results/student_distilled_clean_seed*/; do
  .venv/bin/python scripts/evaluate_checkpoint.py "$d/best.pth" >> "$LOG" 2>&1
done
echo "=== 15-TEACHER CLEAN ENSEMBLE $(date -u) ===" >> "$LOG"
.venv/bin/python scripts/eval_ensemble.py results/teacher_*_seed*/best.pth >> "$LOG" 2>&1
echo "=== CLEAN McNEMAR (clean distilled seed0 vs MallaNet) $(date -u) ===" >> "$LOG"
.venv/bin/python scripts/run_mcnemar.py \
    results/student_distilled_clean_seed0/test_predictions.npz \
    results/mallanet_baseline/test_predictions.npz >> "$LOG" 2>&1

echo "PROGRAM2_COMPLETE $(date -u)" >> "$LOG"
