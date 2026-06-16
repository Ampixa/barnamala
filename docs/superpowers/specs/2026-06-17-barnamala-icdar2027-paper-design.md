# Design: Barnamala — Parameter-Efficient Handwritten Devanagari Recognition (ICDAR 2027 paper)

**Date:** 2026-06-17
**Status:** Design approved; spec under review. Experiments + manuscript pending.
**Working model name:** "Barnamala" (वर्णमाला) for the public/arXiv version; anonymized codename (e.g. **CDN** = "compact Devanagari network") for the double-blind ICDAR submission.
**Primary venue:** ICDAR 2027 (Springer LNCS, **17 pages max incl. everything**, **double-blind**, full-paper deadline **2027-01-31**, conf. Aug 18–22 2027, Kuala Lumpur).
**Near-term venue:** arXiv preprint (non-anonymous), posted once the draft is solid.

This design builds on the completed experiment program summarized in `results/SUMMARY.md`
and the earlier pipeline spec `docs/superpowers/specs/2026-06-11-devanagari-sota-design.md`.
All headline numbers already exist in local prediction dumps; the paper adds three
new, mostly-CPU experiments (efficiency metrics, digit transfer, corruption robustness).

## 1. Positioning and thesis

**Balanced, efficiency-led.** The headline is a constructive efficiency result; the
benchmark-saturation / significance finding is a rigorous, diplomatically-worded
secondary contribution contained in one analysis section (not a polemic against the
community). We do **not** claim a statistically significant win over MallaNet.

**One-line thesis:** A 1.11M-parameter student matches the 17.32M-parameter MallaNet at
statistical parity (McNemar p > 0.3) on DHCD — 15.6× smaller, and it ties even *without*
distillation — while a rigorous paired protocol shows recent SOTA gains sit within noise
against an ~11-error intrinsic floor: DHCD is saturated.

## 2. Contributions (as they will appear)

- **C1 — Compact model at SOTA parity.** 1.11M params, statistical parity with 17.32M
  MallaNet (15.6× smaller); the supervised control alone (no distillation) already ties it.
- **C2 — Rigorous evaluation protocol → saturation.** Exact McNemar + Wilson CIs reveal an
  ~11-error intrinsic floor shared by every model (ours and MallaNet); recent gains are
  within noise. DHCD is saturated.
- **C3 — Ablations.** Ensemble distillation gives a small, non-significant (~3-error) gain
  insensitive to teacher-signal cleanliness; flip-TTA is harmful for script data.
- **C4 — Efficiency, transfer, robustness (new).** FLOPs/latency/memory; DHCD→CMATERdb
  digit distribution-shift transfer; corruption robustness vs MallaNet.
- **C5 — Reproducible artifacts**, released (anonymized mirror for review).

## 3. Paper structure (LNCS, ≤17 pp)

1. **Introduction** (~1.5pp) — DHCD near-saturation; large SOTA models; untested
   significance in the subfield. Contributions C1–C5.
2. **Related work** (~1.5pp) — Indic/Devanagari HCR (MallaNet, Mishra, capsule/HFC nets);
   knowledge distillation & compact models; significance testing / benchmark saturation
   (the methodological gap this paper fills).
3. **Method** (~2.5pp) — 3.1 compact SE-ResNet student (widths 40/80/160, depths 2/2/2,
   1.11M params); 3.2 teacher ensemble (3 configs × 5 seeds = 15 teachers, 5.98–9.90M each);
   3.3 ensemble knowledge distillation (KD loss; clean no-TTA targets); 3.4 script-aware
   training (horizontal flips excluded as semantically invalid; weight EMA; cosine warmup;
   mixup/cutmix); 3.5 **evaluation protocol** (single test pass; 10% stratified-val model
   selection; exact two-sided McNemar; Wilson 95% CIs; ECE).
4. **Experiments** (~4–5pp) — 4.1 setup + exact MallaNet reproduction (incl. class-permutation
   recovery); 4.2 main parity result; 4.3 efficiency metrics (new); 4.4 distillation ablation
   (TTA vs clean targets vs supervised control) + teacher-ensemble scoring; 4.5 flip-TTA
   ablation; 4.6 DHCD→CMATERdb digit transfer (new); 4.7 corruption robustness (new).
5. **Analysis: why DHCD is saturated** (~2pp) — the 11-error floor; error correlation
   (pairwise Jaccard ≈ 0.5); confusable pairs (dha/gha, ba/waw, da/dhaa, tra/ba);
   significance frontier (error count needed for p<0.05 and why no lever reaches it);
   calibration (ECE 0.13–0.16); explicit "no significant win" statement.
6. **Discussion / limitations** (~0.5pp) — transfer is digit-only because the only public
   Devanagari sets sharing DHCD's 36-consonant labels are same-provenance (Acharya & Pant)
   and contaminated; single-benchmark for consonants; implications for efficiency-vs-accuracy.
7. **Conclusion** (~0.5pp).

## 4. Figures and tables

- **F1** student architecture · **F2** efficiency frontier (params & FLOPs vs accuracy:
  ours vs MallaNet, Mishra, Saini) · **F3** the 11-error floor (error-set overlap) ·
  **F4** corruption-robustness curves + hard-pair example glyphs.
- **T1** main parity (distilled / clean / supervised / MallaNet: errors, acc, params,
  McNemar) · **T2** efficiency (params, FLOPs, CPU & edge latency, peak memory) ·
  **T3** distillation + flip-TTA ablations · **T4** transfer + robustness · **T5** the
  15-teacher ensemble.

## 5. New experiments to run (before the draft is complete)

All run from existing checkpoints; only the transfer fine-tune needs a GPU (small).

1. **Efficiency metrics (CPU, local).** Compute params (have), FLOPs/MACs (e.g. `fvcore`/
   `thop`), single-image CPU latency (mean±std over N runs), an edge-class latency proxy,
   and peak inference memory — for the student and the reproduced MallaNet. Produces T2 + F2.
2. **DHCD→CMATERdb digit transfer (mostly CPU).**
   - Datasets: **CMATERdb 3.2.1** (Devanagari numerals, 10 classes, independent provenance,
     Apache-2.0) as primary; **Mendeley pxrnvp4yy8** (10 numerals + 13 vowels, CC-BY-4.0) as
     a second digit sample + open-set vowel probe.
   - **Provenance guard:** perceptual-hash dedup of CMATERdb/Mendeley digit images vs DHCD
     digit images; report that no near-duplicates exist (makes "real shift" defensible).
   - Protocol: zero-shot (DHCD digit head → CMATERdb test) and light fine-tune (small GPU).
     Report accuracy + the efficiency story carrying over. Frame honestly as *digit*
     distribution-shift robustness (10/46 classes).
3. **Corruption robustness (CPU, eval-only).** Apply the existing corruption suite
   (noise/blur/contrast at several severities) to the DHCD test set; evaluate the student
   vs reproduced MallaNet from saved checkpoints. Produces F4 + part of T4.

## 6. Anonymization plan (double-blind compliance)

ICDAR rejects non-anonymized manuscripts **without review**. Two layers:

- **Manuscript:** no author names/affiliations/acknowledgements; cite our own prior work in
  the third person; neutral model name (**CDN**); no "Ampixa"/"Barnamala"/blog references.
- **Artifacts:** do **not** link `Ampixa/barnamala`. Prepare an **anonymized snapshot** — a
  scrubbed copy (fresh squashed repo or dedicated branch) with identifying strings removed
  (model→CDN, no Ampixa/Barnamala/author names/real URLs, history squashed so commit
  metadata doesn't leak) — and serve it via **anonymous.4open.science**; put that random URL
  in the paper. (Alternatively attach a scrubbed zip as supplementary material.)
- **Camera-ready (post-acceptance):** restore real names, Ampixa affiliation, the real
  `Ampixa/barnamala` link, the "Barnamala" name, and the arXiv id.

## 7. Publication strategy and timeline

- **Milestone 1 — arXiv preprint (non-anonymous).** Once the draft + new experiments are
  solid: post to arXiv (cs.CV) under real names with the real repo link. Stakes priority,
  citable now. Check whether an arXiv endorsement is needed for first-time cs.CV submitters.
  arXiv is permanent (withdrawal leaves a tombstone) → post only when solid.
- **Milestone 2 — ICDAR 2027 submission (anonymized).** By 2027-01-31: LNCS ≤17pp,
  anonymized manuscript + anonymized artifact mirror. Do not advertise the arXiv preprint
  during review.
- **Milestone 3 — camera-ready** (if accepted): de-anonymize fully.

ICDAR explicitly permits arXiv preprints; the two tracks are compatible.

## 8. Out of scope / non-claims

- No claim of a statistically significant win over MallaNet.
- No new architecture novelty claim (the student is an SE-ResNet; the contribution is
  empirical/analytical).
- No cross-script generalization claim (declined); no consonant-level transfer claim
  (not cleanly measurable from uncontaminated public data).

## 9. Reproducibility

Public repo `Ampixa/barnamala` already ships the full pipeline + 37 prediction dumps + 31
checkpoints so every figure is recomputable with no GPU. New experiment scripts
(efficiency, transfer, robustness) will be added there and mirrored, anonymized, for review.

## 10. Open questions / risks

- arXiv endorsement availability for the submitting account (verify before Milestone 1).
- Reviewer pushback that the contribution is "analysis, not method" — mitigated by the
  efficiency-led framing + concrete new experiments (C4).
- CMATERdb/Mendeley download + license attribution confirmed before use; dedup must pass.
- LNCS 17-page budget is generous but the analysis section must stay disciplined.
