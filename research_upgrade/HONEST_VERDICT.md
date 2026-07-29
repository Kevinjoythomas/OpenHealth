# HONEST_VERDICT.md — v4 (2026-07-12, after 6 corroborating runs + a 5-lens hostile review + all-majors fix)

## What the paper now is
A rigorous, honest ML4H cautionary+method paper with a verified headline and, now, **six independent lines of corroboration**. On 96 novel, contamination-free vignettes a base Llama-3-8B and its QLoRA fine-tune **under-triage emergencies equivalently** (both ≈24%; case-level TOST-equivalent within ±6.2 pts), yet a keyword safety rubric certifies the fine-tune as **6.5× safer** (a 55-point gap). The rubric's validity **collapses** for the deployed model (κ 0.61→0.03; AUROC 0.91→0.67; of its "safe" answers 75% are under-triage vs 40% for base), traced to a corpus that escalates only 13% of emergencies while saturating with the *doctor* register (69.3%) not urgency (911: 0.0%).

## The six corroborations (all run, all reproducible from released artifacts)
1. **Primary** clean-benchmark (768 gens, judged) — the headline.
2. **Multi-seed** (seeds 101–103): every headline quantity is seed-stable (ft κ 0.03–0.05, kw-safe 88–91%, escalation 20–24%).
3. **Corpus-adjacent** original set, corrected 2048-ctx (§4.7): collapse replicates (κ 0.50→0.06).
4. **External, physician-authored** HealthBench (§4.12): over-crediting replicates (ft kw 2.4× base, no appropriateness gain; PPV-failure 4.3×).
5. **Multi-judge IRR** (§4.6): Fleiss κ=0.30 across Claude+Qwen+Mistral; the base-vs-ft equivalence is **judge-robust** (majority-of-3 43% vs 49%), only the absolute rate is judge-dependent.
6. **Generalization panel** (§4.11 Table 3): a validated invalidation diagnostic flags **only** the ChatDoctor fine-tune; an independently-trained medical fine-tune (Medllama2) genuinely improves triage and is not gamed — scoping the catastrophe to the corpus and elevating the contribution to a *diagnostic method*.

## Independent adversarial review (5 lenses + area chair), 2026-07-12
Mean **3.2 (borderline, Major Revision)** — but every reviewer **reproduced the numbers from the released artifacts** and found no fabrication. All **6 majors fixed from existing data** and re-verified (reviews_v3_and_fixes.md): the joint-vs-conditional flagship-number error (base was understated ~3×), the TOST equivalence over-claim, the missing Goodhart/reward-hacking prior art, the "systemic" over-extrapolation, the under-substantiated rubric-as-practice claim, and a chance-level arithmetic error. Every correction moved numbers in the *un-flattering* direction — the credibility signal the AC called highest-leverage.

## Probability of acceptance (honest)
- **As-is (no clinician anchor): borderline-to-weak-accept (~55–65%)** — materially up from the ~35–45% of v3. The six corroborations (esp. external HealthBench + the negative generalization + seed stability), the fixed flagship number, and the Goodhart positioning are the drivers.
- **With the clinician anchor returned: weak-accept (~65–72%).** It converts the judge-dependent *absolute* rates (which the IRR shows do vary by judge) into human-anchored ones and closes the last circularity.

## The ONE thing between here and submission-ready-strong
**Return `human_validation_sheet.html`** (~40 blinded emergencies, ~30 min of clinician time). The IRR (§4.6) makes this sharper, not weaker: the base-vs-fine-tune *contrast* is judge-robust, but the *absolute* under-triage rate genuinely varies by judge (Claude ~24% strict, open judges ~46% lenient) — a human pins which is right. This is the single highest-leverage remaining action and only you can do it.

## Pre-submission formatting (mechanical, not blocking the science)
- **Trim to ML4H 8pp**: paper is ~6.5k words; move §4.4/4.5/4.6-detail/4.7/4.9/4.12 to an Appendix (ML4H excludes appendix + refs from the 8pp).
- **Anonymize**: 4 "OpenHealth"/"openhealth-doctor" mentions → neutral (the provenance note is already marked "not part of the paper" and is stripped).
- **Bib**: add the 6 verified Goodhart/reward-hacking/shortcut-learning citations (listed in reviews_v3_and_fixes.md) to `comparison_matrix.md` + PMLR bib.

## Venue
**ML4H 2026 Proceedings** (PMLR, 8pp) primary; **CHIL 2027** / **TMLR** backup.

## Bottom line
The science is done, honest, hostile-review-vetted, and corroborated six ways. The paper is a genuinely strong, well-defended contribution — a documented, *diagnosable* failure mode of deployed medical-LLM safety evaluation, scoped correctly. The remaining gap to a confident accept is one clinician anchor (yours) plus mechanical formatting.
