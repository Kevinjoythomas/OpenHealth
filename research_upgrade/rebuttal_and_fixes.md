# rebuttal_and_fixes.md
*Phase 6. Each reviewer criticism → response + status: **FIXED** (done this session), **PENDING** (running), **PROPOSED** (needs a run/Kevin), or **SCOPED** (addressed by framing).*

## R1 — "κ is prevalence-sensitive; the inversion rests on one LLM judge; style/length confound."
- **κ prevalence → FIXED.** We no longer lead with κ. The fine-tuned model emits keywords on 89% of answers, so we report prevalence-robust quantities: **raw agreement 0.31** (vs base 0.72), **P(keyword-safe ∧ under-triage)=0.68**, and **PABAK −0.37** (vs +0.43). All point the same way; κ is reported only for completeness (Table 2 reordered).
- **Length/style confound → FIXED.** Fine-tuned answers judged as escalating vs under-triaging have **identical mean length (82 vs 82 words)**; logistic regression of escalation on model + log(length) leaves the fine-tuned deficit intact (model coef −0.23). The judge is not penalizing terseness. (Added to §4.4.)
- **Second judge / IRR → PENDING.** Qwen2.5-7B independent judge is grading a stratified 200-answer subset; Cohen κ / Krippendorff α(Claude, Qwen) will be inserted (§3, §4, checklist). Author clinical spot-check already corroborates (§4.5).

## R2 — "Single model/corpus; team-authored n=100; is the keyword rubric a strawman?; don't imply base is safe."
- **Single model → SCOPED + PROPOSED.** Abstract/method scope the claim to "a LLaMA-3-8B medical assistant"; the title states the phenomenon. A second base model or PEFT method is **PROPOSED** (would need another fine-tune). Honest scoping added to §6; flagged in HONEST_VERDICT as a gating item.
- **External validity → PROPOSED.** A HealthBench emergency-referral subset replication is the right rebuttal and is listed (experiment_plan P5); access pending.
- **Strawman → SCOPED.** We reframe the critiqued metric as "a representative *lightweight/lexical* safety rubric, as used in this system's own evaluation and common in rapid LLM eval," not as the practice of all rigorous work. The contribution is that *fine-tuning makes such lexical proxies anti-valid* — a concrete, quantified instance, not a claim that no one uses LLM judges. (We cite HealthBench/MedHELM as the better practice we advocate.)
- **Don't imply base is safe → FIXED.** §4.4/§6 state both models triage poorly in absolute terms (base sensitivity 0.16 with no context is itself alarming); the contribution is the *measurement failure* and the *relative* degradation.

## R3 — "Novelty vs known judge>lexical & reward-hacking; metric-vs-model confound; stats hygiene."
- **Novelty positioning → SCOPED.** Related-work + discussion sharpen the kernel: not "LLM judge beats ROUGE" (known) but **a clinical-safety, fine-tuning-induced collapse of a safety metric's validity, quantified (κ 0.43→0.05; 68% mislabel)** — and that the collapse is *model-dependent*, hitting the deployed model. Positioned against G-Eval/MT-Bench and reward-hacking.
- **Metric-vs-model entanglement → FIXED (framing).** We state the two claims separately: (a) *metric invalid for FT* (held model fixed: keyword vs judge, Table 2); (b) *FT < base on the judge* (held metric fixed, Table 1). The κ number is used only for claim (a).
- **Holm correction → FIXED.** Applied to the McNemar family; noisy (p=0.013<0.0167) and adversarial (p=0.0018<0.0125) survive. (Added §4.4.)
- **Multi-seed → PROPOSED.** Single-sample CIs understate generation variance; multi-seed for the headline conditions is listed (experiment_plan P4).

## Area-Chair gating items
1. **Judge credibility in-paper:** IRR **PENDING** (Qwen, running); clinician subset **PROPOSED** (sheet prepared for Kevin). → On track; one needs Kevin.
2. **Scope generalization:** **SCOPED** now (case-study framing + honest §6); a second model remains the strongest **PROPOSED** upgrade.

## Net effect on the verdict
The fixable methodological critiques (κ prevalence, length confound, Holm, metric-vs-model separation, base-not-safe) are **resolved this session and strengthen the result**. The two gating items that remain are **(i) finishing IRR** (in progress) and **(ii) generalization/human-anchor** (needs Kevin / more compute). Per the meta-review, completing (i) and at least the human-anchor part of (ii) moves the paper from **Borderline → Weak Accept**.
