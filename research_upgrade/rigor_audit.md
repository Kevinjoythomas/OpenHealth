# rigor_audit.md — Experimental rigor audit (tough-reviewer lens)
*Phase 3. Audits the existing OpenHealth experiments against what an ML4H/CHIL reviewer will demand. Each gap states the exact additional test and why a reviewer requires it. Status tags: **OK** / **PARTIAL** / **MISSING**. Provenance: all current-state claims verified against repo files & my analysis (see `OpenHealth_Understanding.md` §4).*

## 0. Scope note
The paper's contribution is the **LLM study** (PEFT × retrieval-robustness × safety). The `models/` image/tabular classifiers (brain-tumor, lung, breast, diabetes) are a separate product surface and are **out of scope** — do not bolt their AUROCs onto this paper; if mentioned, one sentence in an appendix.

---

## 1. Baselines
**Current:** base `llama3` vs fine-tuned `openhealth-doctor`, each × {none, clean, noisy, adversarial}. ROUGE base-vs-FT. Retrieval: BM25 vs dense-MMR vs hybrid-RRF.
- **OK:** the most important internal baseline (same model ± LoRA) and the no-retrieval control are present — this is the right design for the core causal claim.
- **MISSING — external SOTA anchor.** No MedQA/MedMCQA/PubMedQA/MMLU number, so the paper can't be placed against the leaderboard reviewers anchor to (Med-PaLM 67.6%, Med-PaLM 2 86.5%, MEDITRON-70B 70.2%, BioMistral-7B ~50%). *Test:* run base + FT on a MedQA/PubMedQA **subset** via Ollama (secondary, capability-preservation framing per Kevin). *Why:* without it, "no capability loss" is asserted, not shown; reviewers will ask "did fine-tuning degrade medical knowledge?"
- **PARTIAL — robustness-trained baselines.** We test *plain* QLoRA but not noise-robustness methods (SA-RetRobust, RAAT) or selective-retrieval (Self-RAG/CRAG). *Test (optional/PROPOSED):* at minimum, discuss; ideally compare the τ-gate against a no-gate and a Self-RAG-style abstention. *Why:* our τ=0.020 gate is sold as a fix; reviewers want it compared to the obvious alternative.

## 2. Ablations (does each claimed contribution have an isolating ablation?)
- **PEFT effect:** **OK** — base vs FT isolates it.
- **Retrieval-quality effect:** **OK** — the 4 conditions isolate clean/noisy/adversarial.
- **Relevance gate τ=0.020:** **MISSING ablation.** τ is presented as a contribution but never swept. *Test:* τ ∈ {0, 0.01, 0.02, 0.05, 0.1} × {clean,noisy,adversarial}, report safety/quality vs τ. *Why:* a single hand-picked threshold with no sweep is the textbook "magic constant" reviewers reject; the sweep also shows the gate actually does something (current data shows retrieval barely hurts, so the gate's value must be demonstrated, not assumed).
- **Hybrid-RRF vs components:** **PARTIAL** — latency compared, but no *retrieval-quality* ablation (recall@k / nDCG of BM25 vs dense vs hybrid on a labeled query set). *Test:* annotate gold passages for the 100 cases; report recall@5/nDCG. *Why:* "hybrid surfaces cross-domain associations" is currently anecdotal (3 example queries).

## 3. Statistical validity
**Current:** single decoding pass per cell (temp 0.3); the keyword-rubric means/rates; no CIs/seeds in the docx; `analyze_research.py` (now run) gives Mann-Whitney but on the keyword metric.
- **MISSING — multiple seeds / sampling variance.** One generation per (case,model,condition). *Test:* ≥3 samples (or seeds) per cell for the headline conditions; report mean ± CI. *Why:* single-sample LLM outputs at temp>0 are not reproducible point estimates; reviewers will not accept single-run deltas as robust.
- **PARTIAL → in progress — CIs & significance.** *Fix (Phase 4):* 95% bootstrap CIs on every reported rate; **McNemar** for paired base-vs-FT safety; **Wilcoxon/Mann-Whitney** for robustness drops; **Holm–Bonferroni** across the family of model×condition comparisons. *Why:* the paper makes ~8+ comparisons; uncorrected p-values invite a "fishing" critique. (Already computed on the keyword metric: robustness effects n.s., p=0.21/0.069; the safety effect is large, McNemar p≈1e-7. Re-run on the judge metric.)
- **MISSING — inter-rater reliability for the metric itself.** *Fix (Phase 4):* multi-judge (Claude + ≥1 independent model) with **Cohen κ / Krippendorff α**. *Why:* a one-judge automatic score has no validity argument; IRR + a human-validation subset is the credibility backbone (Kevin's directive).

## 4. Metrics (correct for the task?)
- **MAJOR — primary metric is a keyword heuristic.** `score_response()` sets safety=2 on any of {consult,doctor,call,…}. This is not triage; it manufactures the noisy-context paradox (see Understanding §4.1). **This is the #1 reviewer kill-shot.** *Fix (Phase 4):* multi-judge clinical rubric (correctness/triage/harm/calibration) + the binary **appropriate_escalation**.
- **OPPORTUNITY — frame safety as a detection problem.** `appropriate_escalation` on `safety_required` cases is a binary classification → report **sensitivity** (true emergencies correctly escalated), **specificity** (non-emergencies not over-escalated, from the 56 non-safety cases), **F1**, and a **calibration** view (does the model's hedging track correctness?). *Why:* reviewers expect sensitivity/specificity for a safety-triage claim, not a mean rubric score; under-triage (false negatives) is the clinically decisive error and must be reported separately from over-triage.
- **PARTIAL — ROUGE.** Real but weak for clinical correctness; **demote** to a secondary table, keep the memorisation/generalisation split (that *is* a nice anti-overfitting design).

## 5. Subgroup / fairness analysis
- **MISSING.** No breakdown beyond the 4 case categories. *Test:* report the safety result **by category** (safety_emergency / general_covered / corpus_gap / edge_case) — already in the data — and add at least a **specialty/clinical-domain** breakdown (cardiac vs neuro vs paeds vs endocrine; the cases already span these). If feasible, vary patient demographic framing (age/sex) in the case text and test for differential triage. *Why:* health-ML reviewers treat subgroup robustness as near-mandatory; differential under-triage across presentations is a publishable safety finding in itself.

## 6. Robustness / external validity
- **OK (core)** — the noisy/adversarial conditions *are* a robustness study; that's the paper's spine.
- **MISSING — external/held-out validation.** All 100 cases are team-authored. *Test:* validate the safety result on an **external** set — a HealthBench emergency-referral subset, or iCliniq cases — even a small one. *Why:* "team authored its own benchmark" is a standard reject reason; one external corpus rebuts it.
- **PARTIAL — adversarial construction.** "Adversarial" = top-3 chunks for a surface-similar wrong query (no threshold). Reasonable, but (a) not validated as actually misleading, and (b) **the injected context text was NOT saved** (only `n_chunks`). *Test:* re-run saving the exact injected context; have a judge confirm each adversarial context is genuinely contradictory. *Why:* reviewers will ask "how adversarial was it, really?" and reproducibility requires the actual inputs.

## 7. Data integrity
- **PARTIAL — leakage/splits.** ROUGE uses a memorisation (rows 0–7,999, seen) vs generalisation (8,000+, unseen) split — **good**. But the 100 safety cases are "drawn from ChatDoctor" — confirm none overlap the 8,000 training rows. *Test:* dedup the 100 case questions against the training slice. *Why:* train/test contamination on the headline safety set would be fatal.
- **OK — class balance disclosed** (25/35/20/20 categories; 44 safety_required). Report it.
- **PARTIAL — dataset documentation.** ChatDoctor/HealthCareMagic-100k is public (state license/source); the 6-PDF corpus provenance is thin ("clinical review"/"ACC/AHA guideline") — document each source properly.

## 8. Reproducibility
- **MISSING — training-loss provenance.** The docx loss curve (3.74→2.19, step-413 spike) has **no log in the repo**; training was a cloud-T4 (Colab) run whose log wasn't saved. *Fix:* re-run training on a logged T4 (save `trainer_state.json`/W&B), **or** remove/mark the curve illustrative. *Why:* an unreproducible figure presented as a result violates basic reproducibility and our ground rules.
- **MISSING — hardware honesty.** Docx says local RTX 3050 8 GB for everything; reality = **trained on cloud T4**, **served on a 4 GB local GPU**. *Fix:* state both accurately.
- **PARTIAL — config.** LoRA hyperparams documented; **missing:** seeds, decoding params per run (temp 0.3, num_predict 300, num_ctx 512 — found in code, must go in the paper), retrieval params (top_k, chunk size/overlap), judge model versions/prompts. *Fix:* a full config + release the seeded harness, the 100-case set, the 792 outputs, and judge prompts (anonymized for double-blind).
- **OK (asset):** the harness (`research_experiment.py`) and analysis (`analyze_research.py`) are runnable end-to-end and crash-safe — a genuine reproducibility strength once cleaned and seeded.

---

## 9. Priority-ordered fix list (what actually moves the needle)
1. **Replace the keyword metric** with the multi-judge clinical rubric + IRR. *(Phase 4, in progress — without this nothing else matters.)*
2. **Report safety as sensitivity/specificity + CIs + McNemar/Holm**, by category.
3. **τ sweep** ablation (kills the "magic constant" critique and justifies the gate).
4. **One external validation set** (HealthBench emergency subset / iCliniq) for the safety claim.
5. **Multiple seeds/samples** on the headline conditions.
6. **Honesty fixes:** training-on-T4 vs local-serving, 4 GB GPU, training-loss log (re-run or mark illustrative), leakage dedup, save injected contexts.
7. **Secondary MedQA/PubMedQA** capability-preservation check.
8. Retrieval-quality ablation (recall@k/nDCG) + τ-gate vs Self-RAG/CRAG discussion.

Items 1–2 are mandatory for any shot; 3–6 are what turn a borderline into a defensible accept; 7–8 are reviewer-pleasers.
