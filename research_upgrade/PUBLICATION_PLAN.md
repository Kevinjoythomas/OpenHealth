# PUBLICATION_PLAN.md — independent re-derivation (v2)
*2026-07-07, written from a first-principles review of the project state. Supersedes the earlier consolidation (which inherited the prior session's framing) and `MASTER_PLAN.md` (kept for queue-ops detail). Every number here was re-derived this session directly from the raw files by new code; provenance for each is in `PROGRESS.md` and `judge_analysis_results.md`. Where I agree with the earlier framing I say so explicitly — agreement after independent checking, not inheritance.*

---

## 1. Independent review of the project state

**What this project actually contains (verified by me, this session):**
- A local medical-QA system: LLaMA-3-8B-Instruct + QLoRA (r=8, α=16, 0.26% params; 1,000 steps on the 112,165-pair HealthCareMagic/ChatDoctor corpus, trained on a cloud T4), served on consumer hardware via Ollama; hybrid BM25+dense retrieval (RRF k=60, τ=0.020) over 1,823 chunks from 6 medical PDFs.
- **The core dataset:** a completed 2-model × 4-retrieval-condition × 100-case generation study (792/800 cells; 44 emergency cases), plus a **blinded 6-axis clinical re-grade of all answers** by one strict LLM judge (Claude, 790/792) and an independent lenient one (Qwen2.5-7B, 200-case subset), plus the original keyword-rubric scores and a real ROUGE study (mem/gen splits).
- A **methodological flaw I found in the original harness**: generation used `num_ctx=512` while injecting ~2.4–4k tokens of retrieved context → the clean/noisy/adversarial arms were truncated. Crucially, I verified the **`none` arm is untouched** (max prompt ≈124 tokens), so the central finding below does not depend on the flawed arms. A corrected, context-logged, seeded re-run is already executing.
- Unverifiable artifacts that must stay out of the paper: the training-loss curve (no log exists; a repo screenshot contradicts the draft's numbers) and the "trained locally on 8 GB" claim (training was cloud T4; the local GPU is 4 GB).

**The numbers everything rests on (re-derived, exact):**
| Finding | Value |
|---|---|
| ROUGE-1 / ROUGE-2 gain from SFT (unseen split) | +26.7% / +94.5% |
| Keyword-rubric "safety failure" on emergencies, base → FT (none) | 81% → 19% (looks like a big safety win) |
| Judged appropriate escalation, base → FT (none), exact 95% CI | 7/43 = 0.16 [0.07,0.31] → 12/43 = 0.28 [0.15,0.44] (no significant win; McNemar p=0.27) |
| Judged escalation under noisy/adversarial retrieval, base vs FT | base rises to 0.44/0.47; FT stuck at 0.19/0.19 (McNemar p=0.013/0.0018, Holm-surviving) |
| Keyword-vs-judge agreement (raw / PABAK), base → FT | 0.715 / +0.430 → 0.314 / −0.372 |
| P(keyword says safe ∧ judge says under-triage), FT | 0.68 |
| **Floor shift (new, mine):** mean safety-keyword count in *under-triaging* answers, base → FT | 0.56 → 1.50 (escalating answers: 2.25 → 2.41); point-biserial r drops 0.55 → 0.36 |
| Training-corpus register (M1, mine) | **79.5% of 112,165 human reference answers contain ≥1 safety keyword** |
| Judge agreement (Claude vs Qwen), escalation | κ=0.31 (fair); every *direction* replicates; absolute FT under-triage spans ~44–81% across judges |

## 2. The research contribution (my framing — revised, not inherited)

The inherited framing targeted "the keyword rubric." That invites the strawman objection ("nobody serious uses regex safety scoring"). My review says the true, defensible target is bigger and the paper already contains the evidence for it:

> **Central claim: fine-tuning on real doctor–patient dialogue improves every surface metric this literature actually uses — ROUGE (+26.7/+94.5%), lexical safety rubrics (81%→19% "failure"), fluent clinical register — while the one behavior that matters clinically, emergency escalation, does not improve and under noisy/adversarial retrieval significantly worsens relative to base. The surface-metric evaluation stack is not merely noisy: its validity is *model-dependent* and collapses on exactly the model one would deploy, because SFT distills the metric-correlated register from the corpus itself (79.5% keyword prevalence) into every answer — including the unsafe ones (keyword floor in unsafe answers: 0.56→1.50).**

This is documented practice, not a strawman: ChatDoctor itself (the corpus's source paper) evaluated with BERTScore against ChatGPT; this project's own original evaluation used ROUGE + a keyword rubric; that is the standard stack of the medical-dialogue-SFT genre. Our own "successful" ROUGE table becomes *evidence of the problem*, inside the paper.

**Mechanistic precision (new, mine):** the metric doesn't become "meaningless" — residual keyword–escalation correlation persists (r=+0.36). What breaks is the *operating point*: SFT lifts the keyword floor in unsafe answers ~3×, so any threshold rule that separated safe from unsafe answers for the base model stops separating them for the fine-tune. I will quantify this as a discriminability collapse (AUC of keyword-count → judged escalation, base vs FT — trivial CPU analysis, queued below). This is the exact, non-overclaimed statement, and it preempts the reviewer who would compute that correlation themselves.

**Alternative directions I considered and why I didn't switch** (per your instruction not to constrain to the existing path):
- **Corpus-audit-first** ("HealthCareMagic teaches under-triage") — genuinely novel, high-impact if true, and cheap for me to test at scale (judge 300–500 human reference answers on emergency presentations; no local GPU needed). **Adopted as co-headline candidate, gated**: if the scaled audit confirms that the *human* answers under-escalate, the mechanism section becomes a field-level claim (everyone fine-tuning on this corpus inherits the deficit); if not, the claim narrows honestly to "SFT under-learned triage." Pre-registered either way.
- **Metric-survival methods study** ("which automatic metrics survive SFT register shift") — adopted *inside* the main paper as the validity table (keyword / ROUGE / BERTScore / small judge / large judge / human anchor), not as a standalone.
- **Retrieval-robustness paper** (the repo's original plan) — falsified by its own data and further undermined by the truncation flaw; remains a corrected secondary result only.
- **Repair/interpolation study** (PEFT-Arena-style α-scaling) — not feasible without the unmerged adapter weights; noted as future work.
- A benchmark-only or position paper — too thin standalone; the 100-case probe + 3-judge protocol ships as the released artifact instead.

## 3. Weaknesses, gaps, and likely reviewer objections (each → disposition)

1. **"Your ground truth is an LLM judge."** Two independent judges agree on every direction but only fairly on threshold (κ=0.31). *Fix:* third judge (queued) → majority-of-3 primary labels + Krippendorff's α; strict/lenient bounds reported wherever they differ; **40-case blinded human anchor** (sheet ready; the one thing compute can't buy). Also: judges are cross-family (Anthropic/Qwen/Mistral) vs Llama subjects; all 1,000+ judgments released.
2. **"Truncation invalidates the study."** Self-caught, disclosed, and structurally contained: the core claim rests on the truncation-free `none` arm; retrieval claims are being re-established by the corrected, context-logged, 3-seed re-run (running). The truncation audit (T1) quantifies v1's exposure.
3. **"n=1 model pair."** Second independently-trained pair (medllama2 vs llama2) queued; if it doesn't replicate, the paper scopes to the recipe studied (pre-registered).
4. **"Team-authored 100 cases."** HealthBench emergency-referral subset (physician-authored, external) queued; contamination dedup of the 100 cases vs the 8k seen training rows (trivial CPU, this week).
5. **"Single decoding sample."** 3 seeds queued; inference will use case-level cluster bootstrap and McNemar on case-majority-over-seeds (cases are the unit, not generations — a stats-reviewer trap the plan now pre-empts).
6. **"κ is prevalence-distorted."** Already handled: lead with raw agreement, PABAK, and the conditional mislabel rate (all recomputed by me); κ reported for completeness.
7. **"The judge penalizes the FT model's terse style."** Refuted: escalating vs under-triaging FT answers have identical mean length (82 vs 82 words); logistic control keeps the deficit.
8. **"Strawman metric."** Dissolved by the reframe in §2 — the target is the documented evaluation stack of this literature, demonstrated on our own ROUGE table.
9. **"So the base model is safe?"** No — 0.16 escalation at none is itself alarming; the paper says so. The finding is the measurement failure plus the relative degradation, not base-model adequacy.
10. **"Did SFT at least preserve knowledge?"** MedQA/PubMedQA base-vs-FT queued, reported with parse-failure rate (format erosion is itself a capability result).
11. **"Any fix?"** One-line triage-instruction mitigation ablation queued; either outcome is actionable guidance.
12. **"Cherry-picked qualitative examples."** Example-selection criterion pre-stated (keyword-safe ∧ judge-unsafe), all judgments released.

## 4. Target venue fit (with one correction to the inherited plan)

**Correction:** the inherited plan implied ML4H → TMLR as a sequential fallback for the *same* paper. That is wrong: **ML4H Proceedings is archival PMLR**, and TMLR does not accept previously-published archival work — the ML4H and TMLR paths are **either/or** for identical content (a *substantially extended* journal version later is the legitimate route).

Decision tree (mine):
- **Human anchor (H1) secured by mid-August → ML4H 2026 Proceedings Track** (PMLR; 8 pp excl. refs; double-blind; reciprocal review; 2025 cadence ≈ Sep 8 submission / Dec symposium — confirm when the 2026 CFP posts). Best acceptance-probability-per-week; exact genre fit (medical-LLM safety evaluation, honest negative result, released artifact); archival and citable for Kevin's timeline.
- **If a real clinician panel materializes (≥2 raters, ~100+ ratings) → consider TMLR instead** (rolling journal; bar = supported claims + reader interest; precedent: *LoRA Learns Less and Forgets Less*) or **npj Digital Medicine** (higher bar, wants clinical validation). A true journal, at the cost of forgoing ML4H for this content.
- **Miss/reject → CHIL 2027** (~Feb; 8–10 pp PMLR; mandatory Data/Code-Availability, Contributions, IRB sections — all already drafted).
- JMLR remains ruled out (genre mismatch — theory/methodology journal).

My recommendation: **ML4H 2026**, because the marginal value of "journal" over "archival PMLR proceedings" does not justify months of delay and the either/or gamble; revisit only if the clinician panel actually happens.

## 5. Required experiments & analyses

**Running now (detached queue on this machine; log `research_upgrade/runs/queue.log`; halt = `runs/STOP`; resume-safe; ETA recalibrated after measuring real throughput — corpus chunks embed at ~20 s, generations ~20–60 s):**
| Job | Purpose | Est. |
|---|---|---|
| R0–R1 corpus embed + 300 logged retrievals | corrected, reproducible context pipeline | 6–10 h |
| T1 truncation audit (CPU) | quantify v1's exposure; honesty section | minutes |
| R2 grid ×3 seeds (2,400 gens, ctx 2048, contexts saved) | corrected core + variance | 20–40 h |
| G1 second pair: medllama2 vs llama2 (800 gens) | generalization | 9–16 h |
| B1 MedQA/PubMedQA (≈1,600 short gens) | capability preservation | 6–11 h |
| B2 HealthBench emergency subset (~160 gens) | external validity | 2–4 h |
| A1 prompt-mitigation (176 gens) | actionable fix test | 1.5–3 h |
| J1 local judges (Qwen full + Mistral 3rd) | panel + IRR | 10–15 h |
*(Queue file order now prioritizes G1/A1/B2/B1 over seeds 102–103 on any relaunch; the in-flight process retains its original order — harmless, revisit only if interrupted.)*

**Mine, no local GPU needed (this week):**
- **Scaled corpus audit (M2+):** extract 300–500 emergency-presentation reference pairs (60 done) and judge the *human* answers blinded, mixed indistinguishably with model answers so judges can't grade "humans" leniently. Decides the mechanism claim. (~Agent batches; my time ≈2 h.)
- **BERTScore** over all v1/v2 outputs (CPU, 2–4 h) — completes the surface-metric validity table (keyword / ROUGE / BERTScore / judges / human).
- **AUC discriminability analysis** (keyword-count → judged escalation, base vs FT; minutes) — the single number for the operating-point collapse.
- **Contamination dedup** (100 cases vs 8k seen rows; minutes).
- Claude-judge grading of every queue output batch as it lands; majority labels; α; full clustered stats; figures.

**Optional, justified, needs Kevin:**
| Item | Cost | Buys |
|---|---|---|
| H1 human anchor (40 blinded cases, sheet ready) | ~30 min human | converts judge-range into anchored estimate; biggest single acceptance lever |
| GPT-4o 4th judge | API key, ~$10, 1 h | cross-vendor IRR |
| Colab T4 logged re-fine-tune | ~3 h | same-recipe second subject + loss-curve provenance |
| ≥2-clinician panel | days, recruiting | unlocks TMLR/npj branch |

## 6. Validation strategy
Majority-of-3 cross-family judges = primary label; per-judge strict/lenient bounds always reported; Krippendorff's α + pairwise κ; human-anchored subset; external HealthBench cases carry their physician-authored rubrics; **pre-registered kill criteria** (non-replication → scope down; human sides with lenient judge → report anchored range; corrected grid contradicts v1 → corrected is primary; human reference answers escalate properly → drop corpus-transmission claim). Blinding throughout: judges never see model/condition; human sheet identical.

## 7. Ablations & controls
Context condition (core design, corrected) · seeds ×3 · judge strictness (3 judges) · **length control (done: 82 vs 82 words; logistic)** · **floor-shift/AUC (new)** · prompt-mitigation (A1) · metric-validity by model (done) · τ-sweep and chunk-budget demoted to appendix/PROPOSED (the gate is no longer a claim).

## 8. Baselines
Same-weights ± adapter (the causal contrast) · second SFT/base pair · literature anchors cited from verified sources (Med-PaLM 2 86.5% MedQA, MEDITRON-70B 70.2%, BioMistral ~50%) · the surface-metric stack itself as the baseline *instrument* · human reference answers as the (possibly low) "human ceiling."

## 9. Figures & tables (all from runs on disk or queued)
- **F1 (money figure):** left panel — surface metrics rise base→FT (ROUGE-1, keyword-safety rate); right panel — judged escalation falls/flat with exact CIs. One image = the paper.
- **F2:** keyword-count distributions in unsafe answers, base vs FT (the floor shift) + AUC collapse inset.
- **F3:** escalation grid (model × condition), corrected v2 data, majority labels, CIs.
- **F4:** mechanism — corpus keyword prevalence (79.5%) and judged escalation of human reference answers vs models'.
- **F5 (appendix):** truncation audit; per-category breakdowns; judge-agreement matrix; seed variance.
- **T1** judge-axis grid (mean ± cluster-bootstrap CI) · **T2** validity table (raw/PABAK/AUC per metric per model) · **T3** IRR (κ pairs, α) · **T4** pair-2 replication · **T5** MedQA/PubMedQA + parse-failure · **T6** mitigation · **T7** HealthBench · appendix: configs, digests, case census.

## 10. Writing structure (8-page PMLR)
1 Intro (the recipe, the stakes, the gap, contributions) · 2 Related work (PEFT retention; RAG robustness/safety; medical-LLM safety evals; judge reliability — 32 verified citations ready) · 3 Study design (system; grid; corrected pipeline + truncation disclosure; metric stack A vs panel B; human anchor; stats) · 4 Results (R1 surface-metrics-up → R2 escalation-not-up → R3 validity collapse + floor shift → R4 corrected robustness → R5 generalization → R6 mechanism → R7 capability/mitigation/external) · 5 Discussion (register distillation = Goodhart under evaluator shift; guidance for practice) · 6 Limitations · 7 Ethics/IRB/data · appendix: reproducibility checklist, all configs. Title candidate: *"Saying 'See a Doctor' Is Not Triage: Surface Metrics Mask Under-Escalation in Dialogue-Tuned Medical LLMs."*

## 11. Limitations (stated plainly)
LLM-judge ground truth with fair inter-judge threshold agreement (bounded, human-anchored, not pinned) · 100 team-authored cases + one external set · English, one corpus family + one replication pair · both models weak in absolute terms · adversarial arm narrower than poisoning threat models · v1 truncation (disclosed; core claim unaffected; corrected re-run) · no clinician co-author yet.

## 12. Ethics, privacy, safety
No PHI, no human subjects → explicit IRB-not-required statement. **Do not republish long verbatim patient texts from the scraped corpus** — paraphrase/truncate quotes (privacy + license caution; new safeguard, mine). Every model output shown is a labeled *failure example*, never advice; explicit not-for-clinical-use statement. Dual-use: disclosing an evaluation failure mode of our *own* released model is the safety contribution; nothing enables harm beyond trivially-known keyword-stuffing. Data provenance: HealthCareMagic-100k is public; we release judgments and case set, not re-hosted corpus text.

## 13. Reproducibility package
Seeded v2 harness with verbatim logged contexts · all model outputs + all judge scores + judge prompts · pinned environment (pip freeze) **and Ollama model digests** (e.g., llama3 365c0bd3c000, openhealth-doctor 44af3e3e112d) · the 100-case probe + human-anchor sheet · the v1 flaw documented as found-and-fixed. Loss curve and any unverifiable artifact excluded.

## 14. Prioritized roadmap
| When | What | Owner |
|---|---|---|
| Now → D+2 | queue grinds (R0→s101→pair2…); M2+ scaled audit judged; BERTScore; AUC; dedup | queue / me |
| **any 30 min now** | **H1 human sheet** | **Kevin** |
| D+2 → D+5 | queue completes; Claude-judge all new batches; majority labels + α; clustered stats; figures | me |
| D+5 → D+8 | full rewrite to §2's claim set; hostile re-review round 2; fix or scope | me |
| → mid-Aug | PMLR LaTeX, anonymization (names/HF/ngrok links out), checklist; freeze | me |
| ~Sep (per 2026 CFP) | **submit ML4H 2026** — or execute the TMLR/npj branch if a clinician panel materialized | Kevin + me |

**Definition of done:** corrected multi-seed grid with logged contexts; 3-judge majority labels + α + human-anchored subset; replication pair reported either way; mechanism section resolved by the scaled audit; capability/mitigation/external tables; anonymized 8-page PMLR paper that survives our own hostile re-review at weak-accept or better — with any shortfall named, not hidden.
