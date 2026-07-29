# PROGRESS.md — OpenHealth Research Upgrade

Running log. Newest entries at top. All times are session-relative; "today" = 2026-06-23.

---

## STATUS 2026-07-12 (cont.) — meditron retried (unusable), VERIFICATION 8/0, PMLR LaTeX done
Kevin's 3 asks all done: (1) **meditron:7b retried** with tuned params (num_predict=200) after clearing the wedged run — confirmed UNUSABLE: as a non-instruction-tuned base model it emits off-task garbage ("You are a chatbot... weather, traffic") not medical answers. Documented honestly in §4.11 (note preempts "why not more models"); 4-model panel stands. (2) **Final verification review** (workflow wf_d47e1c96-ac9, 8 adversarial verifiers + AC synthesis): **8 CONFIRMED, 0 ISSUES** — M1 conditional 75/40 (joint 68/14 correctly labeled), M2 TOST framing, M6 chance-0.29, seed ranges, IRR κ=0.30/majority 43-49, Table 3 panel, M3/M4/M5 framing, no-fabrication sweep — every number recomputed from artifacts and confirmed; paper internally sound + fabrication-free. (3) **PMLR LaTeX**: wrote `md_to_latex.py` → `OpenHealth_Research_Paper_vNext.tex` (pandoc-free; unicode→LaTeX, tables→booktabs, main-body/appendix split §4.4/4.5/4.7/4.9/4.12, 2 figures, 6 new bibitems). Structurally sound: balanced envs (3 tables, 2 figs, bib), 346/346 braces, 0 stray unicode. Manual at submission: swap official ML4H/jmlr style, full .bib, move table captions into \caption{}, straight→curly quotes.
**Paper is DONE — content-complete, verified 8/0, anonymized, LaTeX-drafted. Only Kevin items left: clinician anchor + final LaTeX polish.**

## STATUS 2026-07-12 (cont.) — SEED VARIANCE (3 seeds) + PANEL (Table 3) done; paper content-COMPLETE
Claude limit reset; resumed. **Seed variance §4.1:** judged s102 (768/0) + s103 (754, 1 batch dropped on conn error). All 3 seeds stable — base esc 24/24/26%, ft esc 24/20.5/24%, ft kw-safe 90/91/88%, ft κ 0.03/0.05/0.04 vs base κ 0.61/0.68/0.64. Folded into §4.1; §6/§8/provenance updated (seeds done). **Panel §4.11 Table 3:** `panel_compare.py` on already-judged data (no meditron needed) → diagnostic (κ<0.2 & kw-safe>0.6 & PPV-fail>0.5) flags ONLY the ChatDoctor ft; medllama2 high-kw but earned. Reframes contribution as a *validated diagnostic across a panel* (answers AC's n=1 concern). **meditron:** wedged (base model spills CPU on 4GB GPU, 0 cells/2.5hr), left running per Kevin; excluded from panel (4-model panel carries it). `REPRODUCIBILITY.md` drafted (artifact↔claim map). Goodhart citations verified real. **Paper now content-complete; only outstanding item = clinician anchor (Kevin).** Remaining: 8pp trim/appendix-split, anonymization (4 mentions), add citations to bib, HONEST_VERDICT.

## STATUS 2026-07-12 (cont.) — SESSION LIMIT + IRR §4.6 completed Claude-FREE
Hit Claude session limit mid-s102-judging (resets ~6:50am Asia/Calcutta); 15/55 s102 batches failed. Partial s102 saved (`wfjudge_clean_s102_partial.json`, 560/768). **Rerouted: finished the multi-judge IRR (§4.6) WITHOUT Claude** — `judge_local.py clean_s101 --emergency --cond=none` (Qwen2.5+Mistral, 144 no-retrieval emergency cells, 86.9min local) → `majority_irr.py` combines with the already-done Claude judgments. Result: Fleiss κ=0.30 (fair); pairwise Cohen κ 0.29–0.39; **majority-of-3 base 43% vs ft 49% escalation = judge-robust equivalence** (vs Claude 29%/24%); absolute rate judge-dependent (open judges more lenient, ~46% vs Claude ~24%) → reinforces clinician-anchor framing, contrast is threshold-invariant. Folded into §4.6 + provenance note. `majority_clean_s101.json`.
Queue now (local, Claude-free): meditron panel generation (P1_panel_meditron) → J_local_pair2. DEFERRED to Claude reset: finish s102 (208 cells) + judge s103 → seed-variance §4.1; judge meditron → panel §4.11.

## STATUS 2026-07-12 (cont.) — HOSTILE RE-REVIEW v3 (5 lenses) + all majors fixed
Ran a 5-lens adversarial review workflow (wf_30dae5ef-c79): clinical/stats/novelty/honesty/repro, each reproduced the numbers from artifacts and flagged only genuine defects. Mean 3.2, Major Revision, 18 real issues. **All 6 majors fixed from existing data + re-verified** (see reviews_v3_and_fixes.md):
- M1: flagship "68% of keyword-safe answers under-triage" conflated JOINT (0.677) with the CONDITIONAL its prose asserts; true PPV-failure is 75%(ft)/40%(base). Fixed everywhere (abstract/§1/§4.2/4.3/4.7); base was understated ~3×.
- M2: TOST "positive equivalence" rests on case-averaged CI; the independent no-retrieval test FAILS equivalence (CI +16.8%). §4.1 rewritten to rest on the case-level test + disclose the failing one; McNemar reframed as underpowered null.
- M3: added §2 Goodhart/reward-hacking/shortcut-learning paragraph + explicit delta (incidental, not adversarial, gaming).
- M4: dropped unqualified "systemic"; reframed generalizable lesson (lexical scores invalidated by register shift) + corpus-specific catastrophe.
- M5: incumbent rubric reframed as this system's own eval scheme / metric-family point, not asserted field practice.
- M6: fixed "below 0.50 expected by chance" → 0.31 is AT the 0.29 chance level at the ft's marginals.
- Minors fixed: §4.4 low-to-moderate, §4.5 condition label + 158/288, §4.8 removed unreproducible means + softened "decisively", §4.11 base condition-sensitivity + confounds, §5 base-validity benchmark-specific (HB 0.44), abstract leads with 55-pt gap.
⚠ TODO: verify+add the new Goodhart citations to bib; trim to 8pp (now 6111 words); return clinician anchor (Kevin).

## STATUS 2026-07-11 (cont.) — EXTERNAL VALIDITY (HealthBench) → §4.12: over-crediting REPLICATES externally
B2 HealthBench 80 physician-authored emergency-referral cases × 2 models = 160 gens (63.8min), converted via `healthbench_prep.py` → blinded → Claude-judged 160/0-missing (wf_059dc147-759) → `judged_healthbench.json` → `healthbench_analysis.json`. **Over-crediting replicates on external data:** FT keyword-safe 66% vs base 28% (2.4×), yet FT appropriate-response rate 59% vs base 69% (NO gain; mean triage 1.39 vs 1.55); FT 4.3× more often keyword-safe-yet-clinically-inappropriate (32.5% vs 7.5%). Judge validated as using clinical reasoning not the safety_required flag (explicitly noted "labeled safety flag does not reflect real emergency risk" on benign cases). Wrote §4.12; updated §6 + provenance. `analysis/healthbench_analysis.py` reproduces it.

## STATUS 2026-07-11 (cont.) — ORIG (corpus-adjacent, corrected 2048-ctx) judged → §4.7 rewritten
`grid_orig_s101` 800/800 gen (4.2hr), Claude-judged 800/0-missing (wf_03d90dbf-a1b → `judged_orig_s101.json` → `analysis_orig_s101.json`). **Collapse REPLICATES** on the corpus-adjacent set: keyword↔judge raw 0.82→0.35, κ 0.50→0.06, AUROC 0.83→0.67, 64% of FT "safe" answers under-triage (vs 13% base); escalation base 16–21% vs FT 21–30%, McNemar n.s. all 4 conditions. Spot-checked genuine (FT gives first-aid/side-effect info but no ER referral, "Chat Doctor"-branded). **HONEST CORRECTION:** the v1 "retrieval helps the base model on the corpus-adjacent set" claim was a **512-token truncation artifact** — the corrected 2048-ctx run shows retrieval helps *neither* model (base escalation highest with NO retrieval). Fixed in §4.7 AND §2 (RAG related-work). §4.7 rewritten with corrected numbers; provenance note updated (only §4.6 IRR now in progress).

## STATUS 2026-07-11 (cont.) — GENERALIZATION pair judged: honest NEGATIVE (scopes the claim)
`grid_pair2_clean_s101` (llama2 vs medllama2) generated 767/768, Claude-judged 767/0-missing (workflow `wf_f4f4aa14-7d2`, journal.jsonl → `wfjudge_pair2_clean_s101.json` → `judged_pair2_clean_s101.json` → `analysis_pair2_clean_s101.json`). **The collapse does NOT replicate — and that confirms the mechanism.** medllama2 *improves* triage over llama2 (escalation 21%→55% pooled; 57% vs 6% no-retrieval; specificity 0.91–0.96, stable 54–57% across conditions), and its keyword rubric degrades only mildly (κ 0.45→0.35, raw 0.75→0.70, AUROC 0.87→0.81) vs the primary κ 0.61→0.03. Spot-checked genuine (medllama2 says "go to the emergency department", names PE/meningitis; llama2 deflects with clarifying questions). **→ failure is corpus-specific (ChatDoctor register), not intrinsic to PEFT.** Wrote **§4.11**, updated §5/§6/contributions/provenance; `analysis/generalization_compare.py` → `generalization_compare.json` reproduces it. Net acceptance effect: neutral-to-positive (removes "single-model case study" risk, confirms mechanism). Paper now ~5030 words, 20 subsections.

## STATUS 2026-07-11 (48h autonomous run + full reproducibility audit, Opus)
**Machine freed; generation working again.** Robust unattended run armed: `harness_v2/watchdog.py` (relaunches the resume-safe queue if dead; honors runs/STOP + runs/ALL_DONE) + a **Windows Scheduled Task `OpenHealthWatchdog`** running `watchdog.py --once` every 5 min as the backbone. Queue (`run_queue.py`) now writes `runs/ALL_DONE` only when every critical (non-external) grid is `.done`, and JOBS extended with `J_local_{clean,pair2,orig}` (Qwen+Mistral IRR judging) after generation.
- **In flight:** generalization pair `grid_pair2_clean_s101` (llama2 base vs medllama2 medical-FT) generating (~llama2 done, medllama2 ~260/384, slow on 4GB GPU); then `grid_orig_s101` (corrected 2048-ctx), seeds 102/103, and externals B2 HealthBench (60MB raw pre-cached → skips download) / B1 MCQ (`datasets` lib now installed for MedQA; PubMedQA direct-URL OK; N_PER=400).
- **Pipelines verified ready** for pair2: `judge_grid_prep.py` → `grid_judge.wf.js` (returns merged `{scores}`) → `grid_aggregate.py` → `full_analysis.py` (base/ft detection: llama2=base, medllama2=ft ✓). IRR: `judge_local.py` → `analysis/majority_irr.py` (blindmap/judged/local keys align ✓).

**FULL REPRODUCIBILITY AUDIT of every headline + in-text number (all trace to files):**
- Abstract headline **all verified**: 24.0%/24.0% escalation = 69/288 each (analysis_clean_s101); 6.5× = 65%/10% fail (=35%/90% safe, pooled); κ 0.605→0.026; AUROC 0.912→0.668; PABAK 0.667→−0.382; 68% kw-safe-under-triage (0.677); floor 0.21→1.57; corpus 12.6%→"13%" on 167 emergencies (corpus_audit_result); corpus 79.5% keyword saturation (corpus_register); mitigation base 139/144=97% / ft 71/144=49% WITH triage line (judged_mit_clean). **33/72=45.8%** serious-under-triage-yet-kw-safe (exact).
- **TOST**: paper's 90% CI [−6.2,+6.2] confirmed = the **case-level** test (72 independent units, avg over conditions) → [−6.25,+6.25]; McNemar p-values (0.54/0.40/0.66/0.84) exact. §4.1 correct, no change.
- **NEW script `analysis/robustness_stats.py`** regenerates TOST + stricter-rubric + length-control from raw judgments (closes a provenance gap — these were ad-hoc before).
- **§4.2 stricter-rubric REWRITTEN** (was ad-hoc "57 vs 31", did not reproduce): honest reproducible finding — the 6.5× illusion is *specifically the referral register*; a genuine-emergency-token rubric largely dissolves it (ft "safe" 90%→35%, base 27%; agreement stays 0.89→0.81 for both; ft strict-safe-under-triage 15% vs 68% incumbent). Residual ~1.3× → lexical scoring "mitigated not cured"; validates the urgency-sensitive fix. Sharper & more honest than before.
- **§4.6 length** updated to reproducible FT-only: escalating median 67 vs under-triage 75 words (escalating *shorter* → judge not rewarding length); logit FT coef β=−0.30 (p=0.44, n.s.).
- **§4.8 mechanism strengthened** with the corpus per-keyword breakdown (*doctor* 69.3% vs *911* 0.0%, *urgent* 0.6%) — proves the title from corpus statistics.
- reviews_v2_and_fixes.md strawman row updated to the corrected finding.

## STATUS: FULL IMPLEMENTATION (2026-07-07, Opus). Corrected-grid queue RUNNING detached; corpus-audit + clean-benchmark DONE; reusable grid-judge pipeline ready.

## ⛔ BLOCKER (2026-07-07 ~13:00): local generation impossible — HARDWARE RESOURCE WALL
- **RAM = 7.68 GB total, ~0.36 GB free**; GPU = 4 GB with a leaked CUDA pinned-memory mapping ("resource already mapped"). llama3/openhealth-doctor need ~4.9 GB → fit in NEITHER. CPU-mode fails: "unable to allocate CPU buffer of size 4.9 GB". Ollama error captured in `runs/ollama_cpu.log`.
- **Root cause:** this session's many model load/unload cycles + concurrent apps (VSCode 1.2GB, Edge, AnyDesk, Defender) exhausted the 7.68 GB RAM and leaked the GPU context. It ran earlier this session before exhaustion.
- **What still works (cloud, not local):** all Claude-based judging + all analysis. **What's blocked:** new local generation (v2 grid, clean-benchmark eval, pair-2, HealthBench/MedQA gen) AND local-model judging (Qwen/Mistral IRR — those 4.7GB models also can't load).
- **Unblock options:** (A) **reboot** + keep apps closed, then relaunch queue (resume-safe) — borderline feasible on 7.68GB; (B) **run generation on a cloud T4** (like training) via a prepared notebook — most reliable for the multi-grid plan; (C) proceed with the paper on the EXISTING v1 judged data (792 gens, already done) + the corpus-audit mechanism, treating v2/clean-benchmark eval as a strengthening pass to run once unblocked.
- Queue is STOPPED (was fail-looping on generate). All harness code is ready; contexts_cache_clean/orig.json + clean_benchmark.json are computed. Only GENERATION is blocked.

## HOW TO RESUME (exact steps for the next session)
1. **Check the queue is alive:** `tail research_upgrade/runs/queue.log`; confirm a `python.exe run_queue.py` process exists (WMIC/Get-Process). If dead, relaunch: `cd research_upgrade/harness_v2 && python launch_queue.py` (resume-safe; .done jobs skipped). If a job STALLS (no output ~10min), the embed/gen fix is `keep_alive`+batch `/api/embed` (already in common.py); a hung generate recovers via the 240s socket timeout.
2. **Ollama fragility:** ONE daemon only; if wedged, kill all `ollama.exe`, start ONE detached, models run one-at-a-time on the 4GB GPU. The BATCH `/api/embed` endpoint is reliable; the singular `/api/embeddings` intermittently hangs — do not use it.
3. **When a grid finishes** (monitor emits `DONE R2_grid_clean_s101` etc.), judge it:
   - `python harness_v2/judge_grid_prep.py results_v2/grid_clean_s101.json clean_s101`  (note printed N_BATCHES)
   - `Workflow({scriptPath:'.../harness_v2/grid_judge.wf.js', args:{tag:'clean_s101', nBatches:N}})`
   - `python analysis/grid_aggregate.py <wf_output_file> clean_s101`
   Repeat for grid_orig_s101, grid_pair2_clean_s101, mitigation_clean, grid_clean_s102/s103.
4. **HealthBench/MCQ (B2/B1)** outputs: results_v2/healthbench_runs.json, mcq_results.json — judge HealthBench answers via the same grid pipeline; MCQ is self-scored.
5. **Then:** local judges (Qwen+Mistral via judge_local.py — update it to the new grid file names) for IRR → majority-of-3 + Krippendorff α → figures/tables → rewrite paper → hostile re-review → anonymize + PMLR LaTeX.

## Design note (for the paper's limitations):
- The retrieval corpus is narrow (~21 sources, 40% chest-pain) and cannot supply *relevant* clean context for most of the 96 diverse clean-benchmark emergencies. So clean/noisy/adversarial conditions are meaningful mainly for cardiac/respiratory/neck/headache cases; for the rest, injected context is off-topic regardless. **Therefore the `none` condition (no retrieval) is the PRIMARY, confound-free result on the clean benchmark** (does FT under-triage NOVEL emergencies?); the retrieval conditions are exploratory robustness with a disclosed corpus-coverage caveat. The ORIGINAL 100 cases (designed around the corpus, with curated noisy/adversarial queries) carry the retrieval-robustness axis. Context tokens now fit (median 402, max 537 < 2048) — v1 truncation fixed.

## Confirmed v2 findings (do not re-derive):
- corpus audit: HUMAN refs escalate 13% of 167 true emergencies (mechanism). corpus_audit_result.json.
- contamination: ~2-3 cases (NOT 41% — that was a metric artifact). contamination_v2_original.json.
- clean benchmark: 96 novel cases (72 emerg), contamination-free. clean_benchmark.json.
- floor shift: keyword AUC 0.81→0.72; floor 0.56→1.50. paper_stats.json.
- v1 Claude-judge (contaminated original): FT escalation 0.19-0.28 vs base 0.16-0.47; keyword raw-agree 0.72→0.31.

## (earlier) STATUS: executing PUBLICATION_PLAN v2.

## 2026-07-07 (implementation sprint) — findings & fixes
**Contamination: initially over-estimated, then CORRECTED.** A scattered-block containment metric flagged 41/100 as contaminated, BUT rigorous re-check with the **longest contiguous shared run** (real copies share one long run; coincidental medical phrasing shares many short ones) shows the TRUE number is **only ~2-3 cases** (G1: 127-char verbatim run vs row0; S3: 92 chars vs row149; S4: medication-list paraphrase the metric under-counts). The 41% was a **false-positive artifact** — verified by inspection (e.g. a flagged "match" paired an authored suicidality vignette with a corpus menopause-depression question). **The benchmark is ~clean; the headline is unchanged whether or not the 2-3 cases are excluded.** Honest correction logged. (`analysis/contamination_v2.py`, `results_v2/contamination_v2_original.json`.) Still a minor methods note (adapting cases from the training corpus is a trap); not a headline.

**MECHANISM CO-HEADLINE CONFIRMED (corpus audit, 510 human reference answers, blinded Claude judge):** among **167 judge-confirmed TRUE emergencies** in HealthCareMagic, the **HUMAN doctor answers escalate only 13%** (21/167) [CI .08-.19], mean correctness 1.33/3, harm 1.25/2. So the training corpus itself under-escalates → the fine-tuned model (19-28%) faithfully inherited a ~13% escalation prior; the base model (16-47%) escalates more precisely because it was NOT corpus-trained. **Every model fine-tuned on this corpus inherits the deficit** — a field-level finding. (`analysis/corpus_audit_aggregate.py`, `results_v2/corpus_audit_result.json`; self-selection caveat noted honestly.)

**CLEAN BENCHMARK BUILT (contamination-free, properly powered):** 96 novel vignettes authored by 12 specialty-physician agents + independent QA (all 96 verified), containment-confirmed clean (max shared run 35 chars). 72 emergencies + 24 matched-benign, 8/specialty × 12 (cardiac…vascular_trauma). `results_v2/clean_benchmark.json`. This is the paper's PRIMARY benchmark; the original 100 is kept for v1 comparison.

**Discriminability/floor-shift** (`analysis/paper_stats.py`): keyword-count→escalation AUC 0.81 base → 0.72 FT; keyword floor in under-triaging answers 0.56 → 1.50. The rubric's ≥1-keyword operating threshold sits below FT's floor → can't separate safe/unsafe though count retains partial signal.

**Discriminability/floor-shift quantified** (`analysis/paper_stats.py`): keyword-count→escalation AUC = **0.81 base → 0.72 FT**; keyword floor in UNDER-TRIAGING answers = **0.56 → 1.50** (escalating 2.25→2.41). The rubric's operating threshold (≥1 keyword ⇒ "safe") sits below FT's floor → can't separate safe from unsafe, though count retains partial signal. Precise, non-overclaimed.

**Infra root-cause + fix:** the singular Ollama `/api/embeddings` endpoint intermittently WEDGES on this 4GB box (hung connections that don't hit socket timeout → corpus_build stalled twice at 100/200). The **BATCH `/api/embed` endpoint is reliable** → all embedding switched to it (common.batch_embed); corpus (1823 chunks) embedded in ~3 min. Also added keep_alive="30m" + shorter socket timeouts + retries to generate/embed. Detached breakaway launcher survives session teardown (the earlier "deaths" were my own kill commands).

**Launched (running):** corpus-audit judging (`w4rzyy97c`, 510 human reference answers, blinded: is_emergency + escalated) — the mechanism co-headline; clean-benchmark authoring (`whqz3o78r`, 96 novel vignettes across 12 specialties + independent QA verify). Retrieval reimplementation (harness_v2/retrieve.py) validated end-to-end on the fresh corpus vectors.

## (earlier today) PUBLICATION_PLAN v2 (independent re-derivation)

## 2026-07-07 (later) — Independent re-derivation (Kevin asked for fresh thinking, not inherited framing)
- **Re-derived every load-bearing number from raw files with new code** — all confirmed (ROUGE +26.7/+94.5%; esc 7/43→12/43 @none with exact CIs; agreement 0.715→0.314; PABAK +0.430→−0.372; mislabel 0.68).
- **New analysis (floor shift):** mean safety-keyword count in UNDER-TRIAGING answers rises 0.56→1.50 base→FT (escalating: 2.25→2.41); keyword–escalation r drops 0.55→0.36. The metric isn't "meaningless" post-SFT — its *operating point* is destroyed. AUC discriminability analysis queued as the headline quantification.
- **Verified structurally:** `none`-arm prompts ≤~124 tokens → the core claim is immune to the num_ctx truncation flaw.
- **PUBLICATION_PLAN.md rewritten (v2) with real framing changes:** (1) target reframed from "keyword rubric" to the *surface-metric evaluation stack of the medical-dialogue-SFT literature* (ROUGE/BERTScore/keyword — ChatDoctor itself used BERTScore; our own ROUGE table becomes in-paper evidence) — kills the strawman objection; (2) floor-shift precision instead of binary "anti-valid"; (3) corpus-audit (M2 scaled to 300–500 judged human reference answers) elevated to *gated co-headline*; (4) **correction to inherited venue plan: ML4H Proceedings is archival PMLR → ML4H-then-TMLR for the same content is NOT possible** — either/or, decision tree conditioned on human anchor / clinician panel; (5) clustered inference pre-specified (case-level bootstrap, case-majority McNemar, exact binomial CIs); (6) ethics addition: no long verbatim patient-text quotes (privacy/license); (7) money-figure redesign ("surface metrics up, escalation down").
- Queue left running untouched (Kevin declined the restart); reordered JOBS applies only on any future relaunch. R0 at ~100/1823 (~20s/chunk on real 800-tok chunks → R0 ≈6–10h; total ≈3–4.5 days).
- **Next (me, no GPU needed):** M2+ scaled audit extraction + judging, BERTScore table, AUC analysis, contamination dedup, then judge queue outputs as they land. **Next (Kevin): H1 human sheet (~30 min).**

## 2026-07-07 — Journal-grade escalation (Kevin: "no compute constraints, final hope")
- **Wrote `PUBLICATION_PLAN.md`** (supersedes MASTER_PLAN): contribution, 14 reviewer objections each with fix+status, venue ranking (ML4H 2026 → TMLR → CHIL 2027), validation strategy, ablations, baselines, figures/tables plan, writing skeleton, ethics, roadmap, compute budget.
- **Found & owned a methodological flaw:** v1 ran with num_ctx=512 vs ~2.4–4k tokens of injected context → retrieval arms truncated. v2 pipeline fixes it (3×400-token chunks, ctx 2048, contexts LOGGED verbatim, seeds); truncation audit T1 quantifies v1 damage.
- **Mechanism early result (M1, real):** **79.5% of the 112,165 human reference answers contain a safety keyword** — the corpus embeds the register the keyword metric rewards. M2 extracted 60 emergency reference pairs for the judge panel ("do the humans under-escalate too?").
- **Harness v2 built** (`research_upgrade/harness_v2/`): standalone hybrid retriever (sqlite-extracted corpus, fresh nomic embeddings, MMR+BM25+RRF k=60), corrected grid generator (model-major, resume-safe, contexts saved), MCQ bench (MedQA/PubMedQA, fail-soft), HealthBench emergency-subset runner, prompt-mitigation ablation, local 2-judge grader, resume-safe fail-soft queue runner + detached launcher (survives session death; auto-restarts Ollama).
- **Queue launched** (13→14 jobs): R0 embed → R1 contexts → T1 → R2 grid ×3 seeds → pulls → G1 pair-2 (llama2/medllama2) → B1 MCQ → B2 HealthBench → A1 mitigation → J1 judges. ≈2–3.5 days.
- Loss-curve provenance CLOSED: repo screenshot contradicts the docx curve; both unverifiable → excluded from paper.
- Warm perf on this 4GB box: embed ~2.1s, gen ~20–60s. Ollama daemon must be started detached (session-tied daemons die with the session — root cause of earlier 500s).

---
## (historical) STATUS: all phases delivered; IRR κ=0.31 fair.

**IRR result (2026-06-25):** Qwen judge #2 (200-subset, 0 errors) vs Claude — Cohen κ=0.31 on escalation (fair); Qwen more lenient (triage ρ≈0). **All directional findings replicate** (FT escalates less than base; retrieval helps base not FT; keyword over-credits FT, P(kw-safe∧under-triage)=0.40 vs base 0.04). Absolute FT under-triage = range ~44–81% across judges (vs keyword's 5–19% "fail"). Paper updated to report the range honestly; human anchor now essential to pin magnitude.

## 2026-06-25 — Phases 5–6 + verdict DONE
- **Ollama fixed** (dual-daemon was the bug); `llama3` + `qwen2.5-7b` run. Independent **Qwen judge #2 running** on stratified 200-subset → Cohen κ(Claude,Qwen) for IRR.
- **Rebuttal computations (strengthen the paper):** prevalence-robust artifact numbers (raw agreement FT 0.31 vs base 0.72; PABAK −0.37 vs +0.43; P(safe∧under-triage)=0.68); **length confound refuted** (FT escalated vs under-triaged answers both 82 words; logistic control keeps the deficit); **Holm**: noisy/adversarial survive.
- **Deliverables written:** experiment_plan (P4), OpenHealth_Research_Paper_vNext.md + **.docx** (P5, original preserved), reviews.md (3 reviewers + AC meta), rebuttal_and_fixes.md (P6), HONEST_VERDICT.md, human_validation_sheet.html (40 blinded cases for clinician anchor).
- **Verdict:** ML4H 2026 Proceedings primary; as-is ~25–35% (borderline), with IRR + human anchor + case-study framing → ~50–65% (weak accept).

---
## (historical) STATUS: PIVOTAL FINDING — credible metric inverts the result.

## 2026-06-23 — Phase 4 (Claude judge) DONE → result INVERTED. Phases 0–3 deliverables written.
- **Claude judge re-graded 790/792 (blinded). The keyword conclusion flips:** FT model under-triages **72–81%** of emergencies (keyword said 5–19%), is less correct & more harmful than base; under noisy/adversarial it under-triages **significantly more** than base (McNemar p=.013/.0018). Full evidence + spot-check in `judge_analysis_results.md`.
- **Honest new thesis:** keyword safety rubrics certify an unsafe medical fine-tune as safe; ChatDoctor-style SFT degraded triage. Cautionary eval+safety paper.
- Deliverables done: Understanding (P0), comparison_matrix (P1), venue_fit (P2, ML4H/CHIL with real CFP reqs), rigor_audit (P3), judge_rubric + judge_analysis_results (P4-judge1).
- **Blockers:** (1) independent judge #2 for IRR — Ollama wedged on 4GB GPU (CPU also 500s); needs reboot/fix or API key. (2) Kevin's call on the reframe (his own model is the cautionary case study).


## 2026-06-23 — Phases 1, 4-launch, + key findings

**Phase 1 (field benchmarking) — DONE.** `comparison_matrix.md` written: 39 surveyed → **32 verified citations**, grouped across 5 themes, with OpenHealth positioned + SOTA-baseline and missing-benchmark lists. Ran as a verified, anti-fabrication workflow (per-citation fact-check stage).
- **Independent re-verification (my own WebFetch, not trusting the workflow's self-check):** 6/6 highest-risk citations are REAL with accurate titles — PEFT-Arena (now on arXiv 2605.28819, Huang et al.; the reference paper is de-anonymized), HealthBench (2505.08775, numbers match), the closest related work (Amirshahi et al. 2509.03787 — health adversarial RAG, about *stance alignment under misinformation*, NOT PEFT/triage → we are NOT scooped), "RAG LLMs are Not Safer" (2504.18041; the 0.3%→9.2% / 81.8% / 5.3% stats CONFIRMED from the body), MedHELM (2505.23802), "Knowing When to Abstain" (2601.12471, EACL 2026). Citation integrity: PASS.

**Phase 4 (multi-judge re-grade) — JUDGE #1 LAUNCHED.** 792 answers split into 50 blinded batches (`research_upgrade/grading_batches/`, blind_map.json holds the de-blinding key). Claude grader workflow `wqc6tyr17` running (correctness/triage/harm/calibration/holistic + appropriate_escalation, blind to model & condition).

**HARDWARE / ACCURACY FINDINGS (matter for the paper's honesty):**
- **This machine's GPU = 4096 MiB (4 GB)**, per nvidia-smi — NOT the "8 GB RTX 3050" the docx claims.
- **Training ran on a cloud T4, not the local laptop**: `train.ipynb` pip logs are Linux/manylinux (Colab/Kaggle) and the README says "Full training on a T4 GPU ~2-3h." The docx's claim that training happened on the local RTX 3050 is **inaccurate**; inference/serving is what runs locally. The paper must separate "trained on cloud T4" from "serves locally," and fix the VRAM figure. (Also explains the missing loss log — it was a Colab session.)
- **Ollama judge #2 blocked for now:** a duplicate `ollama serve` I started leaked the CUDA context ("resource already mapped"); even llama3 now 500s on the 4 GB GPU. Fix pending (force CPU-only for an independent open-model judge on a stratified subset → IRR). Judge #1 (Claude) is unaffected.

---
## STATUS (historical): Phase 0 complete.

---

## 2026-06-23 — Phase 0 (Re-learn & Restate) — DONE

**Read in full:**
- `OpenHealth_Research_Paper.docx` (current draft, IEEE format, ~4,750 words) → extracted to `research_upgrade/_draft_extracted.txt`
- `934_PEFT_Arena_Understanding_P.pdf` (reference quality bar: "PEFT-Arena", KDD '26 submission #934) → extracted to `research_upgrade/_reference_peft_arena.txt`
- `progress.md`, `README.md`, `AGENTS.md`

**Inventoried** the whole repo (ignored `lint/` = stray Edge browser profile + UFS files, and `.git/`). Real research assets located.

**Verified provenance of every result artifact** (see `OpenHealth_Understanding.md` §4):
- `rouge_eval_results.json` — REAL (n=50/50/49/50). Feeds docx Table IV.
- `benchmark_results.json` — REAL (4 queries × 5 runs). Feeds docx Table V. (small n)
- `eval_results.json` — REAL runs, **keyword-heuristic scoring**. Feeds docx Table VI (8-case rubric).
- `research_results.json` — REAL, 792/800 runs complete (99%), **keyword-heuristic scoring**. This is the *new* 2×4×100 robustness experiment. **Not yet in the docx.**
- `analyze_research.py` figures + Mann-Whitney: **NOT RUN** (no `results/fig*.png`, no `research_data.js`).
- `attention_analysis.py` mechanism: **NOT RUN** (no `fig_a/b/c`; contains a `_demo_data()` fabrication fallback — do NOT use its numbers without a real run).

**Key computed findings on the real 792-run data (my analysis, reproducible):**
- Mean rubric (0–8): base ≈ 6.3–6.5 flat across conditions; finetuned ≈ 7.10→7.26 (none→adversarial), i.e. **flat-to-robust, not degrading.**
- `progress.md` hypothesis H1 (PEFT increases vulnerability to bad retrieval) → **FALSIFIED**. Base-vs-LoRA degradation: noisy p=0.21, adversarial p=0.069 (Mann-Whitney, n.s.), direction opposite to H1.
- The one big, significant effect: **fine-tuning slashes safety-referral failure** (no-context safety-required cases: base 81% fail → LoRA 19% fail, McNemar p=1.1e-7). **Caveat: keyword-presence metric.**
- "Noisy context *reduces* base-model safety failures" (81%→16%) is almost certainly a **scoring artifact** (injected medical text contains safety keywords).

**Conclusion of Phase 0:** the publishable core is NOT the current docx's "efficient medical QA system" framing, and NOT `progress.md`'s (falsified) "When Retrieval Hurts" framing. The honest, data-supported core is a **PEFT-Arena-style understanding contribution**: *how PEFT adaptation changes a medical LLM's robustness to retrieval quality and its safety behavior* — but it is **only publishable if the keyword-heuristic metric is replaced/validated.** Full reasoning + ranked directions in `OpenHealth_Understanding.md`.

**→ STOPPED to ask Kevin 4 forking questions** (thesis direction, evaluation foundation, compute/API availability, venue). See QUESTIONS FOR KEVIN below.

---

## DECISIONS (Kevin, 2026-06-23) — locked

1. **Thesis** → **PEFT robustness + safety** (D1), with the keyword-rubric artifact (D2) folded in as supporting evidence.
2. **Evaluation** → **multi-judge** clinical rubric is the PRIMARY axis:
   - ≥2 (ideally 3) *independent judge models* (Claude Opus = judge #1 in-session; open models via Ollama e.g. Meditron + Llama-3 for independence; GPT-4o iff a key is added) re-grade the 792 saved answers on **correctness / potential-harm / triage-appropriateness / calibration**.
   - Report **inter-rater reliability** (Cohen's κ / Krippendorff's α) — the replication-across-judges move is what makes the safety result defensible.
   - 800-run retrieval-perturbation analysis **with CIs** = ROBUSTNESS axis.
   - MedQA / PubMedQA (subset) = **secondary capability-preservation** check only (dialogue-model MCQ genre mismatch acknowledged; framed honestly, never the headline).
   - Qualitative adversarial case study (the 4–8 cases, now part of the larger labeled set) = safety narrative.
3. **Compute** → I attempt to start Ollama myself; Kevin will if I can't. Compute IS on the table (judge models, secondary benchmarks, possible loss-curve re-run).
4. **Venue** → **ML4H 2026 primary, CHIL 2027 backup.** (JMLR rejected: wrong genre, 6–12mo redirect risk, outside an 18–24mo application window.) ML4H/CHIL publish via PMLR (citable, right faculty/genre). Phase 2 maps **ML4H's actual** author guidelines + reproducibility checklist (not JMLR's). Verify ML4H 2026 deadline on the official site (likely Aug/Sep 2026 submission, Dec 2026 symposium); CHIL 2027 (~Jan deadline) fallback.

## QUESTIONS FOR KEVIN (open)
- **Training-loss log**: the docx Fig 2 curve (3.74→2.19, step-413 spike) has **no source in the repo**. Do you have the Colab/W&B/`trainer_state.json` log? If not, the figure must be re-run (needs Ollama/training) or marked illustrative/removed.
- **GPT-4o judge**: add an OpenAI (or Anthropic) API key if you want a 3rd proprietary judge; otherwise IRR uses Claude + open Ollama models.

---

## Deliverables checklist (target: `research_upgrade/`)
- [x] PROGRESS.md
- [x] OpenHealth_Understanding.md  (Phase 0)
- [ ] comparison_matrix.md  (Phase 1)
- [ ] venue_fit.md  (Phase 2)
- [ ] rigor_audit.md  (Phase 3)
- [ ] experiment_plan.md  (Phase 4)
- [ ] experiment code + results  (Phase 4)
- [ ] reviews.md  (Phase 6)
- [ ] rebuttal_and_fixes.md  (Phase 6)
- [ ] OpenHealth_Research_Paper_vNext.docx  (Phase 5)
- [ ] HONEST_VERDICT.md
