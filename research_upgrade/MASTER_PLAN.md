# MASTER_PLAN.md — the complete plan to a journal-grade, acceptable paper
*Written 2026-07-07. Compute unconstrained (multi-day local runs authorized). This document is the single source of truth for what remains, why, in what order, and what "done" looks like. Progress ticks in `PROGRESS.md`; live run state in `research_upgrade/runs/`.*

---

## 0. Where we actually stand (one paragraph, no varnish)

We have a real, novel, clinically-consequential finding with a complete draft (`OpenHealth_Research_Paper_vNext.md/.docx`): **dialogue SFT taught a medical LLM to satisfy a lexical safety rubric ("consult a doctor") without learning triage** — keyword-vs-judge agreement collapses (raw 0.72→0.31, PABAK +0.43→−0.37), and two independent LLM judges agree on every direction (fine-tune escalates less than base; retrieval helps base, not the fine-tune; the rubric over-credits the fine-tune) while disagreeing on absolute severity (under-triage ≈44–81% across judges vs the rubric's implied 5–19%). What stands between this and a strong accept is **evidence breadth**: n=1 model pair, single decoding sample, a team-authored 100-case set, fair-only judge agreement with no human anchor, and one newly discovered methodological flaw we must fix ourselves before a reviewer finds it (§2). Everything below is engineered to close those gaps with days of local compute plus ~30 minutes of human time.

## 1. The acceptance model — what reviewers must be able to say

A paper gets in when reviewers can say all five of these. Each maps to concrete work:

| Reviewer sentence | What produces it | Status |
|---|---|---|
| "The central finding is real and not a judge artifact." | 3-judge panel (Claude/Qwen/Mistral) + majority-vote labels + Krippendorff's α + **human anchor** (40-case clinician sheet) | 2 judges done (κ=0.31 fair); 3rd judge + human = plan items J1, H1 |
| "It generalizes beyond one model." | Replicate on a **second, independently-trained medical SFT pair** (medllama2 vs llama2) — zero training needed | plan item G1 |
| "The mechanism is understood." | **Corpus-transmission analysis**: show HealthCareMagic-100k's own reference answers carry the keyword register and under-escalate; SFT faithfully transmits it | plan item M1/M2 |
| "The methodology is airtight." | Fix the **num_ctx=512 truncation flaw** with a corrected, context-logged, **multi-seed** re-run; Holm-corrected stats; released harness | plan items R1–R3 |
| "It engages the field's benchmarks." | **MedQA/PubMedQA** capability check + **HealthBench** emergency-subset external replication | plan items B1, B2 |

## 2. The flaw we must fix before a reviewer finds it: context truncation

The original harness generated with **`num_ctx=512`** while injecting 3–5 chunks of ~800 tokens (~2,400–4,000 tokens of context). The window cannot hold the injected evidence, so **the clean/noisy/adversarial conditions were substantially truncated** — the "retrieval condition" manipulation was weaker than described. Consequences, honestly assessed:
- The **metric-collapse finding (C1) is unaffected** — it holds in the `none` condition with no context at all.
- The **retrieval-robustness numbers are diluted** and must be re-established with a corrected pipeline (contexts that demonstrably fit, and are logged verbatim).
- The v2 design: 3 chunks × ≤400 tokens each (~1,200 tokens context), `num_ctx=2048`, contexts cached and saved verbatim per run, `options.seed` set → fully reproducible. A CPU job quantifies exactly how truncated v1 was, and the paper discloses it.

## 3. Work plan (every item: design → compute → decision gate)

### Tier A — running locally, unattended (the multi-day queue, in order)
| ID | Job | Design | Est. GPU-hours |
|---|---|---|---|
| R0 | `corpus_build` | Extract 1,823 chunks from `website/chroma` SQLite; re-embed with nomic-embed-text; save vectors + docs | 0.3 |
| R1 | `precompute_contexts` | For all 100 cases × {clean, noisy, adversarial}: hybrid retrieval (dense-MMR + BM25 + RRF k=60; τ=0.020 filter for clean; top-3, 400-token cap) → `contexts_cache.json` with verbatim text + provenance | 0.3 |
| T1 | `truncation_audit` (CPU) | Token-count v1 prompts vs 512; report % truncated per condition | 0 |
| M1 | `corpus_register` (CPU) | Keyword-register stats: % of HealthCareMagic reference answers containing safety keywords; register n-grams pre/post SFT | 0 |
| M2 | `reference_sample` (CPU) | Extract ~60 emergency-presentation Q&A pairs from the parquet → file for judges ("do the *human* reference answers escalate?") | 0 |
| R2 | `grid_seed101..103` | Corrected 2×4×100 grid × 3 seeds (2,400 gens, ctx 2048, contexts logged, model-major loop) | 27–47 |
| G1 | `pair2` | Pull `medllama2` + `llama2` (second independent SFT/base pair, different lineage & authors); same 4×100 grid, seed 101 (800 gens) | 9–16 |
| B1 | `mcq_bench` | MedQA (500 q) + PubMedQA (500 q) × base & FT; constrained-answer prompt; parse-failure rate reported | 6–11 |
| B2 | `healthbench_run` | Fetch HealthBench (OpenAI simple-evals); filter emergency-referral theme; run both models on ~100 cases | 2–4 |
| J1 | `judge_local` | Qwen re-grade: full v1 792 + stratified v2/G1 subsets; pull `mistral:7b-instruct` as judge #3 over ~400 stratified | 10–15 |
**Total ≈ 55–95 GPU-hours ≈ 2.5–4 days.** The queue is resume-safe (per-job markers, incremental saves), fail-soft (a failed job logs and yields to the next), stoppable (`runs/STOP` file), and survives session/terminal closure (detached process).

### Tier B — my work as results land (agent/analysis, no local GPU)
- Claude-judge grading of every new output batch (v2 grid, pair2, HealthBench, M2 reference answers) — same blinded 50-batch workflow as before.
- 3-judge majority labels; Krippendorff's α; re-run the full stats suite (McNemar + Holm, bootstrap CIs, sensitivity/specificity) on **majority labels**.
- The M1/M2 mechanism section: if the *human* reference answers under-escalate per the panel, the paper's causal story completes: *the most-used medical dialogue corpus embeds under-triage; SFT transmits it; lexical metrics can't see it.* That is the difference between a case study and a field-level finding — every paper that fine-tunes on ChatDoctor/HealthCareMagic inherits it.
- Rewrite the paper around the upgraded claim set (§4), regenerate reviews (Phase 6 iteration 2), update HONEST_VERDICT.

### Tier C — needs Kevin (the only human-gated items)
| ID | Item | Time | Why it matters |
|---|---|---|---|
| H1 | **Rate `human_validation_sheet.html`** (40 blinded emergencies) — or hand to any clinician | ~30 min | Converts "LLM judges disagree with a regex" into "humans confirm the regex is wrong." The single highest-leverage item on this list. |
| H2 | (Optional) OpenAI/Anthropic API key | 5 min | GPT-4o as 4th judge — cross-vendor IRR |
| H3 | (Optional) One Colab T4 session | ~3 h | A logged re-fine-tune (restores loss-curve provenance + a same-recipe second subject). Not required — G1 covers generalization without it. |

## 4. The upgraded paper (what all this buys)

**Working title:** *"Fluent but Unsafe: Dialogue Fine-Tuning Teaches Medical LLMs to Satisfy Safety Metrics Without Learning Triage"*

Claim set after the queue completes (each with its evidence):
1. **C1 Metric collapse** — lexical safety scoring inverts under SFT (v1 + corrected multi-seed v2; 3-judge majority; human anchor).
2. **C2 Generalization** — replicates on a second, independently-trained medical SFT pair (G1).
3. **C3 Mechanism** — the training corpus itself carries the unsafe register; SFT transmits it (M1/M2). *The most novel claim if it lands.*
4. **C4 Robustness (corrected)** — retrieval quality × escalation, with contexts that verifiably fit and are logged (R2).
5. **C5 Capability** — MedQA/PubMedQA: what SFT did to medical knowledge (B1). Any result is informative.
6. **C6 External validity** — under-triage replicates on physician-authored HealthBench emergency cases (B2).
7. **C7 Protocol + artifact** — released 3-judge escalation-audit protocol + 100-case set + harness ("a reusable triage-safety audit"), the kind of artifact ML4H explicitly solicits.

**Kill criteria / honesty gates (pre-registered here):**
- If the corrected R2 grid materially changes any v1 conclusion → the corrected numbers are primary and the paper says so.
- If G1's second pair does *not* replicate → the paper reports it and scopes C1 to the recipe studied (still publishable; weaker).
- If M2 shows human reference answers DO escalate properly → drop the corpus-transmission claim; the finding becomes "SFT under-learned triage," not "data taught it."
- If the human anchor (H1) sides with the lenient judge → report the range honestly with the human point estimate; the metric-collapse claim survives regardless (it holds under either judge).
- No number enters the paper without a run that produced it. PROPOSED stays PROPOSED.

## 5. Venue strategy (sequential, no dual submission)

1. **ML4H 2026 Proceedings Track** (PMLR, 8 pp, double-blind; 2025 cadence: submissions ~Sep 8, decisions ~Oct 27, symposium Dec). ~10 weeks out — the queue finishes with ~9 weeks to spare for writing and polish. **Primary.**
2. **TMLR** (rolling, no deadline) — the strongest *journal* fit: it published "LoRA Learns Less and Forgets Less" (our closest genre precedent, Featured Certification); TMLR's bar is *claims supported by evidence + audience interest*, not novelty-chasing — exactly this paper's shape. **Journal target if we skip/miss/fail ML4H.**
3. **CHIL 2027** (~Feb 2027, PMLR, 8–10 pp; mandatory Data/Code + IRB statements — already drafted). **Fallback.**
4. **npj Digital Medicine** — stretch; realistic only if H1 grows into a small multi-clinician panel (≥2 raters). Revisit after H1.

## 6. Failure modes & mitigations
- **Machine interruption mid-queue** → every job resumes from incremental saves; queue skips `.done` jobs; relaunch = one command.
- **Ollama wedges again** → root cause was duplicate daemons (documented); queue health-checks the daemon between jobs and restarts it detached if dead.
- **medllama2/HealthBench/MedQA downloads fail** → jobs are fail-soft; alternates listed in the job code; worst case the claim drops out per §4 kill criteria.
- **4 GB VRAM too tight at ctx 2048** → partial CPU offload is automatic (slower, still correct); queue measures per-gen latency and logs ETA drift.
- **Judge panel still disagrees after 3 judges** → majority label + human anchor is the defensible instrument; the disagreement itself is reported as evidence about automatic triage scoring.

## 7. Definition of done
A submission-ready package: corrected multi-seed results with logged contexts; 3-judge majority labels with α and a human-anchored subset; a replicated (or honestly non-replicated) second model pair; mechanism section; MedQA/PubMedQA + HealthBench tables; rewritten 8-page paper (PMLR LaTeX + docx), anonymized repo, completed checklist; Phase-6 re-review reaching a defensible weak-accept/accept meta-review — or an explicit statement of what still falls short and why.
