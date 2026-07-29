# venue_fit.md — Venue selection & requirement gap analysis
*Phase 2. Provenance: requirements quoted from the venues' own CFP pages (fetched 2026-06-23). Decision locked with Kevin: **ML4H 2026 primary, CHIL 2027 backup.** This file documents the honest reasoning and a point-by-point checklist.*

---

## 1. Why NOT JMLR (the original aspiration)

JMLR is a **machine-learning theory & methodology journal**. It publishes general-purpose learning algorithms, statistical learning theory, and broadly-applicable methods. It very rarely publishes an *applied clinical LLM system/evaluation* paper, which is what OpenHealth is even in its sharpened form. Concretely, OpenHealth would fail JMLR's implicit bar on: generality (one base model, one PEFT method, one clinical corpus), methodological novelty (we compose existing techniques and *study* them; we do not introduce a new learning algorithm), and theoretical contribution (none). Expected outcome: desk-reject or a 6–12 month round-trip ending in "out of scope." Inside an 18–24 month application window, that is a bad bet. **JMLR: not pursued.**

## 2. Candidate venues (real, fetched requirements)

> ⚠️ Naming: there are **two** distinct "ML for health" venues. **MLHC** (mlhc.org, *Machine Learning for Healthcare*) had its 2026 deadline **April 17 2026 — already passed**. **ML4H** (ahli.cc/ml4h, *Machine Learning for Health symposium*, AHLI) is the one we target. Don't confuse them.

### 2a. ML4H 2026 — AHLI Machine Learning for Health symposium — **PRIMARY**
*(2026 CFP not yet posted as of 2026-06-23; figures below are the 2025 CFP, which sets the cadence. Confirm 2026 numbers when the CFP drops — expected ~Sep deadline / Dec symposium.)*
- **Proceedings Track:** archival, published in **PMLR**; **up to 8 pages at submission** (excl. references & appendices); double-blind; cannot be under review elsewhere.
- **Findings Track:** non-archival; up to 4 pages; authors retain copyright (a softer landing / fast feedback option).
- **Fully anonymized** submissions. **Reciprocal reviewing:** each submission must nominate an author to review ≥3 papers.
- 2025 dates (template): submission **Sep 8 (AoE)**, decisions **Oct 27**, symposium **Dec 1–2**.
- Fit: **exact.** ML4H 2024/2025 accepted medical-LLM PEFT/RAG/safety-eval papers. Our honest PEFT-robustness + safety study with a multi-judge eval is squarely in scope. PMLR = citable, same proceedings family as ICML/AISTATS; the clinical-ML faculty who matter review here.
- **Target: ML4H 2026 Proceedings Track (8-page archival).**

### 2b. CHIL 2027 — AHLI Conference on Health, Inference, and Learning — **BACKUP**
*(CHIL 2026 deadline was Feb 4 2026 — passed; conference Jun 28–30 2026. CHIL 2027 ≈ Feb 2027 deadline.)*
- **8–10 pages** (incl. all figures/tables; unlimited refs/supp); archival **PMLR**, open access; double-blind (**code repos must be anonymized**).
- Three tracks: Models & Methods / Applications & Practice / **Impact & Society** (we fit Applications & Practice, or Impact & Society for the safety-eval framing).
- **Three MANDATORY sections or desk-reject:** (1) **Data & Code Availability**, (2) **Author Contributions** (camera-ready), (3) **IRB statement**.
- Fit: strong. Slightly more pages than ML4H → room for the full multi-judge analysis.

## 3. Point-by-point requirement checklist (OpenHealth → ML4H / CHIL)

Legend: **PASS** / **PARTIAL** / **FAIL** with the specific fix.

| Requirement | Status | Fix |
|---|---|---|
| **≤8 pages (ML4H Proc.)** archival | **PARTIAL** | The vNext content (intro, related work, 2×4 method, multi-judge eval, robustness, limitations, ethics) must fit 8 pages. Push the full citation matrix, extra tables, and the keyword-artifact deep-dive to the appendix. CHIL's 8–10 gives more room if ML4H is too tight. |
| **PMLR LaTeX template** | **FAIL (not started)** | Produce a LaTeX version in the ML4H/PMLR `jmlr`/`ml4h` style. (Currently only .docx.) Phase 5 deliverable. |
| **Double-blind / full anonymization** | **FAIL** | Strip author names (Kevin Joy Thomas et al.), MVSR affiliation, and the `kevinjoythomas/...` HuggingFace + Vercel/ngrok links from the submission version. Anonymize the released code repo. |
| **Original clinical claim / novelty** | **PASS** (after pivot) | Centered on PEFT retrieval-robustness + safety (data-supported), not the falsified "When Retrieval Hurts." |
| **Proper baselines** | **PARTIAL** | Have base-vs-LoRA × 4 conditions. Reviewers will want ≥1 standard MCQA retention check (MedQA/MMLU) for the "no capability loss" sub-claim — **secondary** benchmark, run when Ollama is back (Kevin's plan). |
| **Statistical validity (CIs, multiple-comparison correction, IRR)** | **PARTIAL → in progress** | Phase 4: 95% bootstrap CIs, McNemar for safety, Holm-corrected family of robustness tests, **inter-rater reliability** (Cohen κ / Krippendorff α) across judges. |
| **Right metrics (not just accuracy/ROUGE)** | **PARTIAL → in progress** | Replacing the keyword rubric with the multi-judge clinical rubric (correctness/triage/harm/calibration). ROUGE demoted to a secondary table. |
| **Reproducibility: seeds, hyperparams, compute, env, runnable code** | **PARTIAL/FAIL** | Hyperparams documented. **Gaps:** (a) **training-loss log is missing** — re-run on a logged T4 or mark the curve illustrative; (b) the experiment used a single decoding pass (temp 0.3) — document, and ideally add seeds/multiple samples for the headline; (c) the retrieved context per run was **not saved** (only `n_chunks`) — document this limitation and ideally re-run saving contexts; (d) release the seeded harness + judge prompts. |
| **Data provenance & integrity** | **PARTIAL** | ChatDoctor (HealthCareMagic-100k) is public — state it. Eval cases are team-authored → disclose, and the multi-judge re-grade + IRR is the mitigation. Note train/eval overlap handling (the ROUGE memorisation/generalisation split is good practice; carry that rigor over). |
| **Data & Code Availability statement (CHIL mandatory)** | **FAIL (not written)** | Add: public training data + released code + the 100-case eval set + 792 graded outputs. |
| **IRB statement (CHIL mandatory)** | **PASS (just state it)** | No human-subjects recruitment, no real patient data (public dataset + synthetic/curated cases) → "No IRB approval required." Must be stated explicitly. |
| **Clinical-safety / ethics / data-privacy statement** | **FAIL (not written)** | Add a real ethics section: not a clinical device, no PHI, intended-use & misuse, the under-triage risk the paper itself studies, local-deployment privacy rationale. Phase 5. |
| **"Trained/served on consumer hardware" honesty** | **FAIL (currently inaccurate)** | Fix: training ran on a **cloud T4** (~2–3 h), not the local RTX 3050; the local GPU here is **4 GB** not 8 GB. Re-state as "trained on a single cloud T4; serves locally on a consumer 4–8 GB GPU." |

## 4. Recommendation & timeline

**Primary: ML4H 2026 Proceedings Track** (8-page archival PMLR). Best ratio of real-publication value to acceptance odds for an honest, narrow PEFT-robustness + safety study; right audience and citation surface for Kevin's US PhD-application window. Expected deadline ~Sep 2026 / symposium Dec 2026 — **confirm when the 2026 CFP posts** (not live yet).
**Backup: CHIL 2027** (~Feb 2027 deadline, 8–10 pages) if we miss ML4H or want more space; note its mandatory Data/Code + IRB sections.
**Fallback for speed/feedback: ML4H 2026 Findings Track** (4 pages, non-archival) — usable as a pressure-test if the Proceedings bar isn't cleared in time.

**Bar we must clear for ML4H Proceedings:** a credible (multi-judge, IRR-backed) safety result with CIs and honest reproducibility, in 8 pages, fully anonymized, with an ethics statement — i.e., finish Phases 4–6 and the honesty fixes above.
