# experiment_plan.md — experiments (done + proposed)
*Phase 4. Reframed around the confirmed thesis: keyword safety rubrics mask under-triage in fine-tuned medical LLMs; ChatDoctor-style SFT degraded triage. "Done" = a real run produced the numbers. "PROPOSED" = not yet run.*

## DONE (real runs, numbers in the paper)
| # | Experiment | Hypothesis tested | Method | Metrics | Result | Reviewer value |
|---|---|---|---|---|---|---|
| E1 | **2×4×100 generation study** | retrieval quality × PEFT changes answers | base `llama3` & FT `openhealth-doctor` × {none,clean,noisy,adversarial} × 100 annotated cases (792/800 real) | stored answers | the substrate for everything | the controlled grid reviewers want |
| E2 | **Multi-judge clinical re-grade** | the metric, not the model, drove the old "safety win" | blinded Claude judge (790/792) on correctness/triage/harm/calibration + appropriate_escalation; **Qwen2.5-7B independent judge (running)** for IRR | per-axis means, escalation rate, **Cohen κ/Krippendorff α** | FT under-triages 72–81% of emergencies; less correct, more harmful than base | the credible metric + independence |
| E3 | **Keyword-vs-judge validity** | the keyword rubric is invalid for the FT model | Cohen κ between `safety_failed` and judge `appropriate_escalation` | κ, P(keyword-safe & judge-undertriage) | κ collapses 0.43(base)→0.05(FT); 68% of "safe" FT answers are under-triage | the headline result |
| E4 | **Detection framing** | FT trades sensitivity for nothing | sensitivity (emergencies escalated) & specificity (no over-triage) by model×cond, 95% CI | sens/spec | FT worse on both axes; retrieval helps base escalate, not FT | the framing health-ML reviewers expect |
| E5 | **Retrieval-robustness** | does bad retrieval break either model? | paired none→{noisy,adversarial} drop; McNemar base-vs-FT escalation | Δ, McNemar p | FT under-triages significantly more than base under noisy/adversarial (p=.013/.0018); overall quality flat | robustness axis |
| E6 | ROUGE (existing) | FT shifts register | ROUGE-1/2/L, mem vs gen split | F-scores | FT > base ROUGE (style shift) — demoted to secondary; shows ROUGE rewards the same surface shift | honest secondary |
| E7 | Retrieval latency (existing) | hybrid ≈ dense latency | 4 queries × 5 runs | ms | hybrid ≈ dense | systems detail |

## IN PROGRESS
- **E2b — Qwen2.5-7B independent judge** (background): stratified ~200-answer subset → Cohen κ(Claude,Qwen) on `appropriate_escalation`. If high agreement on the under-triage calls, the finding is judge-robust; if not, escalates the need for human validation.

## PROPOSED (write the runnable code; run as time/compute allow — Ollama now works locally)
| # | Experiment | Why a reviewer wants it | Blocker / cost |
|---|---|---|---|
| P1 | **Human/clinician validation subset** (~40 emergencies, blind) → κ vs LLM judges | gold-standard anchor for the LLM-judge metric | needs Kevin or a clinician; sheet prepared (`human_validation_sheet.html`) |
| P2 | **Secondary capability check: MedQA/PubMedQA subset**, base vs FT | shows whether FT also lost medical knowledge (capability-preservation) | runnable now via Ollama (~2 h); needs MedQA download; genre-mismatched (frame honestly) |
| P3 | **τ-gate sweep** τ∈{0,.01,.02,.05,.1} | kills the "magic constant" critique | re-run generation (~hours) |
| P4 | **Multiple seeds/samples** for the headline conditions | single-sample LLM outputs aren't robust point estimates | re-run (~hours) |
| P5 | **External validation: HealthBench emergency-referral subset** | rebuts "team authored its own benchmark" | needs HealthBench access |
| P6 | **Re-run saving injected contexts** + judge-verify adversarial docs | reproducibility + validates the adversarial arm | re-run (~hours) |
| P7 | **Train-on-logged-T4 re-run** | restores the (currently unverifiable) loss curve | needs a T4 (Colab) |

## Priority for a submittable paper
1. E2b IRR (running) → 2. P1 human subset (async, Kevin) → 3. P2 capability check (runnable now) → then P3–P6 as time permits. E1–E5 already carry the core contribution.
