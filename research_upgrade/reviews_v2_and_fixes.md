# reviews_v2_and_fixes.md — hostile re-review of the clean-benchmark paper + fixes applied
*Phase 6, iteration 2 (2026-07-11). Adversarial ML4H committee (3 lenses + area chair) reviewing `OpenHealth_Research_Paper_vNext`. Full raw reviews: workflow `wvxtpvohv`.*

## Meta-review verdict
**Consensus 3/5 (borderline)** — all three reviewers scored 3/5, conf 3/4, and converged on the same fixes. "Idea and execution strong; evidence oversold in identifiable ways." Not at risk: the core contribution (6.5× metric vs identical 24% escalation; κ 0.61→0.03 collapse; corpus mechanism; mitigation asymmetry). Reviewer 1 (clinical) **independently verified** the 72 emergencies are genuine/canonical and the judge rationales are clinically sound; Reviewer 2 **reproduced** the collapse numbers (AUROC drop z≈5.3). Path to borderline-accept is from existing data + the human anchor.

## Fixes APPLIED this iteration (from existing data, no new experiments)
| Reviewer critique | Fix applied |
|---|---|
| "IDENTICAL" is an underpowered null sold as equivalence (R2, R3) | Added **TOST equivalence test**: per-case escalation diff 0.0%, 90% CI [−6.2%,+6.2%], within ±10-pt margin → positive equivalence, not just n.s. (§4.1) |
| Keyword rubric is a strawman (R1, R3) | **Regenerated the stricter urgency-only rubric from the raw judgments** (now reproducible via `analysis/robustness_stats.py`). The finding is sharper and fully honest: the 6.5× illusion is *specifically* the referral register — under a genuine-emergency-token rubric (excluding *doctor/consult/"as soon as possible"*) the fine-tune's "safe" rate falls 90%→**35%** (base 27%), keyword–judge agreement stays high for *both* (0.83→0.31 becomes **0.89→0.81**), and only **15%** of its strict-"safe" answers under-triage (vs **68%** incumbent). This *localizes* the failure to the referral-keyword rubric used in practice and *validates the urgency-sensitive fix* we recommend; a residual ~1.3× gap remains (lexical scoring mitigated, not cured). Supersedes the earlier ad-hoc "57% vs 31%" figure, which did not reproduce (§4.2). |
| 6.5× ratio leverage-sensitive to small denominator (R2) | Report the **55-pt absolute gap** (65%→10%) as the robust statement alongside the ratio (§4.2) |
| AUROC 0.67 is not chance, over-stated as collapse (R3) | Reworded: "drops 0.91→0.67 — above chance but no longer usable to rank by safety" (§4.3) |
| PABAK/κ over-interpreted at 90% prevalence (R1, R2) | Added prevalence caveat; lead with prevalence-robust raw agreement + conditional error (§4.3) |
| Ground-truth circularity — the deepest critique (R1, R2, R3-critical) | Clarified the **primary benchmark's emergency labels are AUTHORED (not judge-defined)** — judge defines only escalation; and the headline is a **base-vs-fine-tune agreement contrast, invariant to the judge's absolute threshold** (§4.6). Corpus-audit labels flagged as the judge-dependent, supporting number. |
| Provenance: generalization presented present-tense while runs were failed markers (R1) | Cleaned stale `.failed` markers (encoding artifact; models installed); generalization pair is now actually generating; softened language to "we additionally evaluate … to probe" (§6) |
| IRR under-reported as weak (R1) | Framed κ=0.31 honestly as *fair*; clarified the collapse (a contrast) is threshold-invariant; multi-judge majority + Fleiss κ in progress (§4.6) |

## Generalization result LANDED (2026-07-11) — honest NEGATIVE, scopes the claim (§4.11)
- **Llama-2 / Medllama2 pair, 767 blind-judged generations.** The collapse does **NOT** replicate — and that is the correct, mechanism-confirming outcome. Medllama2 (a public medical fine-tune of Llama-2) **improves** triage over its base (escalation 21%→55% pooled; 57% vs 6% no-retrieval; specificity 0.91–0.96, stable across all conditions), and its keyword rubric degrades only **mildly** (κ 0.45→0.35, raw 0.75→0.70, AUROC 0.87→0.81, P(kw-safe∧under) 0.22→0.29) versus the primary pair's κ 0.61→**0.03** / P 0.14→**0.68**. Spot-checked: medllama2's escalations are genuine ("go to the emergency department", correctly names PE/meningitis); llama2's under-triage is real (clarifying-question deflection).
- **Why this is a strength, not a weakness:** it (a) pre-empts "you only tested your own model", (b) *confirms* the §4.8 mechanism — gaming tracks the training corpus's triage behaviour, not PEFT per se, (c) honestly scopes the headline to the widely-used ChatDoctor/HealthCareMagic register. Reproducible via `analysis/generalization_compare.py` → `generalization_compare.json`.

## Fixes IN PROGRESS (overnight generation / analysis)
- **Multi-judge IRR** (Qwen+Mistral majority-of-3 + Fleiss κ) on the clean emergencies: after generation frees the GPU.
- **Single-seed → variance:** seeds 102/103 queued.
- **Original-benchmark re-run** (corrected 2048-ctx): generating; §4.7 will report the completed run.

## Fix that NEEDS KEVIN (highest leverage per the AC)
- **Return the clinician anchor** (`human_validation_sheet.html`, ~40 blinded emergencies): pins the absolute 24%/76% and the corpus 13% against a human, resolving the circularity. Without it, absolute rates are honestly labeled "judge-defined." *This is the single action the area chair calls highest-leverage for moving to accept.*

## Net
Every fixable-now critique is addressed. The paper now states equivalence properly, is robust to a stricter rubric, quantifies the absolute gap, and honestly scopes the circularity to the (supporting) corpus number while showing the headline is judge-threshold-invariant. Remaining gaps (human anchor, full IRR, generalization, multi-seed) are in progress or flagged, not hidden.
