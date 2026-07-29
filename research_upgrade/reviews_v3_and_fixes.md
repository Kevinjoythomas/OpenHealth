# reviews_v3_and_fixes.md — third hostile review (5 diverse lenses + AC synthesis) + fixes applied
*Phase 6, iteration 3 (2026-07-12). Workflow `wf_30dae5ef-c79`: clinical / statistics / novelty / honesty / reproducibility reviewers, each instructed to find the strongest reasons to reject and to mark only genuine defects `is_real`. Raw dump: `reviews_v3_raw.txt`.*

## Meta-verdict
Scores clinical 3 / stats 3 / novelty 3 / honesty 3 / reproducibility 4 → **mean 3.2, Major Revision**. Every reviewer independently **reproduced the load-bearing numbers from the released artifacts** and found no fabrication. The consensus: the science is sound but several claims tilted toward the narrative (pooling, rounding, conditional-vs-joint, "systemic", "decisively", "stable"). 18 real issues; none threatens the core finding; all fixable without new experiments.

## MAJOR fixes APPLIED (M1–M6)
| # | Issue | Fix applied |
|---|---|---|
| **M1** | **Flagship number conflates JOINT vs CONDITIONAL.** Prose said "68% of the fine-tune's keyword-*safe answers* are under-triage" (conditional) but reported the *joint* P(kw-safe ∧ under-triage)=0.677. True conditional is **75% (ft) / 40% (base)**, not 68%/14%; base was off ~3×. | Report the **conditional (PPV failure) 75% vs 40%** as the validity statement (abstract, §1, §4.3), and where the joint is used, reword prose to "X% of the fine-tune's *emergency answers* are keyword-safe yet under-triage" (matches Table 2). §4.7 → "of its safe answers 72% under-triage (base 49%)"; §4.2-strict → "43% (base 27%) vs 75% incumbent"; §4.3 none-subset → 73% vs 33%. All base/ft pairs now the same measure. |
| **M2** | **TOST equivalence over-claimed.** The tight [−6.2,+6.2] CI is the case-averaged test; the one fully-independent single-condition test (no-retrieval) fails equivalence (CI +16.8% > ±10-pt), and the pooled variant pseudo-replicates. | §4.1 rewritten: equivalence rests on the **case-level TOST (72 independent units)**; explicitly report the no-retrieval TOST fails (CI +16.8%, "directionally consistent but underpowered"), and state the per-condition McNemars are underpowered nulls, reported only to show no *detectable* difference — they do not carry equivalence. |
| **M3** | **Missing Goodhart / reward-hacking / shortcut-learning prior art** — the most direct canon was uncited. | Added a §2 paragraph situating the finding in Goodhart's law [Strathern'97], specification gaming / reward hacking [Amodei'16, Krakovna'20, Pan'22, Skalse'22], and shortcut learning [Geirhos'20], then states the delta: a fine-tuning-induced, model-dependent collapse of a *specific deployed* metric, quantified as a keyword–judge agreement contrast, traced to an identifiable corpus, and **incidental (not adversarial)**. ⚠ CITATIONS NEED VERIFYING + ADDING TO BIB (see below). |
| **M4** | **"the failure is systemic"** extrapolates from n=1 corpus to unnamed systems. | Abstract + contribution 4 reframed: the corpus is widely-used so the failure mode is "a concern for models built the same way"; the *catastrophic* form is **corpus-specific** (§4.11); the **generalizable lesson** is "lexical safety scores can be invalidated by fine-tuning-induced register shift." Dropped unqualified "systemic". |
| **M5** | **Incumbent rubric asserted as field practice** without citation; strict-rubric ablation suggests strawman risk. | §3 Metric A reframed: it is the keyword-presence scheme **this system's own eval harness uses**, typical of ChatDoctor-lineage rubrics; explicitly *not* claimed universal; the point is the **metric family** (any keyword-presence safety axis) is fine-tunable (§4.2 strict variant still partially gamed). |
| **M6** | **"raw agreement 0.31 — below the 0.50 expected by chance"** is wrong; chance at the ft's marginals (kw-safe .90, esc .24) is **0.29**, not 0.50. | §4.3 → "0.31 is at the ≈0.29 chance level implied by these marginals (vs 0.83 for base)". |

## MINOR fixes APPLIED
- **§4.11** (#3): report llama2 base condition-sensitivity (6% none → 31–39% noisy/adv, parroted), note noisy McNemar non-sig (p=0.06); clarify "stable 54–57%" is Medllama2's. (#9): "confirming"→"consistent with"; note base/corpus/recipe co-vary so the contrast localizes but doesn't isolate corpus.
- **§4.8** (#5,#17): softened "decisively"→"on a judge-labeled sample", flagged emergency labels judge-assigned; **removed the un-reproducible "mean correctness 1.33/3; mean harm 1.25/2"** (not in any released artifact).
- **§4.4** (#18): "low in every specialty" (contradicted by 4/6=67%) → "low-to-moderate … max 4/6=67%, most ≤2/6".
- **§4.5** (#15): added "in the no-retrieval condition" to 33/72; added pooled 158/288=55%.
- **§5** (#14): base-validity qualified to the emergency-vignette benchmarks; note HealthBench base agreement 0.44 and llama2 base κ≤0.45 → lexical scoring imperfect everywhere, catastrophic only for the register-shifted ft.
- **Abstract** (#4): lead with the 55-point absolute gap, present 6.5× as the ratio-form.
- Every changed number re-verified against `judged_clean_s101.json` / `judged_orig_s101.json` / `healthbench_analysis.json` (75/40, 72/49, 43/27, 73/33, 158/288, chance 0.29, HB base 0.44).

## ⚠ TODO before submission (flagged)
1. **New §2 citations — VERIFIED real via web search (2026-07-12), add these exact entries to `comparison_matrix.md` + PMLR bib:**
   - Strathern, M. (1997). "'Improving ratings': audit in the British University system." *European Review* 5(3):305–321. (canonical Goodhart's-law formulation)
   - Amodei, D., Olah, C., Steinhardt, J., Christiano, P., Schulman, J., Mané, D. (2016). "Concrete Problems in AI Safety." arXiv:1606.06565.
   - Krakovna, V. et al. (2020). "Specification gaming: the flip side of AI ingenuity." DeepMind technical blog. (cite as tech report)
   - Pan, A., Bhatia, K., Steinhardt, J. (2022). "The Effects of Reward Misspecification: Mapping and Mitigating Misaligned Models." ICLR 2022.
   - Skalse, J., Howe, N., Krasheninnikov, D., Krueger, D. (2022). "Defining and Characterizing Reward Hacking." NeurIPS 2022. arXiv:2209.13085.
   - Geirhos, R. et al. (2020). "Shortcut learning in deep neural networks." *Nature Machine Intelligence* 2:665–673.
   No fabrication — all six independently confirmed to exist with these venue/year details.
2. Deferred (optional, per AC): add a *second* lexically-safe-but-under-triaging corpus to turn the n=1 catastrophe into a pattern; audit Medllama2's corpus to isolate the corpus factor; return the clinician anchor (Kevin).

## Net
All six majors and the genuine minors are closed from existing data; the paper now states the conditional/joint correctly, scopes equivalence to the valid test, engages the Goodhart canon, drops "systemic"/"decisively", and corrects the chance-level error. The corrections move numbers in the *un-flattering* direction (base conditional 14%→40%, equivalence weakened, systemic dropped) — exactly the credibility signal the AC said was highest-leverage.
