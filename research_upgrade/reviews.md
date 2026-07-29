# reviews.md — adversarial self-review of `OpenHealth_Research_Paper_vNext`
*Phase 6. Three independent reviewers + an area-chair meta-review, ML4H Proceedings standard. Deliberately harsh. Scores on ML4H's typical 1–5 (Reject / Weak Reject / Borderline / Weak Accept / Accept) with confidence 1–4.*

---

## Reviewer 1 (methods / ML-evaluation; confidence 4)

**Summary.** Argues a keyword "safety" rubric certifies a QLoRA-fine-tuned Llama-3-8B medical chatbot as safe while an independent LLM judge finds it under-triages 72–81% of emergencies; reports Cohen's κ between rubric and judge collapsing 0.43→0.05.

**Strengths.** Clean 2×4×100 design; the κ-collapse framing is crisp; honest negative result on the authors' own model; releases data and judgments.

**Weaknesses.**
- **κ is prevalence-sensitive.** The fine-tuned model emits escalation keywords on ~93% of answers, so the keyword "safe" label has near-zero variance; Cohen's κ is mechanically deflated under such marginal imbalance. The 0.43→0.05 drop may partly reflect base-rate change, not pure validity loss. Report PABAK or, better, lead with the prevalence-robust quantity P(keyword=safe ∧ judge=under-triage)=0.68 and de-emphasize κ.
- **The judge is the ground truth, and it's an LLM.** The entire inversion rests on one LLM judge. IRR with a second judge is marked PENDING and a human anchor is only PROPOSED. Without at least one of these *in the paper*, the central claim is "one LLM disagrees with a regex," which is not yet a safety result.
- **Possible judge confounds.** The fine-tuned outputs are terser and stylistically distinct; an LLM judge may penalize brevity/register. Need an ablation: does the judge's under-triage label survive length-matching or style-normalization?

**Questions.** What is κ/α with the second judge? Does P(under-triage) replicate on a length-matched subsample? **Score: 2 (Weak Reject) → 3 if IRR + a human subset are added.**

---

## Reviewer 2 (clinical NLP / health-ML; confidence 3)

**Summary.** As above; emphasizes triage and retrieval robustness.

**Strengths.** Triage is the right dependent variable; the qualitative emergency examples (serotonin syndrome, CO poisoning, hypertensive emergency) are compelling and clinically literate; the detection framing (sensitivity/specificity) is appropriate and rarely done in this sub-area.

**Weaknesses.**
- **Single model, single corpus, single PEFT recipe.** The title and abstract generalize ("Fine-Tuned Medical LMs"), but the evidence is n=1 model. Either soften the claim or add ≥1 more (base model or PEFT method).
- **Team-authored benchmark, n=100, English.** No external validity. A HealthBench emergency-referral subset is *available* and would directly rebut this; its omission is conspicuous given the authors cite it.
- **Is the keyword rubric a strawman?** Reviewers will object that "serious" evaluations already use LLM judges (HealthBench, MedHELM). The paper must show that lexical safety rubrics are *actually in use* for systems of this class (cite specific local-medical-LLM papers/repos that score safety lexically), or the contribution reads as "don't use a metric nobody defends."
- **Absolute performance.** Both models are poor at escalation; the base model's 0.16 sensitivity with no context is itself alarming. The framing should avoid implying the base model is safe.

**Questions.** Can you replicate on HealthBench cases? Which deployed/published systems use the lexical rubric you critique? **Score: 3 (Borderline).**

---

## Reviewer 3 (skeptical generalist; confidence 3)

**Summary.** "Fine-tuning games a bad metric."

**Strengths.** Reproducible; honest; the policy implication (semantic, multi-judge, sensitivity/specificity eval) is sound and actionable.

**Weaknesses.**
- **Novelty vs. the known.** That LLM-judges beat lexical metrics and that models can reward-hack metrics are both established. The novel kernel here is the *clinical-safety, fine-tuning-induced* instance with a quantified validity collapse — but the paper must position that kernel sharply against G-Eval/MT-Bench (judge>lexical) and reward-hacking literature, or it reads as a domain restatement.
- **Confound: is it the metric or the model?** The paper shows both that (a) the metric is invalid for FT and (b) FT is worse than base. These are entangled. A cleaner separation: hold the model fixed and show the metric mislabels; hold the metric fixed (use the judge) and show FT<base. The paper does both but should state the two claims separately and not let the dramatic κ number do double duty.
- **Stats hygiene.** Holm correction is marked PARTIAL; single seed; CIs are bootstrap on one decoding sample, so they understate generation variance.

**Questions.** Multi-seed? Holm-corrected family? **Score: 3 (Borderline).**

---

## Area-Chair meta-review (confidence 4)

**Recommendation: Borderline (lean accept) for ML4H Proceedings, conditional on two additions.** The paper has a genuine, clinically-consequential, well-evidenced core: a fine-tuned local medical LLM that an in-use style of safety rubric certifies as safe while it under-triages most emergencies, with a quantified divergence between metric and clinical judgment. That is exactly the cautionary, methods-aware contribution ML4H values, and the authors' willingness to indict their own system is a credibility asset.

Two things gate acceptance, and both are achievable:
1. **Judge credibility must be *in* the paper, not proposed.** Add (a) the second independent judge's IRR (κ/α) and (b) a small clinician-rated anchor (even n≈30–40). Without these the inversion rests on a single LLM.
2. **Scope the generalization honestly.** Either add a second model/PEFT setting or retitle to a case study ("a fine-tuned Llama-3-8B medical assistant") and soften abstract claims.

Secondary but expected: address κ's prevalence sensitivity (lead with the 0.68 conditional probability), a length/style-matched judge robustness check, multi-seed for the headline, Holm correction, and one external (HealthBench) replication. With #1 and #2 done and the secondary items addressed, this is a clear **Weak Accept (3.5–4)**; as-is it is a **Borderline (2.5–3)** that strong reviewers could push either way.

**Current realistic outcome: borderline. With the two gating fixes: weak accept.**
