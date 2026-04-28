# OpenHealth — Research Upgrade Plan
## Target: JAIR-worthy paper on RAG robustness under noisy retrieval

---

## Core Research Question

> **"When and why does retrieval degrade LLM performance — and does LoRA fine-tuning make models more vulnerable?"**

This is novel because existing RAG literature almost exclusively studies when retrieval *helps*. The negative case — that retrieval can actively hurt, and that PEFT changes this sensitivity — is underexplored and directly measurable with our system.

---

## Paper Title (working)

**"When Retrieval Hurts: Context Sensitivity in PEFT-Adapted LLMs under Noisy and Adversarial Retrieval"**

---

## What We Already Have (Strong Foundation)

| Asset | Status |
|---|---|
| LoRA fine-tuned Llama 3 8B (rank=8, alpha=16) | ✅ Running |
| Hybrid RAG: BM25 + ChromaDB + RRF | ✅ Running |
| Relevance threshold (RRF ≥ 0.020) | ✅ Implemented |
| 8-case rubric evaluation (factual, tone, safety, conciseness) | ✅ Done |
| ROUGE eval: N=50, memorisation + generalisation splits | ✅ Done |
| Observed degradation under irrelevant context | ✅ Confirmed |
| Base model comparison | ✅ Done |

**Key existing finding:** Fine-tuned + RAG (7.25/8) > Fine-tuned alone (6.75/8) > Base (6.12/8) — but only with *relevant* context. Irrelevant context pulls results below fine-tuned-only baseline. That gap is the paper.

---

## Experiment Design

### Retrieval Conditions (3)

| Condition | Description | How to generate |
|---|---|---|
| **A — Clean** | Top-5 docs, RRF ≥ 0.020, verified relevant | Current pipeline, query-matched |
| **B — Noisy** | Random chunks from corpus, unrelated to query | Sample random rows from ChromaDB |
| **C — Adversarial** | Semantically similar but factually wrong/misleading | Retrieve docs for *different* medical condition with surface-similar keywords |

Adversarial is the strongest contribution. Example: for a chest pain query, inject documents about GERD (gastroesophageal reflux) that use similar symptom language but recommend no urgent care.

### Model Conditions (2 → ideally 3)

| Model | Description |
|---|---|
| **Base** | Llama 3 8B, no LoRA, no RAG context |
| **LoRA** | Fine-tuned on ChatDoctor-100K |
| **LoRA + RAG** | Fine-tuned + each retrieval condition above |
| **Base + RAG** | Base model + each retrieval condition above |

This gives us a 2×4 matrix (2 models × 4 context conditions: None, Clean, Noisy, Adversarial).

### Test Cases (target: 30 minimum, 50 ideal)

Current: 8 cases (too few for statistical claims). Need to expand to:
- 10 safety-required emergency cases
- 10 general medical queries (well-covered by corpus)
- 10 general medical queries (NOT covered by corpus — corpus gap cases)
- 10 edge cases (ambiguous, polypharmacy, paediatric)

### Metrics

| Metric | Why |
|---|---|
| Rubric score (0–8) | Clinical quality with safety axis |
| ROUGE-1/2/L | Overlap with reference answer |
| Safety escalation rate | Binary: did model correctly refer? |
| **Degradation rate** | How often RAG condition < No-RAG baseline |
| **Below-base rate** | How often RAG condition < base model (no fine-tuning, no RAG) |

The last two are the novel metrics. "Degradation rate" and "below-base rate" quantify exactly when retrieval hurts and how badly.

---

## The Key Graph (must produce)

```
Y-axis: Rubric Score (0–8)
X-axis: Retrieval Condition [None | Clean | Noisy | Adversarial]

Line 1: Base Llama 3 8B      → roughly flat, slight dip on adversarial
Line 2: LoRA fine-tuned      → starts higher, SHARP DROP on noisy/adversarial

Expected shape:
  Score
  7.5 |        * (LoRA + Clean)
  7.0 |  * (LoRA, no RAG)
  6.5 |                    (Base lines, relatively flat)
  6.0 |  * (Base)
  5.5 |            * (LoRA + Noisy)
  5.0 |                  * (LoRA + Adversarial)  ← KEY FINDING
      +-------------------------------------------
       None    Clean    Noisy   Adversarial
```

If this shape holds — LoRA makes the model *more* sensitive to context quality than the base model — that is the paper.

---

## Mechanism Hint (optional but makes it top-tier)

We don't need full mechanistic interpretability. We need one piece of evidence that points to *why* LoRA increases context sensitivity.

**Method:** Attention weight analysis
1. For each test case × condition, extract attention weights from layer 16 (middle integration layer)
2. Compute mean attention mass on context tokens vs. query tokens
3. Compare base vs. LoRA

**Expected finding:** LoRA fine-tuning increases mean attention to context tokens generally. When context is noisy/adversarial, that increased attention propagates wrong information. Base model attends to context less reliably — which is a weakness when context is good, but robustness when context is bad.

**Tools needed:** `transformer_lens` or direct `model.generate()` with `output_attentions=True` via llama.cpp Python bindings or HuggingFace transformers.

---

## Implementation Plan

### Phase 1 — Expand test set ✅ DONE
- [x] 100 test cases across 4 categories (25 safety, 35 general, 20 corpus_gap, 20 edge)
- [x] Each case tagged: safety_required, adversarial_query, noisy_query, expected_keywords
- [x] Clinically realistic, peer-review-defensible cases

### Phase 2 — Build experiment harness ✅ DONE
- [x] `research_experiment.py`: 2 models × 4 conditions × 100 cases (800 total runs)
- [x] Adversarial context baked in per-case (adversarial_query field)
- [x] Extended metrics: degradation_flag, below_base_flag, safety_failed
- [x] Crash-safe incremental saving to `research_results.json`
- [x] `analyze_research.py`: statistical analysis + 4 figures

### Phase 3 — Run experiments (next)
- [ ] Ensure ollama models loaded: `ollama run llama3` + `ollama run openhealth-doctor`
- [ ] Ensure retrieval service running on :5003
- [ ] Run: `python research_experiment.py`
- [ ] Expected: ~800 runs × ~30s each ≈ 6-8 hours on CPU (can resume if interrupted)

### Phase 4 — Analysis + graphs
- [ ] Run: `python analyze_research.py`
- [ ] Results saved to `results/` folder
- [ ] Figures: fig1_performance_vs_condition.png, fig2_category_breakdown.png, fig3 heatmap, fig4 boxplots
- [ ] Statistical tests: Mann-Whitney U on LoRA vs base degradation drops

### Phase 5 — Write paper (3-5 days)
- [ ] Abstract: state the question, the finding, the method
- [ ] Related work: RAG robustness, PEFT sensitivity, retrieval-augmented generation
- [ ] Method: system description, experiment design
- [ ] Results: key graph, tables, mechanism hint
- [ ] Discussion: why LoRA increases sensitivity (hypothesis), implications for production RAG systems
- [ ] Update `model_eval_report.html` with full findings

---

## Hypotheses to Test

1. **H1 (Primary):** LoRA fine-tuned models show greater performance degradation than base models under noisy/adversarial retrieval conditions.

2. **H2 (Secondary):** The degradation is worst in safety-critical cases — i.e., noisy context suppresses emergency escalation behaviour specifically.

3. **H3 (Mechanism):** LoRA increases mean attention weight on context tokens, explaining the increased sensitivity.

4. **H4 (Threshold):** The RRF ≥ 0.020 threshold partially mitigates degradation but does not eliminate it for adversarial context that scores above threshold.

---

## What Would Make This Rejection-Proof

1. **N ≥ 100 test cases** ✅ — 100 cases across 4 categories
2. **Statistical tests** ✅ — Mann-Whitney U, effect size r, significance stars in analyze_research.py
3. **At least one mechanism result** — attention analysis or token influence
4. **Honest limitations section** — 8B model only, single domain, single PEFT method
5. **Actionable recommendation** — what practitioners should do (our threshold approach as partial fix, what would fully fix it)

---

## Files to Create

| File | Purpose |
|---|---|
| `research_experiment.py` | Main 2×4 experiment runner |
| `adversarial_docs.py` | Adversarial context generator |
| `test_cases_extended.json` | 30+ annotated test cases |
| `analyze_research.py` | Graph + stats generator |
| `attention_analysis.py` | Lightweight mechanism hint |
| `research_results.json` | Raw experiment output |
| `research_report.html` | Final paper-style report |

---

## Timeline

| Week | Goal |
|---|---|
| Week 1 | Test cases + experiment harness built |
| Week 2 | Experiments running + attention analysis |
| Week 3 | Analysis + graphs + paper draft |
| Week 4 | Polish + submit |

---

## Current Status: READY TO RUN
Next action: Start services and run `python research_experiment.py` (800 evaluations, crash-safe)
