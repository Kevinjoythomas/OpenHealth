# Comparison Matrix: Positioning OpenHealth Against Verified Prior Work

OpenHealth contributes a controlled study of whether parameter-efficient fine-tuning (QLoRA) of a locally-deployable medical LLM (LLaMA-3-8B-Instruct) makes retrieval-augmented generation more or less fragile under clean, noisy, and adversarial context, evaluated on emergency safety-referral behavior. The work sits at the intersection of five literatures that each examine one facet of this question in isolation: PEFT stability vs. plasticity (does fine-tuning forget?), RAG robustness to noise and poisoning (does retrieval break the reader?), medical LLM safety evaluation (do clinical models behave safely under stress?), LLM-as-judge reliability (can we trust the grader?), and medical QA benchmarks / local clinical LLMs (what is the accuracy ceiling and the open small-model baseline?). The table below maps the most relevant verified papers onto a shared schema, then positions OpenHealth on the same axes. Only papers in the verified list are included; no other work is introduced.

## Comparison table

### Theme 1 — PEFT robustness / forgetting (stability–plasticity)

| Paper (year, venue) | Problem framing | Method | Datasets | Baselines | Metrics | Key results |
|---|---|---|---|---|---|---|
| PEFT-Arena (2026, arXiv cs.LG, May 2026) | PEFT eval overweights downstream accuracy, ignores retention of pretrained capability | Benchmark measuring adaptation (plasticity) vs. forgetting (stability) at matched param budgets; weight- and activation-space analysis; checkpoint "rewinding" case study | Trained: ~50k OpenR1-Math, ~23k MedThink; eval: math500/amc23/aime24, med_eval; retention: BBH, IFEval, NQ, MMLU, GSM8K, etc. | LoRA, AdaLoRA, DoRA, VeRA, PiSSA, MiLoRA, OFT variants, IA3, Full FT | Joint adaptation + general-capability-retention (stability–plasticity profile) | At comparable budgets, orthogonal finetuning (OFT) reaches the most favorable retention-vs-adaptation Pareto frontier (illustrative figures unconfirmed; treat numbers cautiously) |
| LoRA Learns Less and Forgets Less (2024, TMLR, Featured Certification) | Core stability–plasticity tradeoff of PEFT as a regularizer | Controlled LoRA vs. full FT on code/math; instruction tuning (~100K pairs) and continued pretraining (~20B tokens); rank/diversity analysis | Code + math target domains; ~100K instruction pairs; ~20B-token corpus; standard code/math + general benchmarks | Full fine-tuning; weight decay; dropout | In-domain accuracy, out-of-domain retention, effective rank, generation diversity | LoRA underperforms full FT in-domain but forgets less; full FT learns perturbations of rank 10–100× greater; LoRA mitigates forgetting more than weight decay/dropout |
| DoRA (2024, ICML Oral) | Can PEFT recover full-FT learning capacity? | Decompose weights into magnitude + direction; LoRA on directional component; merge at inference (zero added latency) | LLaMA (commonsense), LLaVA (visual instruction), VL-BART (image/video-text) | LoRA at matched param budgets | Downstream accuracy across language + vision-language | Consistently outperforms LoRA across tasks with no inference overhead (per-benchmark deltas not extracted) |
| PiSSA (2024, NeurIPS Spotlight; arXiv 2404.02948) | Better-initialized low-rank PEFT for faster/higher convergence | SVD init of LoRA matrices on principal singular components; frozen residual; 4-bit QPiSSA variant | 11 models (184M–70B); 5 NLG + 8 NLU tasks incl. GSM8K | LoRA, QLoRA (identical setup, different init) | Downstream accuracy (e.g., GSM8K) | GSM8K Gemma-7B 77.7% vs LoRA 74.53% (+3.25); LLaMA-3-70B QPiSSA 86.05% vs QLoRA 81.73% |
| Orthogonal Finetuning Made Scalable / OFTv2 (2025, EMNLP Main) | OFT resists forgetting by spectrum-preserving rotations but is too slow/memory-heavy | Input-centric matrix-free reformulation (quadratic vs cubic); Cayley–Neumann parameterization; quantized training | LLM fine-tuning settings vs QLoRA (full benchmark list not confirmed) | Original OFT; QLoRA | Training stability/efficiency + downstream performance | Up to 10× faster, ~3× lower GPU memory than original OFT; outperforms QLoRA in stability/efficiency/memory for quantized FT |
| Empirical Study of Catastrophic Forgetting in LLMs (2023; IEEE TASLP 2025) | Systematic measurement of forgetting during continual instruction tuning | Continual FT across 1B–7B; decoder-only (BLOOMZ) vs encoder-decoder (mT0); domain/reasoning/reading probes; bias shifts | Domain-knowledge, reasoning, reading-comprehension probes; bias eval sets | Pre- vs post-tuning capability | Forgetting magnitude by scale/architecture | Forgetting is general and intensifies with scale (1B→7B); BLOOMZ forgets less than mT0; general instruction tuning mitigates later forgetting |

### Theme 2 — RAG robustness to noise / adversarial retrieval

| Paper (year, venue) | Problem framing | Method | Datasets | Baselines | Metrics | Key results |
|---|---|---|---|---|---|---|
| The Power of Noise (2024, SIGIR, pp. 719–729) | How retrieved-set composition (relevance, position, count) affects RAG accuracy | Controlled prompt-composition study: relevant vs related-but-irrelevant vs random docs; doc count; gold-passage position | Natural Questions (open-domain QA) | Fixed retriever + several LLM readers; no-noise prompts | Answer accuracy (exact-match-style) | Random/noise documents can improve accuracy up to ~35%; high-ranked but non-relevant ("distracting") docs degrade it; position matters |
| Making RALMs Robust to Irrelevant Context (2024, ICLR) | RALMs degrade sharply on irrelevant retrieved passages | (1) NLI entailment filter with no-retrieval fallback; (2) fine-tune on auto-generated relevant+irrelevant mixes (~1,000 examples) | NQ, 2WikiMQA, Bamboogle, StrategyQA, Fermi | Prompted RALM variants; no-retrieval | Exact Match / F1 | Random contexts drop accuracy on all five sets; SA-RetRobust stays within ~1 std of no-retrieval under random context and gains ~3–4 pts with top-1; robustness with only ~1k examples |
| PoisonedRAG (2025, USENIX Security; arXiv Feb 2024) | First knowledge-poisoning attack on RAG corpora | Inject 5 crafted passages per target question; joint retrieval+generation optimization; black- and white-box; tests existing defenses | NQ, HotpotQA, MS-MARCO (million-scale corpora) | RAG without injection; existing defenses | Attack Success Rate (ASR) | ~90% ASR injecting 5 texts; ~97% NQ, ~99% HotpotQA, ~91% MS-MARCO (PaLM 2, black-box); defenses insufficient |
| Self-RAG (2024, ICLR Oral) | Don't retrieve blindly — decide on-demand and critique | Reflection tokens (Retrieve, IsRel, IsSup, IsUse) distilled from a critic; inference-time tree decoding weighted by token probs | PopQA, TriviaQA, ARC, PubHealth, FEVER, ASQA, biography | ChatGPT; retrieval-augmented Llama2-chat | Accuracy, FactScore, citation precision/recall | Self-RAG 7B/13B outperform ChatGPT and RA-Llama2-chat; large factuality/citation gains on long-form (per-task numbers not re-extracted) |
| CRAG — Corrective RAG (2024, arXiv Jan 2024; ICLR'25 submission withdrawn — cite arXiv) | Self-correct when retrieval is wrong | Lightweight T5 retrieval evaluator → Correct/Incorrect/Ambiguous actions; decompose-recompose; web-search fallback; plug-and-play on RAG/Self-RAG | PopQA, Biography, PubHealth, Arc-Challenge | Standard RAG; Self-RAG | Accuracy, FactScore | Over standard RAG: +19.0% PopQA, +14.9 FactScore Biography, +36.6% PubHealth, +8.1% Arc-Challenge |
| RAAT — Adaptive Adversarial Training (2024, ACL Long, 2024.acl-long.540) | Train generator to resist three real-world noise types | Adaptive adversarial training (worst-case noise per step) + multi-task noise-type classifier; built on LLaMA-2-7B; releases RAG-Bench | RAG-Bench (NQ, TriviaQA, WebQ) under relevant/irrelevant/counterfactual noise | RALM-multiple and other RALM baselines | F1, Exact Match (clean + noisy) | RAAT vs best baseline ~+1.5–3.0 F1 / +1.5–3.5 EM across noise types (e.g., counterfactual 86.29 F1 vs 83.25) |
| RAG Robustness to Adversarial Evidence in Health (2025, arXiv Sep 2025) | How RAG behaves with deliberately misleading **medical** evidence | Helpful/harmful/adversarial doc pools across 4 setups; 6 attack variants (Rewriter, Paraphraser, Fact-Inversion, FSAP-IntraQ/InterQ, Liar); 3 query framings | TREC 2020 & 2021 Health Misinformation Tracks (22 COVID-19 + 27 medical-treatment queries) | Non-RAG baseline; helpful-only pools | Ground-Truth Alignment Rate (stance accuracy) | "Liar" docs (GPT-4.1, TREC 2021) drop alignment 88.9% → ~4.4%; helpful evidence in the pool largely restores robustness; more resilient to COVID-19 than general health misinformation |

### Theme 3 — Medical LLM safety evaluation

| Paper (year, venue) | Problem framing | Method | Datasets | Baselines | Metrics | Key results |
|---|---|---|---|---|---|---|
| HealthBench (2025, arXiv OpenAI report) | Performance + safety in realistic open-ended multi-turn health chat, incl. emergency referrals | 5,000 conversations graded against 48,562 physician-authored rubric criteria; GPT-4.1 model grader; 7 themes incl. emergency referrals; Consensus + Hard variants | HealthBench corpus (+Consensus, +Hard) | GPT-3.5, GPT-4o, o3, GPT-4.1 nano | Rubric-criterion satisfaction (% weighted points); grader–physician agreement | GPT-3.5 16%, GPT-4o 32%, o3 60% overall; Hard top 32%; 262 physicians; explicit emergency-referral axis penalizing under- and over-escalation |
| RAG LLMs are Not Safer (2025, NAACL Long, 2025.naacl-long.281) | Does adding retrieval make LLMs **less** safe? | Safety stress-test of 11 LLMs on 5,592 harmful questions, non-RAG vs RAG; decompose into model/document/RAG-capability effects | 5,592 harmful questions; general-domain retrieved corpora | Non-RAG setting; 11 LLMs incl. GPT-4o, Claude-3.5-Sonnet | Unsafe-response rate; safe- vs unsafe-doc attribution | Llama-3-8B unsafe rate 0.3% → 9.2% with RAG (~30×); 8/11 models worse under RAG; 5.3% docs unsafe yet 81.8% of unsafe Llama-3-8B responses came from **safe** docs; Claude-3.5-Sonnet most robust |
| DAS — Dynamic Red-Teaming Agents (2025–26, arXiv 2508.00923) | Static medical-benchmark scores collapse under adaptive stress ("benchmarking gap") | Autonomous agents apply mutation tools (Answer Negation, Narrative Distraction, Cognitive Bait, Physiological Impossibility) over escalating rounds; multi-agent hallucination detector | MedQA; HealthBench (192 vignettes); 81 privacy scenarios; 415 bias items; 260 hallucination samples; 16 LLMs | GPT-4o, Claude Sonnet-4, DeepSeek-R1, MedGemma, Gemini, Llama | Failure/jailbreak rate; privacy leak; bias rate; hallucination rate; detector F1 | ~94% of initially-correct MedQA answers fail under perturbation; HealthBench jailbreak 76% (tier-1); privacy 86%→91%; hallucination ~74%; detector accuracy 81.9% (F1 81.5%) |
| Red Teaming ChatGPT in Medicine (2025, npj Digital Medicine) | Independent human red-teaming of clinical LLMs across harm axes | 80 participants stress-test GPT-3.5/4(+internet) on real cases; expert adjudication; release public red-team dataset | 376 unique adversarial prompts → 1,504 responses (Red-Teaming-Dataset) | GPT-3.5 vs GPT-4 vs GPT-4+internet (vs GPT-4o) | % responses judged inappropriate, by harm axis | 20.1% inappropriate overall (GPT-3.5 25.8%, GPT-4 16%, GPT-4+internet 17.8%); 21.5% appropriate in GPT-3.5 but inappropriate in updated models (version regressions) |
| Knowing When to Abstain — MedAbstain (2026, EACL Main) | Clinical safety as knowing when **not** to answer | Conformal prediction + adversarial perturbations + explicit abstention options on medical MCQA; compares size/prompting/abstain interventions | Medical MCQA with adversarial perturbations + abstention (MedAbstain) | Model-size scaling; advanced prompting | Abstention rate / accuracy-abstention tradeoff; calibration; conformal coverage | Explicit abstention options increase safe abstention "far more than input perturbations"; even high-accuracy models often fail to abstain when uncertain |

### Theme 4 — LLM-as-judge reliability

| Paper (year, venue) | Problem framing | Method | Datasets | Baselines | Metrics | Key results |
|---|---|---|---|---|---|---|
| G-Eval (2023, EMNLP Main) | Replace low-correlation reference metrics with LLM judging | GPT-4 + chain-of-thought + form-filling rubric; probability-weighted score to de-discretize | SummEval, Topical-Chat (dialogue) | BLEU, ROUGE, prior NLG metrics | Spearman / Kendall-Tau vs human ratings | G-Eval-4 Spearman 0.514 on summarization, beating prior metrics by a large margin; authors flag LLM-judge bias toward LLM-generated text |
| MT-Bench & Chatbot Arena (2023, NeurIPS D&B) | Validate LLM-as-judge and catalogue its biases | MT-Bench (multi-turn) + Chatbot Arena (crowd pairwise); GPT-4 judge; mitigations (swap-order, few-shot, reference-guided) | MT-Bench, 3K expert votes, 30K Arena conversations | Human–human agreement | Judge–human agreement %; bias win-rate deltas | GPT-4 judge >80% agreement (≈ human–human); self-enhancement: GPT-4 ~+10%, Claude-v1 ~+25% on own answers |
| Justice or Prejudice? / CALM (2024 arXiv; ICLR 2025) | Quantify a broad taxonomy of judge biases | CALM: automated, principle-guided, label-free perturbation framework measuring 12 bias types | Multiple general-domain judging tasks in the CALM harness | Strong judge models | Robustness/attack-success rate per bias type | 12 bias types quantified; significant biases persist even in advanced judges on specific tasks |
| Judging the Judges — Position Bias (2024 arXiv; AACL-IJCNLP 2025) | Isolate and characterize position bias | 15 judges over MTBench + DevBench (22 tasks, ~40 generators); >150k instances; pairwise + list-wise | MTBench, DevBench | 15 LLM judges compared | Repetition stability, position consistency, preference fairness | Position bias is systematic (not random), varies by judge/task; solution-quality gap is the dominant driver; prompt length weak |
| Replacing Judges with Juries — PoLL (2024, arXiv Apr 2024) | Replace a single large judge with a diverse panel | Three small disjoint-family models (Command-R, GPT-3.5-turbo, Claude Haiku) vote independently; max-vote/average aggregation | Six datasets across 3 judge settings (single-point, pairwise, QA/Arena-style) | Single GPT-4 judge | Correlation with humans; intra-model bias; cost | PoLL correlates better with humans than a single GPT-4 judge, >7× cheaper, with less intra-model (self-preference) bias |
| Evaluating Clinical AI Summaries with LLM Judges (2025, npj Digital Medicine) | Validate LLM-as-judge against a physician-validated clinical rubric | Zero/few-shot, SFT, DPO, multi-agent judges vs PDSQI-9 (9 attributes); 7 physician raters; MIMIC-III cross-validation | 200 UW Health summaries (160/40); 31-summary MIMIC-III/ProbSum; 8000+ rated items | 7 physician raters; reference metrics | ICC, Krippendorff's alpha, median diff, time, cost | Best judge (GPT-o3-mini) ICC 0.818 (95% CI 0.772–0.854), alpha 0.677; 22 s/eval vs ~600 s (~96% faster) at ~$0.05; PDSQI-9 ICC 0.867 |
| CQA-Eval (2025, arXiv Oct 2025; rev. 2026; prior title LongQAEval) | Reliable physician evaluation of long-form clinical QA under resource constraints | Physician annotation of 300 patient questions (physician + LLM answers); coarse (answer-level) vs fine-grained (sentence-level); subset-reliability analysis | 300 real patient questions, physician-annotated | Answer-level vs sentence-level annotation | Inter-annotator agreement by dimension; cost vs reliability | Fine-grained improves correctness IAA, coarse improves relevance IAA; risk/safety-disclosure judgments remain inconsistent; small sentence subset ≈ coarse reliability |

### Theme 5 — Medical QA benchmarks / local & open clinical LLMs

| Paper (year, venue) | Problem framing | Method | Datasets | Baselines | Metrics | Key results |
|---|---|---|---|---|---|---|
| Large Language Models Encode Clinical Knowledge / Med-PaLM (2023, Nature 620) | Can LLMs reach SOTA on medical MCQA, and where do they fail on safety? | Flan-PaLM (540B) with few-shot/CoT/self-consistency; Med-PaLM via instruction prompt tuning; rubric human-eval | MultiMedQA: MedQA, MedMCQA, PubMedQA, MMLU clinical, LiveQA, MedicationQA, HealthSearchQA | Prior MCQA SOTA; clinicians | MCQA accuracy; human axes (consensus, factuality, harm, bias, completeness) | Flan-PaLM 67.6% MedQA (USMLE), >17 pts over prior SOTA, clears ~60% pass mark; Med-PaLM "remains inferior to clinicians" |
| Towards Expert-Level Medical QA / Med-PaLM 2 (2023 arXiv; Nature Medicine 2024) | Push medical QA to expert level; argue MCQA is saturating | PaLM 2 + medical FT + "ensemble refinement" prompting; pairwise physician/lay preference on long-form | MedQA, MedMCQA, PubMedQA, MMLU clinical; long-form consumer questions | Med-PaLM; physician-written answers | MCQA accuracy; pairwise preference over 9 axes | 86.5% MedQA (>19 pts over Med-PaLM 67.2%); physicians preferred Med-PaLM 2 long-form over physician answers on 8/9 axes (p<0.001) |
| MEDITRON-70B (2023, arXiv Nov 2023; weights + corpus open) | Open medical LLM narrowing the gap to closed models | Continued pretraining of Llama-2 (7B/70B) on GAP-Replay corpus (PubMed + guidelines + replay); SFT + self-consistency CoT | MedQA, MedMCQA, PubMedQA, MMLU medical | Llama-2-70B; GPT-3.5; Med-PaLM; GPT-4; Med-PaLM 2 | MCQA accuracy | MEDITRON-70B 70.2% MedQA, 72.0% avg; ~6 pts over best public same-class baseline; beats GPT-3.5/Med-PaLM; within ~5% of GPT-4, ~10% of Med-PaLM 2 |
| BioMistral-7B (2024, arXiv Feb 2024; Findings of ACL 2024) | Open, deployment-minded 7B medical LLM with multilingual eval | Continued PubMed Central pretraining on Mistral-7B-Instruct; SLERP/TIES/DARE merging; quantization; 7-language auto-translated suite | 10 English medical QA (MedQA 4/5-opt, MedMCQA, PubMedQA, MMLU medical) + translations | MediTron-7B; GPT-3.5 Turbo | Few-shot (3-shot) + SFT accuracy; multilingual | BioMistral-7B 10-task avg 50.3% (MedQA-4opt 37.4, MedMCQA 50.3, MMLU-med 60.9) vs MediTron-7B 38.2, GPT-3.5 66.0; merged/SFT ~57–59% |
| ChatDoctor (2023, arXiv Mar 2023; Cureus 15(6)) | Conversational medical assistant fine-tuned on real doctor-patient dialogue | SFT of LLaMA on ~100k HealthCareMagic dialogues (after Alpaca 52k); self-directed Wikipedia + disease-DB retrieval at inference | HealthCareMagic-100k (train); iCliniq-10k (eval); Wikipedia + disease DBs | ChatGPT (reference physician answers) | BERTScore Precision / Recall / F1 | ChatDoctor 0.8444 P / 0.8451 R / 0.8446 F1 vs ChatGPT 0.837 / 0.8445 / 0.8406 (P and F1 significant p<0.001; Recall n.s.) |

### OpenHealth (ours) — same schema

| Paper (year, venue) | Problem framing | Method | Datasets | Baselines | Metrics | Key results (target) |
|---|---|---|---|---|---|---|
| **OpenHealth (this work, ML4H 2026 target; CHIL 2027 backup)** | Does PEFT of a locally-deployable medical LLM increase RAG fragility on emergency safety-referral behavior? Refutes the "PEFT makes RAG more fragile" hypothesis | LLaMA-3-8B-Instruct + 4-bit QLoRA (r=8, α=16, 20.97M = 0.26% params, 1000 steps) on 112k ChatDoctor pairs; hybrid BM25 + nomic-embed dense over ChromaDB (6 medical PDFs), RRF k=60, relevance gate τ=0.020; runs on RTX 3050 8GB. Controlled **2×4** design (2 models {base Llama-3, LoRA} × 4 retrieval conditions {no-context, clean, noisy, adversarial} × 100 clinically-annotated cases = 800 runs). Multi-judge LLM rubric (correctness / harm / triage-appropriateness / calibration) with reported inter-rater reliability | 112k ChatDoctor doctor-patient pairs (train); 100 clinically-annotated emergency cases (eval); 6 medical PDFs (retrieval corpus) | Base LLaMA-3-8B-Instruct (same model, no LoRA) × {no-context, clean, noisy, adversarial}; keyword-presence auto-rubric vs multi-judge rubric | Safety-referral correctness; harm; triage-appropriateness; calibration; inter-rater reliability; robustness deltas across retrieval conditions | Central claim: QLoRA yields a large, retrieval-robust improvement in emergency safety-referral behavior and does **not** add fragility under noisy/adversarial retrieval (PEFT-makes-RAG-fragile hypothesis refuted). Methods finding: keyword auto-rubrics **invert** the safety conclusion under perturbation; multi-judge rubric re-establishes it |

## SOTA baselines reviewers will expect

Concrete models/methods a strong ML4H/CHIL reviewer will expect OpenHealth to compare against or explicitly justify omitting:

**PEFT method baselines (matched param-budget comparisons):**
- Full fine-tuning (the plasticity ceiling and forgetting upper-bound — *LoRA Learns Less and Forgets Less*)
- Plain LoRA (the direct precursor to QLoRA used here)
- QLoRA (4-bit) — OpenHealth's own setting; baseline is the non-fine-tuned LLaMA-3-8B-Instruct
- PiSSA / QPiSSA (strongest "better-initialized 4-bit PEFT"; LLaMA-3-70B QPiSSA 86.05% on GSM8K)
- DoRA (full-FT-recovering low-rank baseline)
- OFT / OFTv2 (forgetting-resistant, spectrum-preserving alternative to QLoRA)

**RAG-robustness baselines:**
- SA-RetRobust (train-on-noise robustness; *Making RALMs Robust to Irrelevant Context*)
- RAAT (adaptive adversarial training with the relevant/irrelevant/counterfactual taxonomy)
- Self-RAG (selective/on-demand retrieval with reflection tokens)
- CRAG (retrieval-evaluator + Correct/Incorrect/Ambiguous correction — OpenHealth's τ-gate is its offline minimalist analogue)
- PoisonedRAG attack (the adversarial-injection threat model for the "adversarial" arm)

**Medical LLM baselines:**
- Med-PaLM / Med-PaLM 2 (closed-model MCQA SOTA ceiling: 67.6% / 86.5% MedQA)
- MEDITRON-70B (open medical LLM reference: 70.2% MedQA)
- BioMistral-7B (closest size-class open peer: ~50.3% 10-task avg)
- ChatDoctor (the training-data source model; BERTScore-based, no clinician safety review)

**Evaluation / judge baselines:**
- Keyword-presence auto-rubric (OpenHealth's own negative control that inverts under perturbation)
- Single GPT-4 / G-Eval-style judge (the brittle single-judge baseline to beat)
- PoLL panel-of-judges and MedHELM LLM-jury (the multi-judge / jury alternatives OpenHealth's rubric instantiates)

> Note on MedHELM: a clinician-validated holistic medical-evaluation framework (5 categories, 22 subcategories, 121 tasks, 35-benchmark suite, LLM-jury ICC 0.47 vs clinician–clinician 0.43) is referenced in the verified set as a reliability-reporting precedent for the multi-judge rubric, but is **not** a model baseline.

## Standard benchmark datasets reviewers will expect

Datasets a reviewer will expect a medical-LLM safety paper to engage with, with explicit notes on what OpenHealth currently **lacks**:

- **MedQA (USMLE)** — the canonical medical MCQA benchmark. **OpenHealth LACKS** any MedQA evaluation (no MCQA accuracy reported).
- **MedMCQA** — large-scale medical MCQA. **OpenHealth LACKS** this.
- **PubMedQA** — biomedical-literature QA. **OpenHealth LACKS** this.
- **MMLU clinical/medical subsets** — general-capability + medical knowledge retention probe. **OpenHealth LACKS** this (and, given the PEFT-Arena/forgetting framing, MMLU-style retention probes are the obvious way to substantiate "no capability loss").
- **MultiMedQA suite** (MedQA, MedMCQA, PubMedQA, MMLU clinical, LiveQA, MedicationQA, HealthSearchQA) — the standard composite. **OpenHealth LACKS** all components.
- **HealthBench** (and HealthBench Consensus / Hard) — the rubric-based open-ended health benchmark with an explicit emergency-referral theme that most directly mirrors OpenHealth's safety-referral focus. **OpenHealth LACKS** HealthBench evaluation; using its emergency-referral cases or rubric format would substantially strengthen external validity.
- **TREC 2020/2021 Health Misinformation Tracks** — the adversarial-medical-evidence testbed used by the closest prior work. **OpenHealth LACKS** this; it uses its own 100 clinically-annotated cases instead.
- **RAG-Bench** (relevant/irrelevant/counterfactual noise taxonomy from RAAT) — **OpenHealth LACKS** this standardized noise benchmark; its noisy/adversarial arms are bespoke.
- **Natural Questions / HotpotQA / MS-MARCO** — the standard open-domain RAG-robustness and poisoning testbeds (Power of Noise, PoisonedRAG). **OpenHealth LACKS** these; not medical, so omission is defensible but should be acknowledged.
- **iCliniq-10k / HealthCareMagic-100k** — ChatDoctor's dialogue datasets. OpenHealth **HAS** the 112k ChatDoctor training pairs but does **not** evaluate on iCliniq.
- **MIMIC-III / ProbSum** — used as a cross-validation set in the clinical-judge-reliability precedent. **OpenHealth LACKS** any MIMIC-based evaluation.

**Datasets OpenHealth HAS:** 112k ChatDoctor doctor-patient pairs (training); a 6-PDF ChromaDB retrieval corpus; and a bespoke set of 100 clinically-annotated emergency cases under four retrieval conditions (800 runs). The gap reviewers will flag most sharply is the absence of any **standard MCQA retention benchmark (MedQA/MMLU)** to back the "no added fragility / preserved capability" claim, and the absence of **HealthBench**, whose emergency-referral axis is the closest external analogue to OpenHealth's central metric.

## Where OpenHealth genuinely differs / contributes

- **A controlled causal test of the PEFT-fragility hypothesis on the same base model.** *RAG LLMs are Not Safer* established that retrieval makes Llama-3-8B markedly less safe (0.3% → 9.2% unsafe; ~30×) in the **general** domain on harmful-query compliance. OpenHealth runs the analogous controlled study on the **same architecture** but for **clinical emergency safety-referral**, and across a clean 2×4 (model × retrieval-condition) grid. No verified paper isolates whether *task PEFT itself* (not adversarial training) changes RAG fragility — RAAT and SA-RetRobust both *train on noise* to gain robustness, whereas OpenHealth tests ordinary QLoRA with no robustness-specific training and refutes the fragility hypothesis.
- **Emergency safety-referral as the dependent variable, under retrieval perturbation.** HealthBench defines the emergency-referral axis but evaluates static conversations; DAS perturbs but does not isolate under-triage; the health-domain adversarial-RAG paper measures stance alignment, not triage escalation. OpenHealth's combination — triage-appropriateness scored across no-context/clean/noisy/adversarial retrieval — is not covered by any single verified paper.
- **A methods result that the choice of rubric inverts the safety conclusion.** OpenHealth shows keyword-presence auto-rubrics **flip** the safety verdict under retrieval perturbation, then re-establishes the result with a multi-axis multi-judge rubric. This operationalizes, in a medical-safety setting, the general judge-fragility findings of G-Eval (self-preference), MT-Bench/Chatbot Arena (position/self-enhancement), CALM (12 bias types), and PoLL (juries beat single judges), and the clinical reliability bar set by the npj summaries paper (ICC 0.818) and CQA-Eval (risk-disclosure judgments are the least reliable dimension).
- **Genuinely local deployment.** Unlike Med-PaLM 2 (closed, 540B-class), MEDITRON-70B (70B, beyond consumer hardware), or the GPT-4-grader pipelines, OpenHealth's entire stack (QLoRA 8B + hybrid retrieval + relevance gate) runs on an **RTX 3050 8GB**, with a training-free τ-gate as the offline analogue of CRAG/Self-RAG's learned selective retrieval.

## Where OpenHealth is behind or untested

- **No standard MCQA benchmarks.** Without MedQA / MedMCQA / PubMedQA / MMLU numbers, OpenHealth cannot place itself on the leaderboard reviewers anchor to (Med-PaLM 67.6%, Med-PaLM 2 86.5%, MEDITRON-70B 70.2%, BioMistral-7B ~50.3%), nor quantitatively support "no capability loss" the way MMLU/BBH/IFEval retention probes do in PEFT-Arena and the forgetting literature.
- **No comparison to robustness-trained baselines.** OpenHealth tests plain QLoRA but does not compare against SA-RetRobust or RAAT, so it cannot say whether ordinary PEFT matches or trails methods explicitly trained for noise robustness.
- **Bespoke, small evaluation set.** 100 clinically-annotated cases (vs HealthBench's 5,000 conversations / 48,562 rubric criteria, DAS's multi-set roster, or TREC's tracks) limits statistical power and external comparability; CQA-Eval and the npj study both warn that risk/safety-disclosure judgments are the hardest to score reliably — exactly OpenHealth's target dimension.
- **Single base model, single PEFT method.** No DoRA/PiSSA/OFT comparison and no second base model, so the stability–plasticity claim is demonstrated for one point in the design space PEFT-Arena maps.
- **Judge-reliability evidence must clear a published bar.** To be credible, OpenHealth's inter-rater reliability needs to be reported with ICC/Krippendorff's alpha against the ~0.82 ICC / 0.677 alpha precedent, and its panel must address PoLL's "correlated-errors" caveat (a jury of similar models is not automatically independent).
- **Adversarial arm narrower than the literature's threat models.** PoisonedRAG (corpus poisoning) and the six health-domain attack variants (Rewriter/Paraphraser/Fact-Inversion/FSAP/Liar) are broader than OpenHealth's single bespoke adversarial condition.

## Citations used

- PEFT-Arena: Understanding Parameter-Efficient Finetuning from a Stability-Plasticity Perspective — https://arxiv.org/abs/2605.28819
- LoRA Learns Less and Forgets Less — https://arxiv.org/abs/2405.09673
- DoRA: Weight-Decomposed Low-Rank Adaptation — https://arxiv.org/abs/2402.09353
- PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models — https://openreview.net/forum?id=6ZBHIEtdP4
- Orthogonal Finetuning Made Scalable — https://arxiv.org/abs/2506.19847
- An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning — https://arxiv.org/abs/2308.08747
- Integrating Fine-Tuning and Retrieval-Augmented Generation for Healthcare AI Systems: A Scoping Review — https://pmc.ncbi.nlm.nih.gov/articles/PMC12938813/
- The Power of Noise: Redefining Retrieval for RAG Systems — https://arxiv.org/abs/2401.14887
- Making Retrieval-Augmented Language Models Robust to Irrelevant Context — https://arxiv.org/abs/2310.01558
- PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language Models — https://arxiv.org/abs/2402.07867
- Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection — https://arxiv.org/abs/2310.11511
- Corrective Retrieval Augmented Generation (CRAG) — https://arxiv.org/abs/2401.15884
- Enhancing Noise Robustness of Retrieval-Augmented Language Models with Adaptive Adversarial Training (RAAT) — https://arxiv.org/abs/2405.20978
- Evaluating the Robustness of Retrieval-Augmented Generation to Adversarial Evidence in the Health Domain — https://arxiv.org/abs/2509.03787
- HealthBench: Evaluating Large Language Models Towards Improved Human Health — https://arxiv.org/abs/2505.08775
- RAG LLMs are Not Safer: A Safety Analysis of Retrieval-Augmented Generation for Large Language Models — https://arxiv.org/abs/2504.18041
- MedHELM: Holistic Evaluation of Large Language Models for Medical Tasks — https://arxiv.org/abs/2505.23802
- Beyond Benchmarks: Dynamic, Automatic And Systematic Red-Teaming Agents For Trustworthy Medical Language Models (DAS) — https://arxiv.org/abs/2508.00923
- Red teaming ChatGPT in medicine to yield real-world insights on model behavior — https://www.nature.com/articles/s41746-025-01542-0
- Knowing When to Abstain: Medical LLMs Under Clinical Uncertainty — https://arxiv.org/abs/2601.12471
- G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment — https://aclanthology.org/2023.emnlp-main.153/
- Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena — https://arxiv.org/abs/2306.05685
- Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge — https://arxiv.org/abs/2410.02736
- Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge — https://arxiv.org/abs/2406.07791
- Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models (PoLL) — https://arxiv.org/abs/2404.18796
- Evaluating clinical AI summaries with large language models as judges — https://www.nature.com/articles/s41746-025-02005-2
- CQA-Eval: Designing Reliable Evaluations of Multi-paragraph Clinical QA under Resource Constraints — https://arxiv.org/abs/2510.10415
- Large Language Models Encode Clinical Knowledge (Med-PaLM) — https://arxiv.org/abs/2212.13138
- Towards Expert-Level Medical Question Answering with Large Language Models (Med-PaLM 2) — https://arxiv.org/abs/2305.09617
- MEDITRON-70B: Scaling Medical Pretraining for Large Language Models — https://arxiv.org/abs/2311.16079
- BioMistral: A Collection of Open-Source Pretrained Large Language Models for Medical Domains — https://arxiv.org/abs/2402.10373
- ChatDoctor: A Medical Chat Model Fine-Tuned on a Large Language Model Meta-AI (LLaMA) Using Medical Domain Knowledge — https://arxiv.org/abs/2303.14070

### Added 2026-07-12 (§2 Goodhart / metric-gaming positioning; verified via web search)
- Strathern, M. (1997). 'Improving ratings': audit in the British University system. European Review 5(3):305–321 — https://ideas.repec.org/a/cup/eurrev/v5y1997i03p305-321_00.html  (canonical "when a measure becomes a target…" formulation of Goodhart's law)
- Amodei, D. et al. (2016). Concrete Problems in AI Safety — https://arxiv.org/abs/1606.06565  (names reward hacking as a core safety problem)
- Krakovna, V. et al. (2020). Specification gaming: the flip side of AI ingenuity. DeepMind — https://deepmind.google/blog/specification-gaming-the-flip-side-of-ai-ingenuity/
- Pan, A., Bhatia, K., Steinhardt, J. (2022). The Effects of Reward Misspecification: Mapping and Mitigating Misaligned Models. ICLR 2022 — https://openreview.net/forum?id=JYtwGwIL7ye
- Skalse, J., Howe, N., Krasheninnikov, D., Krueger, D. (2022). Defining and Characterizing Reward Hacking. NeurIPS 2022 — https://arxiv.org/abs/2209.13085
- Geirhos, R. et al. (2020). Shortcut learning in deep neural networks. Nature Machine Intelligence 2:665–673 — https://www.nature.com/articles/s42256-020-00257-z

## EXCLUDED (unverifiable)

None. The FLAGGED/UNVERIFIABLE list was empty; no items were excluded, and no paper outside the verified list was cited.
