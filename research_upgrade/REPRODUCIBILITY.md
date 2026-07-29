# REPRODUCIBILITY.md — artifact ↔ claim map for *"Saying 'See a Doctor' Is Not Triage"*

Every quantitative claim in the paper regenerates from a released artifact + script. Paths are under `research_upgrade/`. All generations use temperature 0.3, ≤300 new tokens, `num_ctx=2048`, seed 101 (seeds 102/103 for variance); judging temperature 0.

## Data
| Artifact | What |
|---|---|
| `results_v2/clean_benchmark.json` | 96 de-novo vignettes (72 emergencies / 24 controls), ≤35-char shared run vs corpus |
| (public) ChatDoctor/HealthCareMagic-100k | training corpus (112,165 pairs) |
| `results_v2/corpus_register.json` | corpus keyword-register stats (79.5% any-keyword; *doctor* 69.3% … *911* 0.0%) → §4.8 |
| `results_v2/corpus_audit_result.json` | 510 sampled, 167 judge-labeled emergencies, human escalation 12.6% [.08,.19] → §4.8 |
| `results_v2/contamination_v2_original.json` | longest-run contamination (2–3 adapted) → §4.9a |

## Generation → grids
`harness_v2/gen_grid.py` (grid), `precompute_contexts.py` (retrieval contexts), `prompt_mitigation.py` (§4.10).
| Grid | Models × conditions × cases | → section |
|---|---|---|
| `grid_clean_s101/s102/s103.json` | {llama3, openhealth-doctor} × 4 × 96 | §4.1–4.6 (s101 primary; s102/s103 variance) |
| `grid_orig_s101.json` | same × original 100 | §4.7 |
| `grid_pair2_clean_s101.json` | {llama2, medllama2} × 4 × 96 | §4.11 |
| `mitigation_clean.json` | {base,ft} × {none,adv} + triage line | §4.10 |
| `grid_healthbench.json` (via `healthbench_prep.py`) | {base,ft} × 80 HealthBench | §4.12 |
| `mcq_results.json` (`mcq_bench.py`) | MedQA/PubMedQA acc, 400 ea | §5 knowledge |

## Judging pipeline (per grid)
`judge_grid_prep.py <grid> <tag>` → `judge_batches_<tag>/` + `blindmap_<tag>.json`
→ Workflow `grid_judge.wf.js {tag,nBatches}` (Claude Opus, blinded) → merge journal → `wfjudge_<tag>.json`
→ `analysis/grid_aggregate.py <wfjudge> <tag>` → `judged_<tag>.json` (adds kw_safe, kw_count).
Local judges for IRR: `judge_local.py <tag> --emergency --cond=none` (Qwen2.5-7B + Mistral-7B) → `judge_local_<tag>.json`.

## Analysis → paper numbers
| Script | Output | Regenerates |
|---|---|---|
| `analysis/full_analysis.py <tag>` | `analysis_<tag>.json` | Table 1 escalation + CIs; Table 2 keyword-vs-judge (κ/PABAK/AUROC); McNemar; by-category |
| `analysis/robustness_stats.py clean_s101` | `robustness_stats.json` | §4.1 case-level TOST [−6.2,+6.2] + no-retrieval TOST [−5.7,+16.8]; §4.2 stricter urgency rubric; §4.6 length regression (β=−0.30, p=0.44); §4.5 33/72 |
| `analysis/majority_irr.py clean_s101` | `majority_clean_s101.json` | §4.6 Fleiss κ=0.30; pairwise κ; majority-of-3 43%/49% |
| `analysis/generalization_compare.py` | `generalization_compare.json` | §4.11 primary-vs-gen contrast |
| `analysis/panel_compare.py` | `panel_compare.json` | §4.11 Table 3 panel (invalidation diagnostic) |
| `analysis/healthbench_analysis.py` | `healthbench_analysis.json` | §4.12 (kw 66%/28%, PPV-fail 4.3×) |
| `analysis/corpus_audit_aggregate.py` | `corpus_audit_result.json` | §4.8 13% |
| `analysis/make_figures.py`, `fig_generalization.py` | `../results/*.png` | Figures 1–2 |

## Key headline numbers ↔ source (spot-check)
- 24.0%/24.0% escalation = 69/288 each → `analysis_clean_s101.json` (escalation_sensitivity).
- 6.5× = 65%/10% keyword failure = 1−{0.351,0.903} → `analysis_clean_s101.json` (keyword_vs_judge.keyword_safe_rate).
- κ 0.61→0.03, AUROC 0.91→0.67, PABAK 0.67→−0.38 → same file.
- Conditional PPV failure 75%/40% (ft/base) → recompute from `judged_clean_s101.json` (`robustness_stats.py` reports the joint 0.68/0.14; Table 2).
- corpus 12.6% / 79.5% / *doctor* 69.3% → `corpus_audit_result.json`, `corpus_register.json`.
- mitigation base 139/144=97%, ft 71/144=49% → `judged_mit_clean.json`.
- seed stability: ft keyword-safe 0.90/0.91/0.88, base 0.35/0.35/0.32 across s101/s102/s103 (deterministic, from grids).

## Environment
Fine-tuning: 1× cloud T4 (LoRA r=8/α=16, 1000 steps). Inference/judging: consumer 4 GB GPU via Ollama (Q4_K_M); local judges spill CPU/GPU. Model digests logged in the grids. New §2 citations (Goodhart/reward-hacking/shortcut-learning) to add to the bib are listed in `reviews_v3_and_fixes.md`.
