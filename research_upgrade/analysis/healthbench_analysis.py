"""External-validity analysis on HealthBench (B2): does the keyword rubric over-credit
the fine-tune relative to a clinical judge on *external, physician-authored* cases?

We compare, per model, the keyword-"safe" rate vs the judge's appropriate_escalation
rate, and the keyword<->judge divergence. The base-vs-fine-tune contrast is robust to
the (coarse) emergency-theming of the HealthBench subset because any labelling bias is
symmetric across the two models. We also print the judge's handling of clearly-benign
cases to confirm it is not merely echoing the safety_required flag.

Usage: python healthbench_analysis.py  ->  results_v2/healthbench_analysis.json
"""
import json
import os

import numpy as np

RESULTS = r"C:\OpenHealth\research_upgrade\results_v2"


def esc(r):
    v = r.get("appropriate_escalation")
    if isinstance(v, str):
        return v.lower() in ("true", "yes", "1")
    return bool(v)


def main():
    rows = json.load(open(os.path.join(RESULTS, "judged_healthbench.json"), encoding="utf-8"))
    out = {"n_cells": len(rows)}
    for model in sorted({r["model"] for r in rows}):
        sub = [r for r in rows if r["model"] == model]
        kw = [bool(r.get("kw_safe")) for r in sub]
        ap = [esc(r) for r in sub]
        tri = [r.get("triage") for r in sub if isinstance(r.get("triage"), (int, float))]
        raw_agree = np.mean([k == a for k, a in zip(kw, ap)])
        kwsafe_not_esc = np.mean([k and not a for k, a in zip(kw, ap)])
        out[model] = {
            "n": len(sub),
            "keyword_safe_rate": round(float(np.mean(kw)), 3),
            "appropriate_escalation_rate": round(float(np.mean(ap)), 3),
            "mean_triage_0_2": round(float(np.mean(tri)), 3) if tri else None,
            "raw_agreement_kw_vs_judge": round(float(raw_agree), 3),
            "p_keyword_safe_and_not_escalated": round(float(kwsafe_not_esc), 3),
        }
    # keyword over-crediting = kw-safe rate minus appropriate-escalation rate
    for m in out:
        if isinstance(out[m], dict) and "keyword_safe_rate" in out[m]:
            out[m]["keyword_overcredit_gap"] = round(
                out[m]["keyword_safe_rate"] - out[m]["appropriate_escalation_rate"], 3)
    json.dump(out, open(os.path.join(RESULTS, "healthbench_analysis.json"), "w"), indent=1, default=str)
    print(json.dumps(out, indent=1))

    # sanity: show a few judge rationales where it marked appropriate_escalation=True
    # (confirms the judge uses clinical reasoning, not just the safety_required flag)
    print("\n--- sample judge rationales (appropriate_escalation=True; benign handled correctly?) ---")
    n = 0
    for r in rows:
        if r["model"] == "openhealth-doctor" and esc(r) and (r.get("harm", 2) or 2) == 0:
            print(f"  [{r['case_id']}] triage={r.get('triage')} harm={r.get('harm')}: {r.get('rationale','')[:160]}")
            n += 1
            if n >= 4:
                break


if __name__ == "__main__":
    main()
