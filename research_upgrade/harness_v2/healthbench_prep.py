"""Convert healthbench_runs.json (B2 external validation) into the standard grid
format so the SAME blind-judge -> aggregate -> analysis pipeline applies.
HealthBench cases are physician-authored emergency-referral-themed single-turn
questions; base/finetuned answers with no retrieval (condition 'none').

Usage: python healthbench_prep.py    (writes results_v2/grid_healthbench.json)
Then:  python judge_grid_prep.py results_v2/grid_healthbench.json healthbench
       Workflow grid_judge.wf.js {tag: healthbench, nBatches: N}
       python analysis/grid_aggregate.py <wf_out> healthbench
       python analysis/full_analysis.py healthbench
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results_v2")
SRC = os.path.join(RESULTS, "healthbench_runs.json")
OUT = os.path.join(RESULTS, "grid_healthbench.json")
# match the primary analysis' model naming so base/ft detection works unchanged
MODEL_MAP = {"base": "llama3", "finetuned": "openhealth-doctor"}


def main():
    d = json.load(open(SRC, encoding="utf-8"))
    cases = d.get("cases", {})
    grid = {"llama3": {}, "openhealth-doctor": {}}
    n_ok = 0
    for cid, entry in cases.items():
        q = entry.get("question", "")
        for mk, ans in entry.get("answers", {}).items():
            model = MODEL_MAP.get(mk)
            if model is None or not isinstance(ans, dict):
                continue
            text = ans.get("answer")
            if not text:               # skip error/empty cells
                continue
            key = f"{cid}|none"
            grid[model][key] = {
                "case_id": f"HB_{cid}", "condition": "none",
                "question": q[:2000], "answer": text,
                "safety_required": True,          # emergency-referral-themed by construction
                "category": "healthbench",
            }
            n_ok += 1
    json.dump(grid, open(OUT, "w"), indent=0)
    print(f"wrote {OUT}: llama3={len(grid['llama3'])} openhealth-doctor={len(grid['openhealth-doctor'])} cells ({n_ok} total)")


if __name__ == "__main__":
    main()
