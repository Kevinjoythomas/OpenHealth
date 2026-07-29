"""Combine Claude + Qwen + Mistral judgments into majority-of-3 labels and report
inter-rater reliability (Fleiss' kappa + pairwise Cohen's kappa) on the
appropriate_escalation label, then recompute escalation rates on MAJORITY labels.
Usage: python majority_irr.py <tag>   (expects judged_<tag>.json + judge_local_<tag>.json + blindmap_<tag>.json)
"""
import json, os, sys
import numpy as np
from scipy.stats import beta

RESULTS = r"C:\OpenHealth\research_upgrade\results_v2"


def cohen(a, b):
    a, b = np.array(a), np.array(b)
    po = np.mean(a == b); pa = np.mean(a); pb = np.mean(b)
    pe = pa*pb + (1-pa)*(1-pb)
    return (po - pe) / (1 - pe) if (1-pe) > 0 else float("nan")


def fleiss(mat):
    """mat: n_items x n_categories counts (rows sum to n_raters)."""
    n, k = mat.shape
    N = mat.sum(axis=1)[0]
    p = mat.sum(axis=0) / (n * N)
    P = ((mat**2).sum(axis=1) - N) / (N * (N - 1))
    Pbar = P.mean(); Pe = (p**2).sum()
    return (Pbar - Pe) / (1 - Pe) if (1-Pe) > 0 else float("nan")


def ci(k, n):
    return (round(beta.ppf(.025, k, n-k+1), 3) if k else 0.0,
            round(beta.ppf(.975, k+1, n-k), 3) if k < n else 1.0)


def main(tag):
    claude = {r["blind_id"] if "blind_id" in r else r.get("uid"): r
              for r in json.load(open(os.path.join(RESULTS, f"judged_{tag}.json"), encoding="utf-8"))}
    # judged_<tag>.json is keyed rows w/ meta; build blind_id->record via blindmap uid match
    blind = json.load(open(os.path.join(RESULTS, f"blindmap_{tag}.json"), encoding="utf-8"))
    local = json.load(open(os.path.join(RESULTS, f"judge_local_{tag}.json"), encoding="utf-8"))
    # judged_<tag> rows carry model/condition/appropriate_escalation but keyed how? saved as list of de-blinded rows.
    # index claude by uid
    claude_by_uid = {r["uid"]: r for r in json.load(open(os.path.join(RESULTS, f"judged_{tag}.json"), encoding="utf-8"))}

    def esc(v):
        if isinstance(v, dict): v = v.get("appropriate_escalation")
        if isinstance(v, str): return v.lower() in ("true", "yes", "1")
        return bool(v)

    rows = []
    for bid, meta in blind.items():
        uid = meta["uid"]
        if uid not in claude_by_uid or bid not in local:
            continue
        c = esc(claude_by_uid[uid])
        q = local[bid].get("qwen2.5"); m = local[bid].get("mistral")
        if not (isinstance(q, dict) and isinstance(m, dict)):
            continue
        votes = [c, esc(q), esc(m)]
        rows.append({**meta, "claude": c, "qwen": esc(q), "mistral": esc(m),
                     "majority": sum(votes) >= 2})
    print(f"=== IRR {tag}: {len(rows)} cells with all 3 judges ===")
    C = [int(r["claude"]) for r in rows]; Q = [int(r["qwen"]) for r in rows]; M = [int(r["mistral"]) for r in rows]
    print(f"pairwise Cohen kappa: Claude-Qwen={cohen(C,Q):.3f}  Claude-Mistral={cohen(C,M):.3f}  Qwen-Mistral={cohen(Q,M):.3f}")
    mat = np.array([[3-(a+b+c_), (a+b+c_)] for a, b, c_ in zip(C, Q, M)])  # [not-esc, esc] counts
    print(f"Fleiss kappa (3 judges, escalation): {fleiss(mat):.3f}")
    print("\nMAJORITY-of-3 appropriate escalation on safety cases, by model x condition:")
    for mdl in sorted(set(r["model"] for r in rows)):
        parts = []
        for cond in ["none", "clean", "noisy", "adversarial"]:
            sub = [r for r in rows if r["model"] == mdl and r["condition"] == cond and r["safety_required"]]
            if not sub: parts.append(f"{cond[:4]}=NA"); continue
            k = sum(1 for r in sub if r["majority"]); n = len(sub); lo, hi = ci(k, n)
            parts.append(f"{cond[:4]}={k}/{n}={k/n:.2f}[{lo},{hi}]")
        print(f"  {mdl:20s} " + "  ".join(parts))
    json.dump(rows, open(os.path.join(RESULTS, f"majority_{tag}.json"), "w"), indent=0, default=str)
    print(f"\nSaved results_v2/majority_{tag}.json")


if __name__ == "__main__":
    main(sys.argv[1])
