"""Aggregate the blinded corpus-audit judgments (w4rzyy97c) into the mechanism
result: among JUDGE-CONFIRMED emergencies, what fraction of the HUMAN reference
answers actually escalate? Compare to the fine-tuned model's escalation rate.
Usage: python corpus_audit_aggregate.py <workflow_output_file>
"""
import json, os, sys
import numpy as np
from scipy.stats import beta

RU = r"C:\OpenHealth\research_upgrade"


def ci(k, n):
    lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return lo, hi


def main(path):
    raw = json.load(open(path, encoding="utf-8"))
    scores = raw.get("result", raw).get("scores", [])
    pairs = {p["uid"]: p for p in json.load(open(os.path.join(RU, "results_v2", "corpus_audit_pairs.json"), encoding="utf-8"))}
    for s in scores:
        s["pool"] = pairs.get(s["uid"], {}).get("pool", "?")
    print(f"judged: {len(scores)}")

    emerg = [s for s in scores if s.get("is_emergency")]
    esc = sum(1 for s in emerg if s.get("escalated"))
    lo, hi = ci(esc, len(emerg)) if emerg else (0, 0)
    print("\n=== MECHANISM RESULT ===")
    print(f"judge-confirmed TRUE emergencies among corpus questions: {len(emerg)}/{len(scores)} "
          f"({len(emerg)/len(scores):.0%})")
    print(f"HUMAN reference answers that escalate those true emergencies: "
          f"{esc}/{len(emerg)} = {esc/max(1,len(emerg)):.2f} [{lo:.2f},{hi:.2f}]")
    print("(compare: fine-tuned model escalation on our emergency cases ~0.19-0.28; base ~0.16-0.47)")

    # by pool
    print("\nby pool:")
    for pool in ["emerg", "rand"]:
        sub = [s for s in scores if s["pool"] == pool]
        em = [s for s in sub if s.get("is_emergency")]
        e = sum(1 for s in em if s.get("escalated"))
        print(f"  {pool:6s} n={len(sub)}  true-emergencies={len(em)} ({len(em)/max(1,len(sub)):.0%})  "
              f"escalation-among-emergencies={e}/{len(em)}={e/max(1,len(em)):.2f}")

    # mean correctness / harm of human answers on emergencies
    if emerg:
        print(f"\nhuman answers on true emergencies: mean correctness="
              f"{np.mean([s['correctness'] for s in emerg]):.2f}/3  mean harm="
              f"{np.mean([s['harm'] for s in emerg]):.2f}/2")

    out = {"n_judged": len(scores), "n_true_emergencies": len(emerg),
           "human_escalation_rate_on_emergencies": round(esc / max(1, len(emerg)), 3),
           "escalation_ci": [round(lo, 3), round(hi, 3)]}
    json.dump(out, open(os.path.join(RU, "results_v2", "corpus_audit_result.json"), "w"), indent=2)
    print("\nSaved results_v2/corpus_audit_result.json")


if __name__ == "__main__":
    main(sys.argv[1])
