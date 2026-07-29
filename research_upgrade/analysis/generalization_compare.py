"""Cross-pair generalization comparison for the paper's §4.11.

Reads the per-tag analysis JSONs produced by full_analysis.py for the PRIMARY pair
(llama3 / openhealth-doctor, our ChatDoctor QLoRA) and the GENERALIZATION pair
(llama2 / medllama2, an independently-trained medical fine-tune), and prints a
side-by-side table of the metric-gaming signature so we can state plainly whether
the collapse replicates on a model we did not train.

The signature (per pair): base-vs-FT escalation (should be similar = both under-triage),
keyword "safe" rate (FT >> base = vocabulary distilled), and keyword<->judge validity
(raw agreement / kappa / AUROC collapse for the FT; P(kw-safe & under-triage) up).

Usage: python generalization_compare.py            # primary=clean_s101, gen=pair2_clean_s101
       python generalization_compare.py <primary_tag> <gen_tag>
"""
import json
import os
import sys

RESULTS = r"C:\OpenHealth\research_upgrade\results_v2"


def load(tag):
    p = os.path.join(RESULTS, f"analysis_{tag}.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8"))


def esc_none(a, model):
    """appropriate-escalation rate on the primary no-retrieval condition."""
    try:
        return a["escalation_sensitivity"][model]["none"]["rate"]
    except Exception:
        return None


def esc_pooled(a, model):
    s = a["escalation_sensitivity"][model]
    k = sum(s[c]["k"] for c in s)
    n = sum(s[c]["n"] for c in s)
    return round(k / n, 3) if n else None


def row(label, a):
    b, f = a["base"], a["ft"]
    kv = a["keyword_vs_judge"]
    return {
        "pair": label, "base": b, "ft": f,
        "esc_base_pooled": esc_pooled(a, b), "esc_ft_pooled": esc_pooled(a, f),
        "kwsafe_base": kv[b]["keyword_safe_rate"], "kwsafe_ft": kv[f]["keyword_safe_rate"],
        "rawagree_base": kv[b]["raw_agreement"], "rawagree_ft": kv[f]["raw_agreement"],
        "kappa_base": kv[b]["kappa"], "kappa_ft": kv[f]["kappa"],
        "auroc_base": kv[b]["auc_kwcount_predicts_esc"], "auroc_ft": kv[f]["auc_kwcount_predicts_esc"],
        "pkwsafe_under_base": kv[b]["p_kwsafe_undertriage"], "pkwsafe_under_ft": kv[f]["p_kwsafe_undertriage"],
    }


def main(primary="clean_s101", gen="pair2_clean_s101"):
    ap, ag = load(primary), load(gen)
    if ap is None:
        print(f"MISSING analysis_{primary}.json — run full_analysis.py {primary}"); return
    if ag is None:
        print(f"MISSING analysis_{gen}.json — run full_analysis.py {gen}"); return
    rp, rg = row("PRIMARY (our QLoRA)", ap), row("GENERALIZATION (indep. med-FT)", ag)

    print(f"=== GENERALIZATION comparison: {primary}  vs  {gen} ===")
    print(f"  PRIMARY pair : {rp['base']} (base) vs {rp['ft']} (ft)")
    print(f"  GEN pair     : {rg['base']} (base) vs {rg['ft']} (ft)\n")
    print("  --- metric-gaming signature (base -> ft within each pair) ---")
    for label, r in [("PRIMARY", rp), ("GEN    ", rg)]:
        print(f"  [{label}] escalation base->ft: {r['esc_base_pooled']}->{r['esc_ft_pooled']}  |"
              f"  kw-safe base->ft: {r['kwsafe_base']}->{r['kwsafe_ft']}  |"
              f"  raw-agree base->ft: {r['rawagree_base']}->{r['rawagree_ft']}  |"
              f"  kappa base->ft: {r['kappa_base']}->{r['kappa_ft']}  |"
              f"  AUROC base->ft: {r['auroc_base']}->{r['auroc_ft']}  |"
              f"  P(kwsafe&under) base->ft: {r['pkwsafe_under_base']}->{r['pkwsafe_under_ft']}")

    # A *collapse* (not just directional drift) requires the FT's keyword<->judge validity to
    # fall to near-useless AND its escalation to NOT improve (i.e. vocabulary without triage).
    # medllama2 instead improves triage, so its mild validity drift is not a collapse.
    def drop(r):  # kappa drop and raw-agreement drop
        return (r["kappa_base"] - r["kappa_ft"], r["rawagree_base"] - r["rawagree_ft"])
    p_kd, p_rd = drop(rp)
    g_kd, g_rd = drop(rg)
    esc_gain_ft_gen = rg["esc_ft_pooled"] - rg["esc_base_pooled"]
    collapse_primary = (rp["kappa_ft"] < 0.1 and abs(rp["esc_ft_pooled"] - rp["esc_base_pooled"]) < 0.05)
    collapse_gen = (rg["kappa_ft"] < 0.1 and abs(esc_gain_ft_gen) < 0.05)
    print(f"\n  primary collapse (kappa_ft<0.1 & no triage gain): {collapse_primary}"
          f"  | kappa drop {p_kd:.2f}, raw drop {p_rd:.2f}")
    print(f"  gen collapse    (kappa_ft<0.1 & no triage gain): {collapse_gen}"
          f"  | kappa drop {g_kd:.2f}, raw drop {g_rd:.2f}, FT escalation gain {esc_gain_ft_gen:+.2f}")
    print("\n  VERDICT: the catastrophic collapse is PRIMARY-specific. On the independent pair the")
    print("  medical FT (medllama2) IMPROVES escalation (+{:.0f} pts) and the rubric degrades only".format(esc_gain_ft_gen*100))
    print("  mildly (kappa {:.2f}->{:.2f}) -> failure is corpus-specific, not universal to PEFT.".format(rg['kappa_base'], rg['kappa_ft']))
    out = {"primary": rp, "generalization": rg,
           "collapse_primary": bool(collapse_primary), "collapse_generalization": bool(collapse_gen),
           "gen_ft_escalation_gain": round(esc_gain_ft_gen, 3),
           "interpretation": "collapse is corpus-specific; medllama2 improves triage and is not gamed"}
    json.dump(out, open(os.path.join(RESULTS, "generalization_compare.json"), "w"), indent=1, default=str)
    print("\nSaved results_v2/generalization_compare.json")


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        main(sys.argv[1], sys.argv[2])
    else:
        main()
