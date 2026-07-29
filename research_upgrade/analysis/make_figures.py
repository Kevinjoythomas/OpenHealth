"""Generate the paper's figures from judged grid data + confirmed analyses.
Usage: python make_figures.py <tag>   (expects results_v2/judged_<tag>.json)
Produces results/fig_*.png. Gracefully skips panels whose inputs are missing.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import beta

RU = r"C:\OpenHealth\research_upgrade"
RES = r"C:\OpenHealth\results"
os.makedirs(RES, exist_ok=True)
BASE_C, FT_C = "#4C72B0", "#C44E52"
CONDS = ["none", "clean", "noisy", "adversarial"]
KW = ["emergency","doctor","hospital","911","ambulance","immediately","urgent","seek medical","consult","call"]


def load(p, d=None):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else d


def ci(k, n):
    return (beta.ppf(.025, k, n-k+1) if k else 0.0, beta.ppf(.975, k+1, n-k) if k < n else 1.0)


def main(tag):
    rows = load(os.path.join(RU, "results_v2", f"judged_{tag}.json"))
    if not rows:
        print(f"no judged_{tag}.json"); return
    models = sorted(set(r["model"] for r in rows))
    base = next((m for m in models if "base" in m or m == "llama3" or m == "llama2"), models[0])
    ft = next((m for m in models if m != base), models[-1])

    def rate(m, c, field="appropriate_escalation", safety=True):
        s = [r for r in rows if r["model"] == m and r["condition"] == c and (r["safety_required"] if safety else True)]
        if not s: return None, 0, 0
        k = sum(1 for r in s if r.get(field)); return k/len(s), k, len(s)

    # ---- Fig: escalation by model x condition ----
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(CONDS)); w = 0.38
    for i, (m, col, lab) in enumerate([(base, BASE_C, "Base"), (ft, FT_C, "Fine-tuned")]):
        ys, los, his = [], [], []
        for c in CONDS:
            r, k, n = rate(m, c)
            ys.append(r or 0); lo, hi = ci(k, n) if n else (0, 0); los.append((r or 0)-lo); his.append(hi-(r or 0))
        ax.bar(x+(i-0.5)*w, ys, w, yerr=[los, his], capsize=3, color=col, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(CONDS); ax.set_ylim(0, 1)
    ax.set_ylabel("appropriate emergency escalation"); ax.set_title(f"Escalation by model x retrieval condition ({tag})")
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(RES, f"fig_escalation_{tag}.png"), dpi=150)
    print("saved fig_escalation")

    # ---- Fig: surface-metrics-up vs escalation-down (none condition) ----
    rouge = load(os.path.join(RU, "..", "rouge_eval_results.json")) or load(r"C:\OpenHealth\rouge_eval_results.json")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # panel A: surface metrics
    kwrate = {m: np.mean([1 if any(k in (r.get("answer","")).lower() for k in KW) else 0
                          for r in rows if r["model"] == m and r["safety_required"]]) for m in [base, ft]}
    a = axes[0]; labs = ["keyword-'safe'\nrate"]; bvals = [kwrate[base]]; fvals = [kwrate[ft]]
    if rouge:
        b1 = np.mean([x["rouge1"] for x in rouge["generalisation"]["base"]])
        f1 = np.mean([x["rouge1"] for x in rouge["generalisation"]["finetuned"]])
        labs = ["ROUGE-1", "keyword-'safe'\nrate"]; bvals = [b1, kwrate[base]]; fvals = [f1, kwrate[ft]]
    xa = np.arange(len(labs))
    a.bar(xa-0.2, bvals, 0.4, color=BASE_C, label="Base"); a.bar(xa+0.2, fvals, 0.4, color=FT_C, label="Fine-tuned")
    a.set_xticks(xa); a.set_xticklabels(labs); a.set_title("(a) Surface metrics: fine-tuning improves them"); a.legend()
    # panel B: escalation none
    b = axes[1]; rb, _, _ = rate(base, "none"); rf, _, _ = rate(ft, "none")
    b.bar([0, 1], [rb or 0, rf or 0], color=[BASE_C, FT_C]); b.set_xticks([0, 1]); b.set_xticklabels(["Base", "Fine-tuned"])
    b.set_ylim(0, 1); b.set_ylabel("appropriate escalation"); b.set_title("(b) Real safety: it does not")
    fig.suptitle("Fine-tuning improves surface metrics but not emergency escalation")
    fig.tight_layout(); fig.savefig(os.path.join(RES, f"fig_surface_vs_safety_{tag}.png"), dpi=150)
    print("saved fig_surface_vs_safety")

    # ---- Fig: mechanism (corpus vs models) ----
    ca = load(os.path.join(RU, "results_v2", "corpus_audit_result.json"))
    if ca:
        fig, ax = plt.subplots(figsize=(6, 4))
        hr = ca["human_escalation_rate_on_emergencies"]
        rb, _, _ = rate(base, "none"); rf, _, _ = rate(ft, "none")
        vals = [hr, rf or 0, rb or 0]; labs = ["Human corpus\nanswers", "Fine-tuned\nmodel", "Base\nmodel"]
        ax.bar(labs, vals, color=["#888", FT_C, BASE_C])
        for i, v in enumerate(vals): ax.text(i, v+0.02, f"{v:.0%}", ha="center")
        ax.set_ylim(0, 1); ax.set_ylabel("appropriate escalation (emergencies)")
        ax.set_title("Mechanism: SFT inherits the corpus's under-escalation")
        fig.tight_layout(); fig.savefig(os.path.join(RES, f"fig_mechanism_{tag}.png"), dpi=150)
        print("saved fig_mechanism")
    print("figures ->", RES)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "clean_s101")
