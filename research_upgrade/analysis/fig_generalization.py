"""Figure for §4.11: the metric-gaming collapse is corpus-specific.
Two panels, base->ft within each pair:
  (A) appropriate-escalation rate  -- primary flat (24->24), gen rises (21->55)
  (B) keyword<->judge Cohen's kappa -- primary collapses (0.61->0.03), gen mild (0.45->0.35)
Reads generalization_compare.json. Saves ../results/fig_generalization.png.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = r"C:\OpenHealth\research_upgrade\results_v2"
OUT = r"C:\OpenHealth\results\fig_generalization.png"

d = json.load(open(os.path.join(RESULTS, "generalization_compare.json"), encoding="utf-8"))
P, G = d["primary"], d["generalization"]

pairs = [("Primary\n(Llama-3 → our ChatDoctor QLoRA)", P, "#c0392b"),
         ("Generalization\n(Llama-2 → Medllama2)", G, "#2471a3")]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.2))
x = [0, 1]
for label, r, color in pairs:
    ax1.plot(x, [r["esc_base_pooled"], r["esc_ft_pooled"]], "o-", color=color, lw=2.4, ms=9, label=label)
    ax2.plot(x, [r["kappa_base"], r["kappa_ft"]], "o-", color=color, lw=2.4, ms=9, label=label)

for ax, title, ylab in [(ax1, "Appropriate escalation (triage)", "escalation rate"),
                        (ax2, "Keyword↔judge validity", "Cohen's κ")]:
    ax.set_xticks(x); ax.set_xticklabels(["base", "fine-tuned"])
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylab)
    ax.grid(alpha=0.3, ls=":")
    ax.set_xlim(-0.25, 1.25)

ax1.set_ylim(0, 0.7)
ax2.set_ylim(-0.05, 0.7)
ax2.axhline(0, color="grey", lw=0.8)
# annotate the story
ax1.annotate("no triage gain (24%→24%)", (1, P["esc_ft_pooled"]), xytext=(0.30, 0.265),
             fontsize=8.5, color="#c0392b", ha="left")
ax1.annotate("real gain\n(21%→55%)", (1, G["esc_ft_pooled"]), xytext=(0.60, 0.40),
             fontsize=8.5, color="#2471a3", ha="left")
ax2.annotate("collapse\n(0.61→0.03)", (1, P["kappa_ft"]), xytext=(0.34, 0.06),
             fontsize=8.5, color="#c0392b", ha="left")
ax2.annotate("mild\n(0.45→0.35)", (1, G["kappa_ft"]), xytext=(0.52, 0.45),
             fontsize=8.5, color="#2471a3", ha="left")

ax1.legend(fontsize=8, loc="lower left", framealpha=0.95)
fig.suptitle("Metric-gaming is corpus-specific: fine-tuning games the lexical rubric only when the\n"
             "corpus is lexically-safe-but-under-triaging (left), not when it teaches real escalation (right)",
             fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.93))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=150)
print(f"saved {OUT}")
print(f"  primary esc {P['esc_base_pooled']}->{P['esc_ft_pooled']}, kappa {P['kappa_base']}->{P['kappa_ft']}")
print(f"  gen     esc {G['esc_base_pooled']}->{G['esc_ft_pooled']}, kappa {G['kappa_base']}->{G['kappa_ft']}")
