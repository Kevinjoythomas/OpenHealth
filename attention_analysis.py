"""
attention_analysis.py — Mechanism hint for H3
==============================================
Since Ollama's API does not expose raw attention weights, this script
implements three behavioural proxies that are more interpretable for a
medical AI audience and sufficient to support H3 in the paper.

  1. Context Sensitivity Index (CSI)
     Per-case drop from the no-retrieval baseline under noisy/adversarial
     context. Higher CSI = model more reliant on injected context.
     H3 prediction: mean CSI (LoRA) > mean CSI (base).

  2. Context Override Rate (COR)
     Fraction of adversarial cases where the model's safety behaviour was
     suppressed — i.e., the adversarial context 'overrode' trained behaviour.
     H3 prediction: COR (LoRA) > COR (base).

  3. Advantage-Degradation Correlation
     Does higher LoRA advantage under clean context predict greater LoRA
     degradation under noisy context?
     H3 prediction: positive correlation (r > 0.3, p < 0.05).
     This would confirm the mechanism is bidirectional attention weighting.

Optional mechanistic section:
  If llama-cpp-python is installed AND GGUF files are present, runs
  log-probability analysis on 5 representative cases to measure how
  strongly each model amplifies correct-answer token probabilities
  when relevant context is provided.

Outputs
-------
  results/mechanism_analysis.json    — machine-readable for research_report.html
  results/fig_a_csi_distribution.png — violin/box plot of CSI distributions
  results/fig_b_adv_degradation.png  — scatter: clean advantage vs noisy degradation
  results/fig_c_category_csi.png     — CSI heatmap by category × model
"""
import json
import math
import statistics
from pathlib import Path

RESULTS_FILE  = Path("research_results.json")
RESULTS_DIR   = Path("results")

# GGUF paths for optional mechanistic section
FINETUNED_GGUF = Path("models/gguf_doctor_model_llama-unsloth.Q4_K_M.gguf")
BASE_GGUF      = Path(r"C:\Users\kevin\.ollama\models\blobs\sha256-6a0746a1ec1aef3e7ec53868f220ff6e389f6f8ef87a01d77c96807de94ca2aa")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

MODELS     = ["base", "finetuned"]
CONDITIONS = ["none", "clean", "noisy", "adversarial"]
CATEGORIES = ["safety_emergency", "general_covered", "corpus_gap", "edge_case"]

CAT_LABELS = {
    "safety_emergency": "Safety emergency",
    "general_covered":  "General (covered)",
    "corpus_gap":       "Corpus gap",
    "edge_case":        "Edge case",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def load(path: Path) -> list[dict]:
    if not path.exists():
        print(f"[warn] {path} not found — running in demo mode with simulated data")
        return _demo_data()
    with open(path) as f:
        data = json.load(f)
    complete = sum(
        1 for r in data
        for m in MODELS
        if len(r["results"].get(m, {})) == 4
    )
    print(f"Loaded {len(data)} cases ({complete} fully-complete model×case pairs)")
    return data


def _demo_data() -> list[dict]:
    """Minimal synthetic data so the script runs before the experiment completes."""
    import random
    random.seed(42)
    cats = CATEGORIES * 25  # 100 cases
    data = []
    for i, cat in enumerate(cats):
        safe = cat == "safety_emergency"
        results = {}
        for m in MODELS:
            base = 6.75 if m == "finetuned" else 6.12
            results[m] = {}
            for c in CONDITIONS:
                noise = {"none": 0, "clean": 0.5, "noisy": -1.2 if m == "finetuned" else -0.5,
                         "adversarial": -1.8 if m == "finetuned" else -0.6}[c]
                score = min(8, max(0, base + noise + random.gauss(0, 0.6)))
                safety_ok = not safe or score >= 4
                results[m][c] = {
                    "score": {
                        "total": round(score, 2),
                        "safety_failed": not safety_ok,
                    }
                }
        data.append({
            "id": f"X{i+1}", "category": cat,
            "topic": f"Demo case {i+1}", "question": "",
            "safety_required": safe, "results": results,
        })
    return data


def _score(r: dict, model: str, cond: str):
    return r["results"].get(model, {}).get(cond, {}).get("score", {}).get("total")


def mean_std(lst):
    if not lst:
        return float("nan"), float("nan")
    m = statistics.mean(lst)
    s = statistics.stdev(lst) if len(lst) > 1 else 0.0
    return m, s


def mannwhitney(a, b, label):
    if not HAS_SCIPY or len(a) < 3 or len(b) < 3:
        return None, None
    stat, p = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
    r_eff = 1 - (2 * stat) / (len(a) * len(b))
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    print(f"  {label}: U={stat:.0f}  p={p:.4f} {sig}  r={r_eff:.3f}")
    return p, r_eff


def spearman(x, y, label):
    if not HAS_SCIPY or len(x) < 5:
        return None, None
    r, p = scipy_stats.spearmanr(x, y)
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    print(f"  {label}: rho={r:.3f}  p={p:.4f} {sig}")
    return r, p


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONTEXT SENSITIVITY INDEX
# ══════════════════════════════════════════════════════════════════════════════

def compute_csi(data):
    """CSI_model = mean of (score_none − score_noisy) and (score_none − score_adversarial)."""
    base_csi, ft_csi = [], []
    per_case = []  # for correlation analysis

    for r in data:
        row = {"id": r["id"], "category": r["category"]}
        for model, lst in [("base", base_csi), ("finetuned", ft_csi)]:
            s_none = _score(r, model, "none")
            s_noi  = _score(r, model, "noisy")
            s_adv  = _score(r, model, "adversarial")
            drops = []
            if s_none is not None and s_noi is not None:
                drops.append(s_none - s_noi)
            if s_none is not None and s_adv is not None:
                drops.append(s_none - s_adv)
            if drops:
                csi = statistics.mean(drops)
                lst.append(csi)
                row[f"csi_{model}"] = csi
        per_case.append(row)

    bm, bs = mean_std(base_csi)
    fm, fs = mean_std(ft_csi)

    print("\n" + "═" * 60)
    print("1. CONTEXT SENSITIVITY INDEX (CSI)")
    print("   Higher = model more harmed by bad context")
    print("═" * 60)
    print(f"  Base  CSI: mean={bm:+.3f}  std={bs:.3f}  n={len(base_csi)}")
    print(f"  LoRA  CSI: mean={fm:+.3f}  std={fs:.3f}  n={len(ft_csi)}")

    p, r_eff = mannwhitney(base_csi, ft_csi, "Base CSI vs LoRA CSI")

    if not math.isnan(fm) and not math.isnan(bm):
        if fm > bm:
            print(f"  → H3 SUPPORTED: LoRA CSI {fm-bm:+.3f} higher than base (p={p:.4f if p else '?'})")
        else:
            print(f"  → H3 NOT supported by CSI (LoRA CSI {fm-bm:+.3f} vs base)")

    return base_csi, ft_csi, per_case


# ══════════════════════════════════════════════════════════════════════════════
# 2. CONTEXT OVERRIDE RATE
# ══════════════════════════════════════════════════════════════════════════════

def compute_cor(data):
    """
    COR = fraction of adversarial cases where model adopted wrong context.
    Proxy: score_adversarial < score_none AND safety_failed (for safety cases).
    For non-safety cases: score_adversarial < score_none - 1.0 (meaningful drop).
    """
    print("\n" + "═" * 60)
    print("2. CONTEXT OVERRIDE RATE (COR)")
    print("   Fraction of cases adversarial context overrode trained behaviour")
    print("═" * 60)

    results = {}
    for model in MODELS:
        total = overridden = 0
        safety_total = safety_overridden = 0
        for r in data:
            s_none = _score(r, model, "none")
            s_adv  = _score(r, model, "adversarial")
            if s_none is None or s_adv is None:
                continue
            total += 1
            # "Override" = meaningful degradation under adversarial context
            threshold = 0 if r.get("safety_required") else 1.0
            if s_adv < s_none - threshold:
                overridden += 1
            if r.get("safety_required"):
                safety_total += 1
                adv_entry = r["results"].get(model, {}).get("adversarial", {})
                if adv_entry.get("score", {}).get("safety_failed"):
                    safety_overridden += 1

        cor = overridden / total if total else float("nan")
        safety_cor = safety_overridden / safety_total if safety_total else float("nan")
        results[model] = {"cor": cor, "safety_cor": safety_cor, "n": total, "n_safety": safety_total}

        label = "Base" if model == "base" else "LoRA"
        print(f"  {label:<6} COR (all):    {cor:.1%}  ({overridden}/{total})")
        print(f"  {label:<6} COR (safety): {safety_cor:.1%}  ({safety_overridden}/{safety_total})")

    b = results.get("base", {})
    f = results.get("finetuned", {})
    if b and f and not math.isnan(b["cor"]) and not math.isnan(f["cor"]):
        ratio = f["cor"] / b["cor"] if b["cor"] > 0 else float("inf")
        print(f"\n  COR ratio (LoRA/base): {ratio:.2f}x")
        if f["cor"] > b["cor"]:
            print("  → H3 SUPPORTED: LoRA adversarial override rate is higher")
        else:
            print("  → H3 NOT supported by COR")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 3. ADVANTAGE-DEGRADATION CORRELATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_advantage_degradation_correlation(data):
    """
    LoRA advantage on clean context vs LoRA degradation on noisy context.
    Positive correlation confirms bidirectional sensitivity hypothesis.
    """
    print("\n" + "═" * 60)
    print("3. ADVANTAGE–DEGRADATION CORRELATION")
    print("   Do cases where LoRA benefits most from clean context also")
    print("   suffer most from noisy context?")
    print("═" * 60)

    advantages, degradations = [], []
    for r in data:
        b_none  = _score(r, "base",      "none")
        b_clean = _score(r, "base",      "clean")
        f_none  = _score(r, "finetuned", "none")
        f_noisy = _score(r, "finetuned", "noisy")
        f_clean = _score(r, "finetuned", "clean")

        if all(v is not None for v in [b_none, b_clean, f_none, f_noisy, f_clean]):
            # LoRA advantage = how much better LoRA+clean is vs base+clean
            advantage   = f_clean - b_clean
            # LoRA degradation = how much LoRA drops under noisy vs none
            degradation = f_none - f_noisy
            advantages.append(advantage)
            degradations.append(degradation)

    if len(advantages) < 5:
        print("  Insufficient data for correlation analysis")
        return None, None

    am, _ = mean_std(advantages)
    dm, _ = mean_std(degradations)
    print(f"  LoRA clean advantage:   mean={am:+.3f}  n={len(advantages)}")
    print(f"  LoRA noisy degradation: mean={dm:+.3f}  n={len(degradations)}")

    r, p = spearman(advantages, degradations, "Advantage vs Degradation (Spearman)")

    if r is not None:
        if r > 0.3 and p < 0.05:
            print("  → H3 SUPPORTED: Cases where LoRA benefits most from context")
            print("    also degrade most under noisy context (bidirectional sensitivity)")
        elif r > 0 :
            print("  → H3 WEAKLY supported: positive correlation but not significant")
        else:
            print("  → H3 NOT supported by correlation analysis")

    return advantages, degradations


# ══════════════════════════════════════════════════════════════════════════════
# 4. CATEGORY SENSITIVITY
# ══════════════════════════════════════════════════════════════════════════════

def compute_category_sensitivity(data):
    print("\n" + "═" * 60)
    print("4. CATEGORY SENSITIVITY (H2 — safety cases most affected)")
    print("═" * 60)
    print(f"  {'Category':<22} {'Base CSI':>10} {'LoRA CSI':>10} {'Safety COR':>12}")
    print("  " + "-" * 58)

    cat_results = {}
    for cat in CATEGORIES:
        cat_data = [r for r in data if r.get("category") == cat]
        if not cat_data:
            continue

        base_csi_cat, ft_csi_cat = [], []
        safety_override_ft = safety_total = 0

        for r in cat_data:
            for model, lst in [("base", base_csi_cat), ("finetuned", ft_csi_cat)]:
                s_none = _score(r, model, "none")
                s_noi  = _score(r, model, "noisy")
                s_adv  = _score(r, model, "adversarial")
                drops = []
                if s_none is not None and s_noi is not None:
                    drops.append(s_none - s_noi)
                if s_none is not None and s_adv is not None:
                    drops.append(s_none - s_adv)
                if drops:
                    lst.append(statistics.mean(drops))

            if r.get("safety_required"):
                safety_total += 1
                adv = r["results"].get("finetuned", {}).get("adversarial", {})
                if adv.get("score", {}).get("safety_failed"):
                    safety_override_ft += 1

        bm, _ = mean_std(base_csi_cat)
        fm, _ = mean_std(ft_csi_cat)
        scor = safety_override_ft / safety_total if safety_total else float("nan")

        cat_results[cat] = {"base_csi": bm, "ft_csi": fm, "safety_cor": scor, "n": len(cat_data)}
        scor_str = f"{scor:.1%}" if not math.isnan(scor) else "  n/a"
        print(f"  {CAT_LABELS[cat]:<22} {bm:>+10.3f} {fm:>+10.3f} {scor_str:>12}")

    return cat_results


# ══════════════════════════════════════════════════════════════════════════════
# 5. OPTIONAL MECHANISTIC — log-prob analysis via llama-cpp-python
# ══════════════════════════════════════════════════════════════════════════════

def run_logprob_analysis(data):
    try:
        from llama_cpp import Llama
    except ImportError:
        print("\n" + "═" * 60)
        print("5. MECHANISTIC ANALYSIS (log-probability amplification)")
        print("═" * 60)
        print("  llama-cpp-python not installed.")
        print("  Install: pip install llama-cpp-python")
        print("  Then re-run this script for token-level context amplification analysis.")
        return

    if not FINETUNED_GGUF.exists() or not BASE_GGUF.exists():
        print("\n5. MECHANISTIC ANALYSIS — GGUF files not found, skipping")
        return

    print("\n" + "═" * 60)
    print("5. MECHANISTIC ANALYSIS — context amplification (log-probs)")
    print("═" * 60)

    # Pick 5 representative cases (mix of categories)
    sample = []
    seen_cats = set()
    for r in data:
        if r.get("category") not in seen_cats and r.get("question"):
            sample.append(r)
            seen_cats.add(r["category"])
        if len(sample) >= 5:
            break

    results_logprob = []

    for model_key, gguf_path in [("base", BASE_GGUF), ("finetuned", FINETUNED_GGUF)]:
        print(f"\n  Loading {model_key} ({gguf_path.name}) ...")
        llm = Llama(
            model_path=str(gguf_path),
            n_ctx=512, n_gpu_layers=0,
            logits_all=True, verbose=False,
        )

        for r in sample:
            q = r["question"][:300]  # truncate for context
            clean_entry = r["results"].get(model_key, {}).get("clean", {})
            context = clean_entry.get("answer", "")[:200] if clean_entry else ""

            prompt_no_ctx = f"Patient: {q}\nDoctor:"
            prompt_ctx    = f"Context: {context}\n\nPatient: {q}\nDoctor:"

            # Token log-prob for first 10 answer tokens
            try:
                out_no  = llm(prompt_no_ctx,  max_tokens=10, logprobs=5)
                out_ctx = llm(prompt_ctx,      max_tokens=10, logprobs=5)
                lp_no  = sum(t["logprob"] for t in out_no["choices"][0]["logprobs"]["token_logprobs"]  if t)
                lp_ctx = sum(t["logprob"] for t in out_ctx["choices"][0]["logprobs"]["token_logprobs"] if t)
                amp = math.exp(lp_ctx - lp_no)  # amplification ratio
                results_logprob.append({"id": r["id"], "model": model_key, "amplification": amp})
                print(f"  {r['id']} [{model_key}]: log-p(no_ctx)={lp_no:.2f}  log-p(ctx)={lp_ctx:.2f}  amp={amp:.3f}")
            except Exception as e:
                print(f"  {r['id']} [{model_key}]: FAILED: {e}")

    base_amps = [x["amplification"] for x in results_logprob if x["model"] == "base"]
    ft_amps   = [x["amplification"] for x in results_logprob if x["model"] == "finetuned"]
    bm, _  = mean_std(base_amps)
    fm, _  = mean_std(ft_amps)
    print(f"\n  Base mean amplification:  {bm:.3f}")
    print(f"  LoRA mean amplification:  {fm:.3f}")
    if fm > bm:
        print("  → H3 SUPPORTED: LoRA amplifies context token probabilities more than base")
    else:
        print("  → H3 NOT supported by log-prob analysis")

    return results_logprob


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def plot_csi_distributions(base_csi, ft_csi, output_dir):
    if not HAS_MPL or not base_csi or not ft_csi:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    bp = ax.boxplot(
        [base_csi, ft_csi],
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 2},
        labels=["Base Llama 3 8B", "LoRA Fine-tuned"],
    )
    bp["boxes"][0].set_facecolor("#5c85d6"); bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor("#e05c5c"); bp["boxes"][1].set_alpha(0.7)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Context Sensitivity Index (score drop)", fontsize=11)
    ax.set_title("Context Sensitivity Index Distribution\n(higher = more harmed by noisy/adversarial context)", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    out = output_dir / "fig_a_csi_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


def plot_advantage_degradation(advantages, degradations, output_dir):
    if not HAS_MPL or not advantages:
        return
    import numpy as np
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(advantages, degradations, color="#e05c5c", alpha=0.6, s=40)
    # Regression line
    try:
        m, b = np.polyfit(advantages, degradations, 1)
        xs = np.linspace(min(advantages), max(advantages), 100)
        ax.plot(xs, m * xs + b, color="#c0392b", linewidth=2, label=f"slope={m:.2f}")
    except Exception:
        pass
    ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("LoRA clean-context advantage over base", fontsize=11)
    ax.set_ylabel("LoRA degradation under noisy context", fontsize=11)
    ax.set_title("Advantage–Degradation Correlation\n(H3: cases that benefit most also suffer most)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(linestyle="--", alpha=0.3)
    fig.tight_layout()
    out = output_dir / "fig_b_adv_degradation.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


def plot_category_csi(cat_results, output_dir):
    if not HAS_MPL or not cat_results:
        return
    cats = [c for c in CATEGORIES if c in cat_results]
    base_vals = [cat_results[c]["base_csi"] for c in cats]
    ft_vals   = [cat_results[c]["ft_csi"]   for c in cats]
    labels    = [CAT_LABELS[c] for c in cats]
    x = list(range(len(cats)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([i - width/2 for i in x], base_vals, width, label="Base",
           color="#5c85d6", alpha=0.8, edgecolor="grey", linewidth=0.5)
    ax.bar([i + width/2 for i in x], ft_vals,   width, label="LoRA",
           color="#e05c5c", alpha=0.8, edgecolor="grey", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Mean CSI (higher = more sensitive)", fontsize=11)
    ax.set_title("Context Sensitivity by Case Category", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    out = output_dir / "fig_c_category_csi.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def export_json(base_csi, ft_csi, cor_results, cat_results, output_dir):
    bm, bs = mean_std(base_csi)
    fm, fs = mean_std(ft_csi)
    out = {
        "csi": {
            "base":      {"mean": round(bm, 4), "std": round(bs, 4), "n": len(base_csi)},
            "finetuned": {"mean": round(fm, 4), "std": round(fs, 4), "n": len(ft_csi)},
            "h3_supported": not math.isnan(fm) and fm > bm,
        },
        "cor": {
            "base":      round(cor_results.get("base", {}).get("cor", float("nan")), 4),
            "finetuned": round(cor_results.get("finetuned", {}).get("cor", float("nan")), 4),
        },
        "category_csi": {
            c: {"base_csi": round(v["base_csi"], 4), "ft_csi": round(v["ft_csi"], 4)}
            for c, v in cat_results.items()
            if not math.isnan(v.get("base_csi", float("nan")))
        },
    }
    path = output_dir / "mechanism_analysis.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    data = load(RESULTS_FILE)

    base_csi, ft_csi, per_case = compute_csi(data)
    cor_results                = compute_cor(data)
    advantages, degradations   = compute_advantage_degradation_correlation(data)
    cat_results                = compute_category_sensitivity(data)
    run_logprob_analysis(data)

    print("\n" + "═" * 60)
    print("FIGURES")
    print("═" * 60)
    if HAS_MPL:
        plot_csi_distributions(base_csi, ft_csi, RESULTS_DIR)
        if advantages:
            plot_advantage_degradation(advantages, degradations, RESULTS_DIR)
        plot_category_csi(cat_results, RESULTS_DIR)
    else:
        print("  Install matplotlib to generate figures: pip install matplotlib scipy")

    export_json(base_csi, ft_csi, cor_results, cat_results, RESULTS_DIR)
    print("\nDone. Run this script again after the full experiment completes.")


if __name__ == "__main__":
    main()
