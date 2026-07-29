"""
analyze_research.py — Statistical analysis and visualisation
============================================================
Loads research_results.json and produces:

  1. Summary table — mean ± std for every model × condition cell
  2. Degradation analysis — rate, magnitude, per-category breakdown
  3. Safety suppression analysis — H2 test
  4. Statistical tests — Mann-Whitney U on LoRA vs base drop (H1)
  5. Key figure — Performance vs Context Condition (saves to results/)
  6. Heatmap — per-case degradation flags

Run after research_experiment.py has completed.
Requires: matplotlib, scipy (pip install matplotlib scipy)
Without them, only text tables and stats print; graphs are skipped.
"""
import json
import math
import statistics
from pathlib import Path

RESULTS_FILE = Path("research_results.json")
RESULTS_DIR  = Path("results")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[warn] matplotlib not found — graphs will be skipped")

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[warn] scipy not found — statistical tests will be skipped")


MODELS     = ["base", "finetuned"]
CONDITIONS = ["none", "clean", "noisy", "adversarial"]
CATEGORIES = ["safety_emergency", "general_covered", "corpus_gap", "edge_case"]

CONDITION_LABELS = {
    "none":        "No Retrieval",
    "clean":       "Clean RAG",
    "noisy":       "Noisy RAG",
    "adversarial": "Adversarial RAG",
}
MODEL_LABELS = {
    "base":      "Base Llama 3 8B",
    "finetuned": "LoRA Fine-tuned",
}
COLORS = {
    "base":      "#5c85d6",
    "finetuned": "#e05c5c",
}


# ── Data loading ───────────────────────────────────────────────────────────────

def load(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run research_experiment.py first.")
    with open(path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} cases from {path}")
    return data


def score_matrix(data: list[dict]) -> dict:
    """
    Returns {model: {condition: [score, ...]}} — only valid (non-error) scores.
    """
    m = {model: {cond: [] for cond in CONDITIONS} for model in MODELS}
    for r in data:
        for model in MODELS:
            for cond in CONDITIONS:
                entry = r["results"].get(model, {}).get(cond, {})
                if "score" in entry:
                    m[model][cond].append(entry["score"]["total"])
    return m


def category_matrix(data: list[dict]) -> dict:
    """Returns {category: {model: {condition: [scores]}}}."""
    cats = {cat: {model: {cond: [] for cond in CONDITIONS} for model in MODELS}
            for cat in CATEGORIES}
    for r in data:
        cat = r.get("category", "unknown")
        if cat not in cats:
            cats[cat] = {model: {cond: [] for cond in CONDITIONS} for model in MODELS}
        for model in MODELS:
            for cond in CONDITIONS:
                entry = r["results"].get(model, {}).get(cond, {})
                if "score" in entry:
                    cats[cat][model][cond].append(entry["score"]["total"])
    return cats


# ── Helpers ────────────────────────────────────────────────────────────────────

def mean_std(lst: list) -> tuple[float, float]:
    if not lst:
        return float("nan"), float("nan")
    m = statistics.mean(lst)
    s = statistics.stdev(lst) if len(lst) > 1 else 0.0
    return m, s


def fmt(lst: list) -> str:
    m, s = mean_std(lst)
    n = len(lst)
    if math.isnan(m):
        return "  ---  "
    return f"{m:.2f}±{s:.2f} (n={n})"


def mannwhitney(a: list, b: list, label: str):
    if not HAS_SCIPY:
        return
    if len(a) < 3 or len(b) < 3:
        print(f"  {label}: too few samples (n={len(a)}, {len(b)})")
        return
    stat, p = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    r = 1 - (2 * stat) / (len(a) * len(b))
    print(f"  {label}: U={stat:.0f}  p={p:.4f} {sig}  r={r:.3f}")


# ── Table 1: Summary statistics ───────────────────────────────────────────────

def print_summary_table(matrix: dict):
    print("\n" + "═" * 70)
    print("TABLE 1 — Mean ± Std rubric score (0–8) per model × condition")
    print("═" * 70)
    header = f"{'Condition':<20}" + "".join(f"  {MODEL_LABELS[m]:<24}" for m in MODELS)
    print(header)
    print("-" * 70)
    for cond in CONDITIONS:
        row = f"{CONDITION_LABELS[cond]:<20}"
        for model in MODELS:
            row += f"  {fmt(matrix[model][cond]):<24}"
        print(row)
    print()


# ── Table 2: Degradation analysis ────────────────────────────────────────────

def print_degradation_table(data: list[dict]):
    print("═" * 70)
    print("TABLE 2 — Degradation analysis (condition vs no-retrieval baseline)")
    print("═" * 70)
    print(f"{'Model':<14} {'Condition':<18} {'Degradation rate':>18} {'Mean drop':>12}")
    print("-" * 70)

    for model in MODELS:
        for cond in ["clean", "noisy", "adversarial"]:
            flags, drops = [], []
            for r in data:
                baseline = r["results"].get(model, {}).get("none", {}).get("score", {}).get("total")
                entry = r["results"].get(model, {}).get(cond, {})
                if "score" in entry and baseline is not None:
                    s = entry["score"]["total"]
                    flags.append(1 if s < baseline else 0)
                    drops.append(s - baseline)
            if not flags:
                continue
            rate = sum(flags) / len(flags)
            mean_drop, _ = mean_std(drops)
            print(f"{MODEL_LABELS[model]:<14} {CONDITION_LABELS[cond]:<18} "
                  f"{rate:>16.1%}  {mean_drop:>+10.2f}")
    print()


# ── Table 3: Below-base rate ──────────────────────────────────────────────────

def print_below_base_table(data: list[dict]):
    print("═" * 70)
    print("TABLE 3 — Below-base rate (score < base model with no retrieval)")
    print("═" * 70)
    base_nones = {r["id"]: r["results"].get("base", {}).get("none", {}).get("score", {}).get("total")
                  for r in data}

    for model in MODELS:
        for cond in CONDITIONS:
            flags = []
            for r in data:
                base_none = base_nones.get(r["id"])
                entry = r["results"].get(model, {}).get(cond, {})
                if "score" in entry and base_none is not None:
                    flags.append(1 if entry["score"]["total"] < base_none else 0)
            if not flags:
                continue
            rate = sum(flags) / len(flags)
            print(f"  {MODEL_LABELS[model]:<24} {CONDITION_LABELS[cond]:<18} below-base: {rate:.1%}")
    print()


# ── Table 4: Safety suppression (H2) ─────────────────────────────────────────

def print_safety_table(data: list[dict]):
    print("═" * 70)
    print("TABLE 4 — Safety suppression (H2): safety_required cases only")
    print("═" * 70)
    safety_cases = [r for r in data if r.get("safety_required")]
    print(f"  Safety-required cases: {len(safety_cases)}")

    for model in MODELS:
        for cond in CONDITIONS:
            fails = total = 0
            for r in safety_cases:
                entry = r["results"].get(model, {}).get(cond, {})
                if "score" in entry:
                    total += 1
                    if entry["score"].get("safety_failed"):
                        fails += 1
            if total == 0:
                continue
            print(f"  {MODEL_LABELS[model]:<24} {CONDITION_LABELS[cond]:<18} "
                  f"safety failures: {fails}/{total} ({fails/total:.1%})")
    print()


# ── Per-category breakdown ────────────────────────────────────────────────────

def print_category_breakdown(cat_matrix: dict):
    print("═" * 70)
    print("TABLE 5 — Per-category mean scores (noisy condition)")
    print("═" * 70)
    cat_labels = {
        "safety_emergency": "Safety emergency",
        "general_covered":  "General (covered)",
        "corpus_gap":       "Corpus gap",
        "edge_case":        "Edge case",
    }
    print(f"{'Category':<22} {'Base noisy':>12} {'LoRA noisy':>12} {'LoRA drop':>12}")
    print("-" * 70)
    for cat, label in cat_labels.items():
        if cat not in cat_matrix:
            continue
        b_noisy = cat_matrix[cat]["base"]["noisy"]
        f_noisy = cat_matrix[cat]["finetuned"]["noisy"]
        f_none  = cat_matrix[cat]["finetuned"]["none"]
        bm, _  = mean_std(b_noisy)
        fm, _  = mean_std(f_noisy)
        fn, _  = mean_std(f_none)
        drop = fm - fn if not math.isnan(fm) and not math.isnan(fn) else float("nan")
        print(f"{label:<22} {bm:>10.2f}   {fm:>10.2f}   {drop:>+10.2f}")
    print()


# ── H1 statistical test ───────────────────────────────────────────────────────

def test_h1(data: list[dict]):
    """
    H1: LoRA drop (none→noisy) > base drop (none→noisy)
    Test by collecting per-case (none - noisy) drop for each model,
    then Mann-Whitney U on the two distributions.
    """
    print("═" * 70)
    print("STATISTICAL TESTS — H1: LoRA more sensitive than base to noisy context")
    print("═" * 70)

    base_drops, ft_drops = [], []
    for r in data:
        b_none = r["results"].get("base", {}).get("none", {}).get("score", {}).get("total")
        b_noi  = r["results"].get("base", {}).get("noisy", {}).get("score", {}).get("total")
        f_none = r["results"].get("finetuned", {}).get("none", {}).get("score", {}).get("total")
        f_noi  = r["results"].get("finetuned", {}).get("noisy", {}).get("score", {}).get("total")

        if b_none is not None and b_noi is not None:
            base_drops.append(b_none - b_noi)
        if f_none is not None and f_noi is not None:
            ft_drops.append(f_none - f_noi)

    bm, bs = mean_std(base_drops)
    fm, fs = mean_std(ft_drops)
    print(f"  Base  drop distribution: mean={bm:+.3f}  std={bs:.3f}  n={len(base_drops)}")
    print(f"  LoRA  drop distribution: mean={fm:+.3f}  std={fs:.3f}  n={len(ft_drops)}")

    mannwhitney(base_drops, ft_drops, "Base drops vs LoRA drops (none→noisy)")

    # Also for adversarial
    base_drops_adv, ft_drops_adv = [], []
    for r in data:
        b_none = r["results"].get("base", {}).get("none", {}).get("score", {}).get("total")
        b_adv  = r["results"].get("base", {}).get("adversarial", {}).get("score", {}).get("total")
        f_none = r["results"].get("finetuned", {}).get("none", {}).get("score", {}).get("total")
        f_adv  = r["results"].get("finetuned", {}).get("adversarial", {}).get("score", {}).get("total")

        if b_none is not None and b_adv is not None:
            base_drops_adv.append(b_none - b_adv)
        if f_none is not None and f_adv is not None:
            ft_drops_adv.append(f_none - f_adv)

    bm2, bs2 = mean_std(base_drops_adv)
    fm2, fs2 = mean_std(ft_drops_adv)
    print(f"\n  Base  adv-drop: mean={bm2:+.3f}  std={bs2:.3f}  n={len(base_drops_adv)}")
    print(f"  LoRA  adv-drop: mean={fm2:+.3f}  std={fs2:.3f}  n={len(ft_drops_adv)}")
    mannwhitney(base_drops_adv, ft_drops_adv, "Base drops vs LoRA drops (none→adversarial)")
    print()


# ── Figure 1: Performance vs Condition ───────────────────────────────────────

def plot_key_figure(matrix: dict, output_dir: Path):
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    x = list(range(len(CONDITIONS)))
    labels = [CONDITION_LABELS[c] for c in CONDITIONS]

    for model in MODELS:
        means = []
        errs  = []
        for cond in CONDITIONS:
            m, s = mean_std(matrix[model][cond])
            means.append(m)
            errs.append(s)

        ax.errorbar(
            x, means, yerr=errs,
            label=MODEL_LABELS[model],
            color=COLORS[model],
            marker="o", linewidth=2.5, markersize=8,
            capsize=5, elinewidth=1.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Rubric Score (0–8)", fontsize=12)
    ax.set_xlabel("Retrieval Condition", fontsize=12)
    ax.set_title(
        "Performance vs Context Condition\nBase Llama 3 8B vs LoRA Fine-tuned",
        fontsize=13, fontweight="bold",
    )
    ax.set_ylim(0, 8.5)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(fontsize=11)
    fig.tight_layout()

    out = output_dir / "fig1_performance_vs_condition.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 2: Per-category bar chart ─────────────────────────────────────────

def plot_category_bars(cat_matrix: dict, output_dir: Path):
    if not HAS_MPL:
        return

    cat_order = ["safety_emergency", "general_covered", "corpus_gap", "edge_case"]
    cat_labels_short = {
        "safety_emergency": "Safety",
        "general_covered":  "General",
        "corpus_gap":       "Corpus Gap",
        "edge_case":        "Edge Case",
    }
    cond_subset = ["none", "noisy", "adversarial"]
    cond_colors = {"none": "#aad4f5", "noisy": "#f5c6aa", "adversarial": "#f5aaaa"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax_idx, model in enumerate(MODELS):
        ax = axes[ax_idx]
        width = 0.25
        x = list(range(len(cat_order)))

        for i, cond in enumerate(cond_subset):
            means = []
            errs  = []
            for cat in cat_order:
                if cat in cat_matrix:
                    m, s = mean_std(cat_matrix[cat][model][cond])
                else:
                    m, s = float("nan"), 0.0
                means.append(m)
                errs.append(s)

            offset = (i - 1) * width
            bars = ax.bar(
                [xi + offset for xi in x], means, width,
                label=CONDITION_LABELS[cond],
                color=cond_colors[cond],
                edgecolor="grey", linewidth=0.5,
                yerr=errs, capsize=3, error_kw={"linewidth": 1},
            )

        ax.set_title(MODEL_LABELS[model], fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([cat_labels_short[c] for c in cat_order], fontsize=10)
        ax.set_ylabel("Rubric Score (0–8)", fontsize=11)
        ax.set_ylim(0, 9)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(fontsize=9)

    fig.suptitle("Score by Category and Retrieval Condition", fontsize=13, fontweight="bold")
    fig.tight_layout()

    out = output_dir / "fig2_category_breakdown.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 3: Degradation heatmap ────────────────────────────────────────────

def plot_degradation_heatmap(data: list[dict], output_dir: Path):
    if not HAS_MPL:
        return

    conds = ["clean", "noisy", "adversarial"]
    n = len(data)
    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, n * 0.22 + 2)), sharey=True)

    for ax_idx, model in enumerate(MODELS):
        ax = axes[ax_idx]
        matrix = []
        ylabels = []

        for r in data:
            row = []
            for cond in conds:
                entry = r["results"].get(model, {}).get(cond, {})
                if "score" in entry:
                    row.append(entry["score"].get("degradation_flag", 0))
                else:
                    row.append(-1)
            matrix.append(row)
            ylabels.append(f"{r['id']} {r['topic'][:20]}")

        import numpy as np
        mat = np.array(matrix, dtype=float)
        mat[mat == -1] = 0.5

        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([CONDITION_LABELS[c] for c in conds], fontsize=9)
        ax.set_yticks(range(n))
        ax.set_yticklabels(ylabels, fontsize=7)
        ax.set_title(MODEL_LABELS[model], fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=ax, label="Degradation (1=worse than no-RAG)")

    fig.suptitle("Per-Case Degradation Flags", fontsize=12, fontweight="bold")
    fig.tight_layout()

    out = output_dir / "fig3_degradation_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 4: Box plots ───────────────────────────────────────────────────────

def plot_boxplots(matrix: dict, output_dir: Path):
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax_idx, model in enumerate(MODELS):
        ax = axes[ax_idx]
        data_by_cond = [matrix[model][cond] for cond in CONDITIONS]
        bp = ax.boxplot(
            data_by_cond,
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 2},
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(COLORS[model])
            patch.set_alpha(0.7)

        ax.set_xticks(range(1, len(CONDITIONS) + 1))
        ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITIONS], fontsize=10)
        ax.set_title(MODEL_LABELS[model], fontsize=12, fontweight="bold")
        ax.set_ylabel("Rubric Score (0–8)", fontsize=11)
        ax.set_ylim(-0.5, 8.5)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("Score Distribution by Condition", fontsize=13, fontweight="bold")
    fig.tight_layout()

    out = output_dir / "fig4_boxplots.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ── JS export (for research_report.html) ───────────────────────────────────────

def export_js(matrix: dict, data: list, output_dir: Path) -> None:
    """Write results/research_data.js so the HTML report can load real data."""
    def mean(lst): return sum(lst) / len(lst) if lst else 0.0
    def std(lst):
        if len(lst) < 2:
            return 0.0
        m = mean(lst)
        return math.sqrt(sum((x - m) ** 2 for x in lst) / (len(lst) - 1))

    base_means = {c: round(mean(matrix["base"][c]), 3)      for c in CONDITIONS}
    ft_means   = {c: round(mean(matrix["finetuned"][c]), 3) for c in CONDITIONS}
    base_stds  = {c: round(std(matrix["base"][c]), 3)       for c in CONDITIONS}
    ft_stds    = {c: round(std(matrix["finetuned"][c]), 3)  for c in CONDITIONS}

    # Degradation rate: fraction of cases where score < no-retrieval score
    def degradation_rate(model_key, condition):
        n = 0
        total = 0
        for case in data:
            s_none = case.get(model_key, {}).get("none", {}).get("score")
            s_cond = case.get(model_key, {}).get(condition, {}).get("score")
            if s_none is not None and s_cond is not None:
                total += 1
                if s_cond < s_none:
                    n += 1
        return round(n / total, 3) if total else 0.0

    # Safety suppression rate: fraction of safety_emergency cases that lost urgency flag
    def safety_suppression(model_key, condition):
        n = 0
        total = 0
        for case in data:
            if case.get("category") != "safety_emergency":
                continue
            s_none = case.get(model_key, {}).get("none", {}).get("score")
            s_cond = case.get(model_key, {}).get(condition, {}).get("score")
            if s_none is not None and s_cond is not None:
                total += 1
                if s_cond < s_none:
                    n += 1
        return round(n / total, 3) if total else 0.0

    payload = {
        "base":      base_means,
        "finetuned": ft_means,
        "base_std":  base_stds,
        "ft_std":    ft_stds,
        "degradation_rate": {
            "base": [0.0] + [degradation_rate("base", c) for c in CONDITIONS[1:]],
            "ft":   [0.0] + [degradation_rate("finetuned", c) for c in CONDITIONS[1:]],
        },
        "safety_suppression": {
            "base": [0.0] + [safety_suppression("base", c) for c in CONDITIONS[1:]],
            "ft":   [0.0] + [safety_suppression("finetuned", c) for c in CONDITIONS[1:]],
        },
    }

    js_path = output_dir / "research_data.js"
    with open(js_path, "w") as f:
        f.write("// Auto-generated by analyze_research.py — do not edit\n")
        f.write(f"const RESEARCH_DATA = {json.dumps(payload, indent=2)};\n")
    print(f"  Exported {js_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    data = load(RESULTS_FILE)
    matrix     = score_matrix(data)
    cat_matrix = category_matrix(data)

    # Sanity check
    total_scores = sum(len(matrix[m][c]) for m in MODELS for c in CONDITIONS)
    expected     = len(data) * len(MODELS) * len(CONDITIONS)
    print(f"Scores collected: {total_scores} / {expected} expected "
          f"({'complete' if total_scores == expected else 'INCOMPLETE'})\n")

    print_summary_table(matrix)
    print_degradation_table(data)
    print_below_base_table(data)
    print_safety_table(data)
    print_category_breakdown(cat_matrix)
    test_h1(data)

    print("═" * 70)
    print("FIGURES")
    print("═" * 70)

    if HAS_MPL:
        plot_key_figure(matrix, RESULTS_DIR)
        plot_category_bars(cat_matrix, RESULTS_DIR)
        try:
            plot_degradation_heatmap(data, RESULTS_DIR)
        except Exception as e:
            print(f"  [heatmap skipped: {e}]")
        plot_boxplots(matrix, RESULTS_DIR)
        print(f"\nAll figures saved to {RESULTS_DIR}/")
    else:
        print("  Install matplotlib to generate figures: pip install matplotlib")

    print("\nExporting JS data for HTML report...")
    export_js(matrix, data, RESULTS_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
