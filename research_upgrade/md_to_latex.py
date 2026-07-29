"""Convert OpenHealth_Research_Paper_vNext.md -> PMLR-style LaTeX (ML4H 2026).
Pandoc-free. Handles the paper's unicode + LaTeX specials, markdown tables, and
splits supplementary sections (§4.4/4.5/4.7/4.9/4.12) into an Appendix.

Output: OpenHealth_Research_Paper_vNext.tex  (compile after dropping in the official
ml4h2026/jmlr style; a generic PMLR-ish preamble is included so it compiles standalone).
Run from C:\\OpenHealth\\research_upgrade.
"""
import re

SRC = "OpenHealth_Research_Paper_vNext.md"
OUT = "OpenHealth_Research_Paper_vNext.tex"

# sections (by "### 4.x title") that move to the appendix for the 8-page limit
APPENDIX_SUBSECS = {"4.4", "4.5", "4.7", "4.9", "4.12"}

UNI = [
    ("κ", r"\(\kappa\)"), ("α", r"\(\alpha\)"), ("τ", r"\(\tau\)"), ("β", r"\(\beta\)"),
    ("→", r"\(\rightarrow\)"), ("↔", r"\(\leftrightarrow\)"), ("×", r"\(\times\)"),
    ("≈", r"\(\approx\)"), ("≤", r"\(\leq\)"), ("≥", r"\(\geq\)"), ("∧", r"\(\wedge\)"),
    ("−", "-"), ("–", "--"), ("—", "---"), ("§", r"\S{}"), ("’", "'"), ("‘", "`"),
    ("“", "``"), ("”", "''"), ("…", r"\ldots{}"), ("≠", r"\(\neq\)"),
    ("±", r"\(\pm\)"), ("·", r"\(\cdot\)"), ("ï", r'\"{\i}'),
]


def uni(t):
    for a, b in UNI:
        t = t.replace(a, b)
    return t


def inline(t):
    """Correct order: protect code spans, escape RAW LaTeX specials, THEN insert
    intentional LaTeX (unicode, bold/italic) so it is never re-escaped."""
    codes = []
    def _code(m):
        inner = m.group(1)
        for a, b in [("\\", r"\textbackslash{}"), ("%", r"\%"), ("&", r"\&"),
                     ("#", r"\#"), ("$", r"\$"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}")]:
            inner = inner.replace(a, b)
        codes.append(inner)
        return f"\x00{len(codes)-1}\x00"
    t = re.sub(r"`([^`]+)`", _code, t)
    # 1. escape raw LaTeX specials on the prose (backslash first)
    t = t.replace("\\", r"\textbackslash{}")
    for ch in ["%", "&", "#", "$"]:
        t = t.replace(ch, "\\" + ch)
    t = t.replace("_", r"\_")
    # 2. unicode -> LaTeX (clean backslashes; not re-escaped)
    t = uni(t)
    # 3. bold/italic wrap already-escaped content
    t = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", t)
    t = re.sub(r"\*([^*]+)\*", r"\\textit{\1}", t)
    # 4. restore code spans
    t = re.sub(r"\x00(\d+)\x00", lambda m: r"\texttt{" + codes[int(m.group(1))] + "}", t)
    return t


def table(rows):
    # rows: list of markdown "| a | b |" lines (header, sep, data...)
    def cells(r):
        return [c.strip() for c in r.strip().strip("|").split("|")]
    header = cells(rows[0])
    ncol = len(header)
    out = [r"\begin{table}[t]\centering\small", r"\begin{tabular}{" + "l" * ncol + "}", r"\toprule"]
    out.append(" & ".join(inline(h) for h in header) + r" \\ \midrule")
    for r in rows[2:]:
        out.append(" & ".join(inline(c) for c in cells(r)) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def main():
    raw = open(SRC, encoding="utf-8").read().split("\n")
    title = next(l[2:].strip() for l in raw if l.startswith("# "))
    body, i, in_appendix = [], 0, False
    appendix = []
    def emit(s):
        (appendix if in_appendix else body).append(s)

    lines = raw
    n = len(lines)
    idx = 0
    abstract = []
    in_abstract = False
    while idx < n:
        l = lines[idx]
        if l.startswith("# "):
            idx += 1; continue
        if l.strip().startswith("> **Provenance note"):
            idx += 1; continue  # drop provenance note
        if l.startswith("## Abstract"):
            in_abstract = True; idx += 1; continue
        if in_abstract:
            if l.startswith("## "):
                in_abstract = False  # fall through to handle this heading
            else:
                if l.strip():
                    abstract.append(inline(l.strip()))
                idx += 1; continue
        # headings
        m2 = re.match(r"## (\d+)\.\s*(.*)", l)
        m2b = re.match(r"## (.*)", l)
        m3 = re.match(r"### (\d+\.\d+)\s*(.*)", l)
        if m3:
            num, ttl = m3.group(1), m3.group(2)
            if num in APPENDIX_SUBSECS and not in_appendix:
                pass  # appendix subsecs are emitted into the appendix list
            target_appendix = num in APPENDIX_SUBSECS
            (appendix if target_appendix else body).append(r"\subsection{" + inline(ttl) + "}")
            # remember where subsequent paragraphs go
            emit_target = "appendix" if target_appendix else "body"
            idx += 1
            # consume until next heading
            while idx < n and not lines[idx].startswith("#"):
                para = lines[idx]
                if para.strip().startswith("|"):
                    tb = []
                    while idx < n and lines[idx].strip().startswith("|"):
                        tb.append(lines[idx]); idx += 1
                    (appendix if target_appendix else body).append(table(tb)); continue
                if para.strip():
                    (appendix if target_appendix else body).append(inline(para.strip()) + "\n")
                idx += 1
            continue
        if m2 and not l.startswith("###"):
            (body).append(r"\section{" + inline(m2.group(2)) + "}")
            idx += 1; continue
        if m2b and not l.startswith("###") and not re.match(r"## \d", l):
            ttl = m2b.group(1)
            body.append(r"\section{" + inline(ttl) + "}")
            idx += 1; continue
        # table block at top level
        if l.strip().startswith("|"):
            tb = []
            while idx < n and lines[idx].strip().startswith("|"):
                tb.append(lines[idx]); idx += 1
            body.append(table(tb)); continue
        if l.strip():
            body.append(inline(l.strip()) + "\n")
        idx += 1

    tex = []
    tex.append(r"""% ML4H 2026 Proceedings (PMLR). Replace the preamble below with the official
% ml4h2026/jmlr style at submission; this generic PMLR-ish preamble compiles standalone.
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs,graphicx,amsmath,amssymb,hyperref,microtype,xcolor}
\usepackage[numbers]{natbib}
\graphicspath{{../results/}}
\title{""" + inline(title) + r"""}
\author{Anonymous submission --- ML4H 2026 Proceedings Track (double-blind)}
\date{}
\begin{document}
\maketitle
\begin{abstract}
""" + " ".join(abstract) + r"""
\end{abstract}
""")
    tex.append("\n\n".join(body))
    tex.append(r"""
\section*{Figures}
\begin{figure}[t]\centering\includegraphics[width=.7\linewidth]{fig_mechanism_clean_s101.png}
\caption{Keyword count vs.\ judged escalation: the rubric's operating point shifts for the fine-tune (\S4.3).}\end{figure}
\begin{figure}[t]\centering\includegraphics[width=\linewidth]{fig_generalization.png}
\caption{Metric-gaming is corpus-specific (\S4.11): the primary pair's escalation stays flat while its rubric validity collapses; the independent pair's escalation rises while validity barely moves.}\end{figure}

\appendix
\section{Supplementary results}
""")
    tex.append("\n\n".join(appendix))
    tex.append(r"""
\bibliographystyle{plainnat}
% References: see comparison_matrix.md for the full verified list (38 entries incl. the
% Goodhart/reward-hacking/shortcut-learning additions); format to PMLR .bib at submission.
\begin{thebibliography}{99}
\bibitem{strathern97} M. Strathern. `Improving ratings': audit in the British University system. \emph{European Review}, 5(3):305--321, 1997.
\bibitem{amodei16} D. Amodei et al. Concrete Problems in AI Safety. arXiv:1606.06565, 2016.
\bibitem{krakovna20} V. Krakovna et al. Specification gaming: the flip side of AI ingenuity. DeepMind, 2020.
\bibitem{pan22} A. Pan, K. Bhatia, J. Steinhardt. The Effects of Reward Misspecification. ICLR, 2022.
\bibitem{skalse22} J. Skalse et al. Defining and Characterizing Reward Hacking. NeurIPS, 2022.
\bibitem{geirhos20} R. Geirhos et al. Shortcut learning in deep neural networks. \emph{Nature Machine Intelligence}, 2:665--673, 2020.
\end{thebibliography}
\end{document}
""")
    open(OUT, "w", encoding="utf-8").write("\n".join(tex))
    print(f"wrote {OUT} ({len(''.join(tex).split())} words of LaTeX source)")
    print(f"main-body sections + {len(appendix)} appendix blocks; drop in official ML4H style + full .bib before submitting")


if __name__ == "__main__":
    main()
