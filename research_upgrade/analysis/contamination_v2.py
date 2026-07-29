"""Rigorous contamination detection via LONGEST CONTIGUOUS matching block.
Real excerpts/paraphrases share one long contiguous run with a training row;
coincidental medical phrasing shares many SHORT scattered blocks. We classify on
the longest block (chars) and its ratio to the test length. Audits both the
original 100 eval cases and the authored clean benchmark, prints examples.
"""
import json, os, sys
from difflib import SequenceMatcher
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = r"C:\OpenHealth"; RU = os.path.join(ROOT, "research_upgrade")
sys.path.insert(0, ROOT); import research_experiment as re1
df = pd.read_parquet(os.path.join(ROOT, "data", "train-00000-of-00001-5e7cb295b9cff0bf.parquet"))
train_q = df["input"].astype(str).tolist()
SEEN = 8000


def norm(s): return " ".join(str(s).lower().split())


def audit(cases, qkey="question", idkey="id"):
    qs = [norm(c[qkey]) for c in cases]
    vec = TfidfVectorizer(min_df=3, max_features=40000, dtype=np.float32).fit(train_q + qs)
    Xtr = vec.transform(train_q); Xte = vec.transform(qs)
    sims = (Xte @ Xtr.T).tocsr()
    rows = []
    for i, c in enumerate(cases):
        cand = np.argsort(-sims.getrow(i).toarray().ravel())[:10]
        t = qs[i]
        best_block, best_row, best_ratio = 0, -1, 0.0
        for r in cand:
            sm = SequenceMatcher(None, t, norm(train_q[r]), autojunk=False)
            m = sm.find_longest_match(0, len(t), 0, len(norm(train_q[r])))
            if m.size > best_block:
                best_block, best_row = m.size, int(r)
                best_ratio = m.size / max(1, len(t))
        label = ("duplicate" if best_ratio >= 0.75 else
                 "paraphrase" if (best_block >= 80 and best_ratio >= 0.35) else "clean")
        rows.append({"id": c[idkey], "longest_block_chars": best_block,
                     "block_ratio": round(best_ratio, 3), "nearest_row": best_row,
                     "in_seen": best_row < SEEN, "label": label})
    return rows


print("=== ORIGINAL 100 CASES (longest-block metric) ===")
o = audit(re1.TEST_CASES)
from collections import Counter
print("labels:", dict(Counter(r["label"] for r in o)))
contam = [r for r in o if r["label"] != "clean"]
print(f"contaminated: {len(contam)}/100 (seen-slice: {sum(1 for r in contam if r['in_seen'])})")
print("examples (top block_ratio):")
for r in sorted(o, key=lambda x: -x["block_ratio"])[:5]:
    print(f"  {r['id']} block={r['longest_block_chars']}c ratio={r['block_ratio']} {r['label']} row{r['nearest_row']}")
json.dump(o, open(os.path.join(RU, "results_v2", "contamination_v2_original.json"), "w"), indent=1)

cb_path = os.path.join(RU, "results_v2", "clean_benchmark.json")
if os.path.exists(cb_path):
    cb = json.load(open(cb_path, encoding="utf-8"))
    print("\n=== CLEAN BENCHMARK (longest-block metric) ===")
    cbo = audit(cb)
    print("labels:", dict(Counter(r["label"] for r in cbo)))
    print("max block_ratio:", max(r["block_ratio"] for r in cbo),
          " max block chars:", max(r["longest_block_chars"] for r in cbo))
    json.dump(cbo, open(os.path.join(RU, "results_v2", "contamination_v2_clean_bench.json"), "w"), indent=1)
