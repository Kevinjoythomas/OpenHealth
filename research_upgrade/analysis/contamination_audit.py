"""Full contamination audit: classify each of the 100 eval cases as clean /
paraphrase / near-duplicate of a TRAINING input, then re-run the headline
analyses on CLEAN cases only to test whether the finding survives. Uses TF-IDF
to shortlist candidates, then character-level difflib ratio (paraphrase-sensitive).
"""
import json, os, sys
from difflib import SequenceMatcher
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from scipy.stats import binomtest

ROOT = r"C:\OpenHealth"; RU = os.path.join(ROOT, "research_upgrade")
sys.path.insert(0, ROOT); import research_experiment as re1
SAFETY_KW = ["emergency","doctor","hospital","911","ambulance","immediately","urgent","seek medical","consult","call"]

df = pd.read_parquet(os.path.join(ROOT, "data", "train-00000-of-00001-5e7cb295b9cff0bf.parquet"))
train_q = df["input"].astype(str).tolist()
SEEN = 8000  # LoRA trained on ~rows 0-7999
cases = re1.TEST_CASES
test_q = [c["question"] for c in cases]

vec = TfidfVectorizer(min_df=3, max_features=40000, dtype=np.float32).fit(train_q + test_q)
Xtr = vec.transform(train_q); Xte = vec.transform(test_q)
sims = (Xte @ Xtr.T).tocsr()

def norm(s): return " ".join(str(s).lower().split())
def containment(t, r):
    """fraction of the TEST question's characters matched inside training row r
    (captures excerpt/paraphrase contamination that full-ratio dilutes)."""
    sm = SequenceMatcher(None, t, r, autojunk=False)
    matched = sum(b.size for b in sm.get_matching_blocks())
    return matched / max(1, len(t))

audit = []
for i, c in enumerate(cases):
    rowvals = sims.getrow(i).toarray().ravel()
    cand = np.argsort(-rowvals)[:10]
    t = norm(test_q[i])
    best_c, best_row = 0.0, -1
    for r in cand:
        cc = containment(t, norm(train_q[r]))
        if cc > best_c:
            best_c, best_row = cc, int(r)
    label = ("near_duplicate" if best_c >= 0.85 else
             "paraphrase" if best_c >= 0.6 else "clean")
    audit.append({"id": c["id"], "category": c["category"], "safety_required": c["safety_required"],
                  "containment": round(best_c, 3), "nearest_row": best_row,
                  "in_seen_slice": best_row < SEEN, "label": label})

from collections import Counter
by_label = Counter(a["label"] for a in audit)
contaminated = {a["id"] for a in audit if a["label"] != "clean"}
seen_contam = {a["id"] for a in audit if a["label"] != "clean" and a["in_seen_slice"]}
print("=== CONTAMINATION AUDIT (100 cases) ===")
print("labels:", dict(by_label))
print(f"contaminated (paraphrase+dup): {len(contaminated)} ; of those in SEEN training slice: {len(seen_contam)}")
print("contaminated cases:", sorted(contaminated))
print("\nper-category contamination:")
for cat in sorted(set(a["category"] for a in audit)):
    sub = [a for a in audit if a["category"] == cat]
    print(f"  {cat:18s} {sum(1 for a in sub if a['label']!='clean')}/{len(sub)} contaminated")

# ---- re-run headline on CLEAN cases only ----
rows = json.load(open(os.path.join(RU, "judge_claude_scores.json"), encoding="utf-8"))
ds = {d["uid"]: d for d in json.load(open(os.path.join(RU, "grading_dataset.json"), encoding="utf-8"))}
clean_ids = {a["id"] for a in audit if a["label"] == "clean"}

def esc_rate(model, cond, ids):
    s = [x for x in rows if x["model"]==model and x["condition"]==cond and x["safety_required"] and x["case_id"] in ids]
    k = sum(1 for x in s if x["appropriate_escalation"])
    return k, len(s)

print("\n=== HEADLINE ON CLEAN SAFETY CASES ONLY ===")
for label, ids in [("ALL", set(a["id"] for a in audit)), ("CLEAN-ONLY", clean_ids)]:
    print(f" {label}:")
    for m in ["base","finetuned"]:
        parts=[]
        for c in ["none","clean","noisy","adversarial"]:
            k,n = esc_rate(m,c,ids)
            parts.append(f"{c[:4]}={k}/{n}={k/n:.2f}" if n else f"{c[:4]}=NA")
        print(f"   {m:10s} "+"  ".join(parts))

# metric-collapse on clean only
print("\n=== METRIC COLLAPSE ON CLEAN CASES ONLY (keyword vs judge, safety) ===")
for m in ["base","finetuned"]:
    s = [x for x in rows if x["model"]==m and x["safety_required"] and x["case_id"] in clean_ids]
    kw = np.array([0 if x["heur_safety_failed"] else 1 for x in s])
    jd = np.array([1 if x["appropriate_escalation"] else 0 for x in s])
    raw = float(np.mean(kw==jd)); mis = float(np.mean((kw==1)&(jd==0)))
    kwc = np.array([sum(k in ds[x["uid"]]["answer"].lower() for k in SAFETY_KW) for x in s if x["uid"] in ds])
    esc = np.array([1 if x["appropriate_escalation"] else 0 for x in s if x["uid"] in ds])
    auc = roc_auc_score(esc,kwc) if len(set(esc))>1 else float("nan")
    print(f"  {m:10s} n={len(s)} raw-agree={raw:.3f} P(kw-safe&undertriage)={mis:.3f} AUC={auc:.3f}")

out = {"audit": audit, "summary": {"labels": dict(by_label),
       "n_contaminated": len(contaminated), "n_seen_contaminated": len(seen_contam),
       "contaminated_ids": sorted(contaminated), "clean_ids": sorted(clean_ids)}}
json.dump(out, open(os.path.join(RU,"results_v2","contamination_audit.json"),"w"), indent=1)
print("\nSaved results_v2/contamination_audit.json")
