"""Deterministic paper statistics (no GPU):
  1. Contamination dedup   -- 100 eval cases vs training corpus (all 112k + seen 8k)
  2. Discriminability       -- AUC of keyword-count -> judged escalation, base vs FT
  3. Floor shift            -- keyword counts in escalating vs under-triaging answers
Saves research_upgrade/results_v2/paper_stats.json and prints a summary.
"""
import json, os, sys
import numpy as np

ROOT = r"C:\OpenHealth"
RU = os.path.join(ROOT, "research_upgrade")
OUT = os.path.join(RU, "results_v2", "paper_stats.json")
sys.path.insert(0, ROOT)
import research_experiment as re1

SAFETY_KW = ["emergency", "doctor", "hospital", "911", "ambulance",
             "immediately", "urgent", "seek medical", "consult", "call"]
res = {}

# ---------- 1. contamination ----------
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
df = pd.read_parquet(os.path.join(ROOT, "data", "train-00000-of-00001-5e7cb295b9cff0bf.parquet"))
train_q = df["input"].astype(str).tolist()
test_q = [c["question"] for c in re1.TEST_CASES]
vec = TfidfVectorizer(min_df=3, ngram_range=(1, 1), max_features=40000, dtype=np.float32).fit(train_q + test_q)
Xtr = vec.transform(train_q)              # 112k x V
Xte = vec.transform(test_q)               # 100 x V
sims = (Xte @ Xtr.T)                        # 100 x 112k sparse
max_all = np.asarray(sims.max(axis=1).todense()).ravel()
argmax = np.asarray(sims.argmax(axis=1)).ravel()
seen = sims[:, :8000]
max_seen = np.asarray(seen.max(axis=1).todense()).ravel()
flagged = [{"case": re1.TEST_CASES[i]["id"], "max_sim_all": round(float(max_all[i]), 3),
            "max_sim_seen8k": round(float(max_seen[i]), 3),
            "nearest_train_row": int(argmax[i])}
           for i in range(len(test_q)) if max_all[i] > 0.6]
res["contamination"] = {
    "n_cases": len(test_q),
    "max_cosine_any_case_vs_all": round(float(max_all.max()), 3),
    "max_cosine_any_case_vs_seen8k": round(float(max_seen.max()), 3),
    "mean_max_cosine": round(float(max_all.mean()), 3),
    "cases_over_0.6": flagged,
    "verdict": ("NO contamination (all cases < 0.6 cosine to any training input)"
                if not flagged else f"{len(flagged)} cases >0.6 -- inspect"),
}

# ---------- 2 & 3. discriminability + floor ----------
from sklearn.metrics import roc_auc_score
rows = json.load(open(os.path.join(RU, "judge_claude_scores.json"), encoding="utf-8"))
ds = {d["uid"]: d for d in json.load(open(os.path.join(RU, "grading_dataset.json"), encoding="utf-8"))}
res["discriminability"] = {}
for m in ["base", "finetuned"]:
    s = [x for x in rows if x["model"] == m and x["safety_required"] and x["uid"] in ds]
    kwc = np.array([sum(k in ds[x["uid"]]["answer"].lower() for k in SAFETY_KW) for x in s])
    esc = np.array([1 if x["appropriate_escalation"] else 0 for x in s])
    auc = float(roc_auc_score(esc, kwc)) if len(set(esc)) > 1 else float("nan")
    res["discriminability"][m] = {
        "n": len(s), "auc_keywordcount_predicts_escalation": round(auc, 3),
        "mean_kw_escalating": round(float(kwc[esc == 1].mean()), 2),
        "mean_kw_undertriage": round(float(kwc[esc == 0].mean()), 2),
        "escalation_rate": round(float(esc.mean()), 3),
    }

json.dump(res, open(OUT, "w", encoding="utf-8"), indent=2)
print(json.dumps(res, indent=2))
print("\nSaved", OUT)
