"""
ROUGE evaluation — generalisation vs memorisation.

Two conditions tested on both models:
  Memorisation  : 100 samples from rows 0–7,999   (seen during training)
  Generalisation: 100 samples from rows 8,000–112,164 (never seen)

Two models:
  A: llama3            (base, no medical fine-tuning)
  B: openhealth-doctor (LoRA fine-tuned)

Metrics per response (rouge_score library):
  ROUGE-1  unigram overlap
  ROUGE-2  bigram overlap
  ROUGE-L  longest common subsequence

Runtime estimate: ~200 LLM calls x ~30s each = 90-120 minutes.
Progress is printed after every sample and saved incrementally to
rouge_eval_results.json so you can Ctrl+C and resume later.
"""
import json, os, random, sys, time, urllib.request
import pandas as pd
from rouge_score import rouge_scorer

# ── Config ─────────────────────────────────────────────────────────────────────
PARQUET  = "C:/OpenHealth/data/train-00000-of-00001-5e7cb295b9cff0bf.parquet"
RESULTS  = "C:/OpenHealth/rouge_eval_results.json"
OLLAMA   = "http://localhost:11434/api/generate"
MODELS   = {
    "base":      "llama3",
    "finetuned": "openhealth-doctor",
}
N        = 50           # samples per condition
SEED     = 42
SEEN_MAX = 7999         # rows the model trained on (1000 steps x batch 8 = ~8k)

INSTRUCTION = "If you are a doctor, please answer the medical questions based on the patient's description."
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

# ── Helpers ────────────────────────────────────────────────────────────────────

def build_prompt(question: str) -> str:
    return f"{INSTRUCTION}\n\nPatient: {question}\nDoctor:"


def generate(model: str, prompt: str) -> str:
    body = json.dumps({
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": 0.1, "num_predict": 300},
    }).encode()
    req = urllib.request.Request(
        OLLAMA, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["response"].strip()


def rouge(prediction: str, reference: str) -> dict:
    scores = scorer.score(reference, prediction)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


def avg(records: list, key: str) -> float:
    vals = [r[key] for r in records if r.get(key) is not None]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


# ── Load dataset ───────────────────────────────────────────────────────────────

print("Loading dataset...")
df = pd.read_parquet(PARQUET)
print(f"  {len(df):,} total rows\n")

random.seed(SEED)
seen_idx    = random.sample(range(0, SEEN_MAX + 1), N)
unseen_idx  = random.sample(range(SEEN_MAX + 1, len(df)), N)

samples = {
    "memorisation":    [{"row": i, "question": df.iloc[i]["input"], "reference": df.iloc[i]["output"]} for i in seen_idx],
    "generalisation":  [{"row": i, "question": df.iloc[i]["input"], "reference": df.iloc[i]["output"]} for i in unseen_idx],
}

# ── Load or init results file ──────────────────────────────────────────────────

if os.path.exists(RESULTS):
    with open(RESULTS) as f:
        results = json.load(f)
    print(f"Resuming from existing results file ({RESULTS})")
else:
    results = {
        "memorisation":   {"base": [], "finetuned": []},
        "generalisation": {"base": [], "finetuned": []},
    }

def already_done(condition, model_key, row_idx):
    return any(r["row"] == row_idx for r in results[condition][model_key])

def save():
    with open(RESULTS, "w") as f:
        json.dump(results, f, indent=2)

# ── Main loop ──────────────────────────────────────────────────────────────────

total_calls = sum(
    1 for cond in ["memorisation", "generalisation"]
    for mk in ["base", "finetuned"]
    for s in samples[cond]
    if not already_done(cond, mk, s["row"])
)
print(f"LLM calls remaining: {total_calls}  (est. {total_calls * 30 // 60} min)\n")

done = 0
for cond in ["memorisation", "generalisation"]:
    print(f"\n{'='*60}")
    print(f"CONDITION: {cond.upper()}")
    print(f"{'='*60}")
    for mk, model_name in MODELS.items():
        print(f"\n  Model: {model_name}")
        for i, s in enumerate(samples[cond]):
            if already_done(cond, mk, s["row"]):
                continue
            prompt = build_prompt(s["question"][:1000])  # cap very long inputs
            try:
                t0  = time.perf_counter()
                ans = generate(model_name, prompt)
                elapsed = round(time.perf_counter() - t0, 1)
                r   = rouge(ans, s["reference"])
                record = {"row": s["row"], **r, "words": len(ans.split()), "seconds": elapsed}
                results[cond][mk].append(record)
                done += 1
                print(f"    [{cond[:3].upper()} {mk[:4]:4s}] {i+1:>3}/{N} "
                      f"row={s['row']:>6}  "
                      f"R1={r['rouge1']:.3f}  R2={r['rouge2']:.3f}  RL={r['rougeL']:.3f}  "
                      f"({elapsed}s)")
                if done % 10 == 0:
                    save()
            except Exception as e:
                print(f"    FAILED row={s['row']}: {e}")

save()

# ── Summary ────────────────────────────────────────────────────────────────────

print("\n\n" + "="*65)
print(f"{'Condition':<18} {'Model':<12} {'ROUGE-1':>8} {'ROUGE-2':>8} {'ROUGE-L':>8} {'n':>5}")
print("-"*65)

for cond in ["memorisation", "generalisation"]:
    for mk, model_name in MODELS.items():
        recs = results[cond][mk]
        if recs:
            r1 = avg(recs, "rouge1")
            r2 = avg(recs, "rouge2")
            rl = avg(recs, "rougeL")
            print(f"{cond:<18} {model_name:<12} {r1:>8.4f} {r2:>8.4f} {rl:>8.4f} {len(recs):>5}")

print("\nKey:")
print("  Memorisation  = rows 0–7,999   (seen during training)")
print("  Generalisation = rows 8,000+  (never seen by model)")
print("  Fine-tune gain = finetuned - base (positive = fine-tuning helped)")
print(f"\nFull results saved to {RESULTS}")
