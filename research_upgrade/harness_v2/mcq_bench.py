"""B1: MedQA + PubMedQA capability check, base vs fine-tuned. Constrained-answer
prompt; reports accuracy AND parse-failure rate (a dialogue-tuned model that can no
longer follow MCQ format is itself an informative capability result). Fail-soft:
if a dataset can't be acquired, it is skipped and logged."""
import json
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS, generate, log, read_json, write_json

MODELS = {"base": "llama3", "finetuned": "openhealth-doctor"}
N_PER = 400
OUT = os.path.join(RESULTS, "mcq_results.json")


def _get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def acquire_pubmedqa():
    try:
        raw = _get("https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json")
        d = json.loads(raw)
        items = []
        for k, v in d.items():
            dec = v.get("final_decision", "").lower()
            if dec in ("yes", "no", "maybe"):
                ctx = " ".join(v.get("CONTEXTS", []))[:1200]
                items.append({"q": v["QUESTION"], "context": ctx,
                              "options": ["yes", "no", "maybe"], "answer": dec})
        log(f"PubMedQA acquired: {len(items)}")
        return items[:N_PER]
    except Exception as e:
        log(f"PubMedQA acquire FAILED: {e}"); return []


def acquire_medqa():
    # try HF datasets, then HF hub parquet, else skip
    try:
        from datasets import load_dataset
        ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
        items = []
        for r in ds:
            opts = r["options"]
            letters = list(opts.keys()) if isinstance(opts, dict) else ["A", "B", "C", "D"]
            vals = list(opts.values()) if isinstance(opts, dict) else opts
            ans = r.get("answer_idx") or r.get("answer")
            items.append({"q": r["question"], "context": "",
                          "options": vals, "letters": letters, "answer": str(ans)})
        log(f"MedQA acquired via datasets: {len(items)}")
        return items[:N_PER]
    except Exception as e:
        log(f"MedQA acquire FAILED (datasets): {e}"); return []


def ask(model, item):
    opts = item["options"]
    letters = item.get("letters") or [chr(65 + i) for i in range(len(opts))]
    body = "\n".join(f"{l}. {o}" for l, o in zip(letters, opts))
    ctx = f"Context: {item['context']}\n\n" if item.get("context") else ""
    prompt = (f"{ctx}Question: {item['q']}\n{body}\n\n"
              f"Answer with ONLY the letter ({'/'.join(letters)}). Answer:")
    txt, _ = generate(model, prompt, seed=101, temperature=0, num_predict=8, num_ctx=2048)
    m = re.search(r"\b([A-E]|yes|no|maybe)\b", txt.strip(), re.I)
    return (m.group(1).lower() if m else None), txt.strip()[:40]


def eval_ds(name, items, results):
    for mk, model in MODELS.items():
        key = f"{name}|{mk}"
        if key in results:
            continue
        correct = parsed = 0
        rows = []
        for it in items:
            pred, raw = ask(model, it)
            letters = it.get("letters") or [chr(65 + i) for i in range(len(it["options"]))]
            gold = str(it["answer"]).lower()
            gold_letter = gold
            if gold in [str(x).lower() for x in it["options"]]:  # gold is a value
                gi = [str(x).lower() for x in it["options"]].index(gold)
                gold_letter = letters[gi].lower()
            ok = pred is not None and pred == gold_letter
            parsed += pred is not None
            correct += ok
            rows.append({"pred": pred, "gold": gold_letter, "ok": ok, "raw": raw})
        n = len(items)
        results[key] = {"model": model, "n": n, "accuracy": round(correct / n, 3),
                        "parse_fail_rate": round(1 - parsed / n, 3)}
        write_json(OUT, results)
        log(f"B1 {key}: acc={results[key]['accuracy']} parse_fail={results[key]['parse_fail_rate']} (n={n})")


def main():
    results = read_json(OUT, {})
    pub = acquire_pubmedqa()
    if pub:
        eval_ds("pubmedqa", pub, results)
    med = acquire_medqa()
    if med:
        eval_ds("medqa", med, results)
    if not pub and not med:
        log("B1: no datasets acquired -- skipped (kill-criterion: drop capability claim)")
    log("B1 DONE")


if __name__ == "__main__":
    main()
