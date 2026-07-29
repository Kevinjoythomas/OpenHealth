"""B2: External validation on HealthBench (OpenAI, physician-authored).
Discovers the dataset URL from the simple-evals source, downloads the oss eval
JSONL, filters emergency-referral-themed cases, and runs base + fine-tuned on
them (no retrieval). Outputs stored with the physician rubric criteria attached
so the judge panel can grade them later. Fail-soft: any acquisition failure
logs and exits 0 with a SKIPPED marker in the results file."""
import json
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS, generate, log, read_json, write_json

MODELS = {"base": "llama3", "finetuned": "openhealth-doctor"}
OUT = os.path.join(RESULTS, "healthbench_runs.json")
RAW = os.path.join(RESULTS, "healthbench_raw.jsonl")
MAX_CASES = 80
EMERG = re.compile(r"emergen|urgent care|call 911|refer.*(er|emergency)|escalat", re.I)


def _get(url, timeout=180):
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def acquire():
    if os.path.exists(RAW) and os.path.getsize(RAW) > 1e6:
        log("HealthBench raw already downloaded")
        return True
    try:
        src = _get("https://raw.githubusercontent.com/openai/simple-evals/main/healthbench_eval.py").decode()
        urls = re.findall(r"https://openaipublic\.blob\.core\.windows\.net/simple-evals/healthbench/[^\"']+\.jsonl", src)
        base = [u for u in urls if "consensus" not in u and "hard" not in u and "meta" not in u]
        url = (base or urls)[0]
        log(f"HealthBench URL: {url}")
        data = _get(url, timeout=600)
        with open(RAW, "wb") as f:
            f.write(data)
        log(f"downloaded {len(data)/1e6:.1f} MB")
        return True
    except Exception as e:
        log(f"HealthBench acquisition FAILED: {e}")
        return False


def emergency_cases():
    cases = []
    with open(RAW, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            tags = " ".join(map(str, row.get("example_tags", [])))
            rubrics = row.get("rubrics", [])
            rubric_txt = " ".join(str(r.get("criterion", "")) for r in rubrics)
            if not (EMERG.search(tags) or EMERG.search(rubric_txt)):
                continue
            msgs = row.get("prompt", [])
            users = [m["content"] for m in msgs if m.get("role") == "user"]
            if not users or len(msgs) > 3:      # keep near-single-turn cases
                continue
            cases.append({"hb_id": row.get("prompt_id") or row.get("example_id") or len(cases),
                          "question": users[-1][:2000],
                          "rubrics": [{"criterion": r.get("criterion"), "points": r.get("points")}
                                      for r in rubrics][:20],
                          "tags": tags[:300]})
            if len(cases) >= MAX_CASES:
                break
    return cases


def main():
    out = read_json(OUT, {})
    if not acquire():
        out["_status"] = "SKIPPED_ACQUISITION_FAILED"
        write_json(OUT, out)
        return
    cases = emergency_cases()
    log(f"B2: {len(cases)} emergency-themed HealthBench cases")
    out.setdefault("cases", {})
    n = 0
    for c in cases:
        cid = str(c["hb_id"])
        entry = out["cases"].get(cid, {"question": c["question"], "rubrics": c["rubrics"],
                                       "tags": c["tags"], "answers": {}})
        for mk, model in MODELS.items():
            if mk in entry["answers"]:
                continue
            from common import SYSTEM_PROMPT
            prompt = f"{SYSTEM_PROMPT}\n\nPatient: {c['question']}\nDoctor:"
            try:
                ans, secs = generate(model, prompt, seed=101, num_ctx=2048)
                entry["answers"][mk] = {"answer": ans, "seconds": round(secs, 1)}
            except Exception as e:
                entry["answers"][mk] = {"error": str(e)[:150]}
        out["cases"][cid] = entry
        n += 1
        if n % 5 == 0:
            write_json(OUT, out)
            log(f"B2: {n}/{len(cases)}")
    out["_status"] = f"OK_{len(out['cases'])}_cases"
    write_json(OUT, out)
    log(f"B2 DONE: {out['_status']}")


if __name__ == "__main__":
    main()
