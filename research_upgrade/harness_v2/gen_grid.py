"""R2/G1: Generate the corrected grid. Usage:
  python gen_grid.py --models llama3,openhealth-doctor --seed 101 --tag grid_s101
Model-major loop (minimizes GPU model swaps). num_ctx=2048, contexts from cache,
verbatim prompt context saved with every run. Resume-safe per (case,model)."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import (RESULTS, SYSTEM_PROMPT, approx_tokens, generate,
                    load_cases, log, read_json, write_json)

CONDS = ["none", "clean", "noisy", "adversarial"]


def build_prompt(question, context=""):
    if context:
        return (f"{SYSTEM_PROMPT}\n\nRelevant medical context:\n{context}\n\n"
                f"Patient: {question}\nDoctor:")
    return f"{SYSTEM_PROMPT}\n\nPatient: {question}\nDoctor:"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cases", default="default")
    ap.add_argument("--contexts", default="contexts_cache_orig.json")
    ap.add_argument("--num_ctx", type=int, default=2048)
    args = ap.parse_args()
    models = args.models.split(",")

    cases = load_cases(args.cases)
    ctx_cache = read_json(os.path.join(RESULTS, args.contexts), None)
    if ctx_cache is None:
        raise SystemExit(f"{args.contexts} missing -- run precompute_contexts first")
    out_path = os.path.join(RESULTS, f"{args.tag}.json")
    out = read_json(out_path, {})

    total = len(models) * len(cases) * len(CONDS)
    done0 = sum(1 for m in out.values() for c in m.values() if isinstance(c, dict) and c.get("answer"))
    log(f"gen_grid {args.tag}: models={models} seed={args.seed} total={total} already_ok={done0}")
    from common import restart_daemon
    RESTART_EVERY = 40          # bound the 4GB-GPU pinned-memory leak (batch small)
    n_done = 0
    for model in models:                      # model-major: avoid swaps
        out.setdefault(model, {})
        restart_daemon()                       # fresh daemon at each model switch
        for case in cases:
            cid = case["id"]
            for cond in CONDS:
                key = f"{cid}|{cond}"
                # RETRY error/empty cells on resume; only skip genuinely-complete ones
                if key in out[model] and out[model][key].get("answer"):
                    continue
                context = "" if cond == "none" else ctx_cache[cid][cond]["text"]
                prompt = build_prompt(case["question"], context)
                try:
                    ans, secs = generate(model, prompt, seed=args.seed,
                                         num_ctx=args.num_ctx)
                    out[model][key] = {
                        "case_id": cid, "condition": cond, "seed": args.seed,
                        "category": case["category"],
                        "safety_required": case["safety_required"],
                        "question": case["question"],
                        "context_text": context,
                        "prompt_tokens_approx": approx_tokens(prompt),
                        "answer": ans, "seconds": round(secs, 1),
                    }
                except Exception as e:
                    out[model][key] = {"case_id": cid, "condition": cond,
                                       "seed": args.seed, "error": str(e)[:200]}
                n_done += 1
                if n_done % 5 == 0:
                    write_json(out_path, out)
                    log(f"  {args.tag} {model} {n_done}: {key} ({out[model][key].get('seconds','ERR')}s)")
                if n_done % RESTART_EVERY == 0:
                    restart_daemon()           # periodic fresh daemon within the grid
    write_json(out_path, out)
    ok = sum(1 for m in out.values() for c in m.values() if c.get("answer"))
    err = sum(1 for m in out.values() for c in m.values() if not c.get("answer"))
    log(f"DONE {args.tag}: {ok}/{total} ok, {err} errors")


if __name__ == "__main__":
    main()
