"""R1: Precompute and cache retrieval contexts for all 100 cases x 3 conditions.
Clean: question query, top-5, tau=0.020 filter (v1 parity). Noisy/adversarial:
case's noisy_query/adversarial_query, top-3, no threshold (v1 parity).
Every chunk truncated to CHUNK_TOKEN_CAP tokens so the whole prompt verifiably
fits num_ctx=2048; both full and truncated text stored (contexts now LOGGED,
fixing the v1 gap)."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import RU, RESULTS, approx_tokens, load_cases, log, read_json, write_json
from retrieve import hybrid

TAU = 0.020
CHUNK_TOKEN_CAP = 400


def _truncate(text, cap=CHUNK_TOKEN_CAP):
    words = text.split()
    limit = int(cap / 1.35)
    return " ".join(words[:limit]) if len(words) > limit else text


def _fmt(chunks):
    parts = []
    for c in chunks:
        head = f"[{os.path.basename(c['source'])} p.{(c['page'] or 0) + 1}]" if c["source"] else ""
        body = _truncate(c["content"])
        parts.append(f"{head}\n{body}" if head else body)
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="default")
    ap.add_argument("--tag", default="orig")
    args = ap.parse_args()
    OUT = os.path.join(RESULTS, f"contexts_cache_{args.tag}.json")
    cases = load_cases(args.cases)
    log(f"precompute_contexts tag={args.tag}: {len(cases)} cases -> {OUT}")
    cache = read_json(OUT, {})
    for n, case in enumerate(cases):
        cid = case["id"]
        if cid in cache:
            continue
        entry = {}
        clean = hybrid(case["question"], top_k=5)
        clean = [c for c in clean if c["rrf"] >= TAU]
        entry["clean"] = {"chunks": clean, "text": _fmt(clean), "n": len(clean)}
        for cond, qkey in [("noisy", "noisy_query"), ("adversarial", "adversarial_query")]:
            ch = hybrid(case[qkey], top_k=3)
            entry[cond] = {"chunks": ch, "text": _fmt(ch), "n": len(ch), "query": case[qkey]}
        for cond in ("clean", "noisy", "adversarial"):
            entry[cond]["approx_tokens"] = approx_tokens(entry[cond]["text"])
        cache[cid] = entry
        if (n + 1) % 10 == 0:
            write_json(OUT, cache)
            log(f"contexts {n+1}/{len(cases)}")
    write_json(OUT, cache)
    toks = [e[c]["approx_tokens"] for e in cache.values() for c in ("clean", "noisy", "adversarial")]
    log(f"DONE precompute_contexts: {len(cache)} cases; context tokens "
        f"min={min(toks)} med={sorted(toks)[len(toks)//2]} max={max(toks)} (cap fits ctx 2048)")


if __name__ == "__main__":
    main()
