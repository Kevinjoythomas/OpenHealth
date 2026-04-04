"""
Fair benchmark: vector vs BM25 vs hybrid over N warm runs.

Testing matrix
--------------
Queries  : 4 medical questions
Methods  : 3 (vector, bm25, hybrid)
Runs     : 1 warm-up (discarded) + 5 timed runs per method per query
Isolation: each method is called via its own dedicated endpoint so caches
           are shared only within the same method across runs — not across methods.
           This means vector run 1..5 all benefit from the same warm model,
           bm25 run 1..5 all benefit from the same warm index, etc.
           Neither method gets a head-start from the other.

What this does NOT test
-----------------------
Cold-start latency (first ever call per process). That is dominated by
embedding model load time (~9s) which is a one-time startup cost, not a
steady-state concern. Cold-start is noted separately.

Endpoints used
--------------
POST /v1/retrieve/vector  — vector only
POST /v1/retrieve/bm25    — bm25 only
POST /v1/retrieve/hybrid  — hybrid only
(added below — see add_isolated_endpoints.py)
"""
import json
import statistics
import time
import urllib.request

BASE = "http://localhost:5003/v1/retrieve"
N_RUNS = 5

QUERIES = [
    "chest tightness and difficulty breathing",
    "heart attack warning signs",
    "fever in children treatment",
    "back pain radiating to leg",
]

METHODS = {
    "vector": f"{BASE}/vector",
    "bm25":   f"{BASE}/bm25",
    "hybrid": f"{BASE}/hybrid",
}


def post(url: str, query: str) -> dict:
    body = json.dumps({"query": query, "top_k": 3}).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def run_method(url: str, query: str) -> list[float]:
    # Warm-up — not counted
    post(url, query)

    samples = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        post(url, query)
        samples.append(round((time.perf_counter() - t0) * 1000, 1))
    return samples


def stats(samples: list[float]) -> dict:
    return {
        "mean":   round(statistics.mean(samples), 1),
        "median": round(statistics.median(samples), 1),
        "min":    round(min(samples), 1),
        "max":    round(max(samples), 1),
        "stdev":  round(statistics.stdev(samples), 1) if len(samples) > 1 else 0.0,
        "runs":   samples,
    }


def main():
    print("OpenHealth Retrieval Benchmark — Isolated method endpoints")
    print(f"Matrix: {len(QUERIES)} queries × {len(METHODS)} methods × {N_RUNS} runs (+ 1 warm-up each)")
    print(f"Each method warmed independently — no cross-method cache contamination")
    print("=" * 72)

    all_results = []

    for query in QUERIES:
        print(f'\nQuery: "{query}"')
        result = {"query": query}

        for method, url in METHODS.items():
            print(f"  {method:<8} warming...", end=" ", flush=True)
            try:
                samples = run_method(url, query)
                s = stats(samples)
                result[method] = s
                print(f"mean={s['mean']}ms  runs={s['runs']}")
            except Exception as e:
                print(f"FAILED: {e}")
                result[method] = None

        all_results.append(result)

    # Summary table
    print("\n\n" + "=" * 72)
    print(f"{'QUERY':<42} {'METHOD':<8} {'MEAN':>8} {'MEDIAN':>8} {'MIN':>7} {'MAX':>7} {'SD':>6}")
    print("-" * 72)
    for r in all_results:
        label = r["query"][:40]
        for method in ["vector", "bm25", "hybrid"]:
            s = r.get(method)
            if not s:
                print(f"{label:<42} {method:<8} {'ERROR':>8}")
                label = ""
                continue
            print(f"{label:<42} {method:<8} {s['mean']:>7.1f}ms {s['median']:>7.1f}ms {s['min']:>6.1f}ms {s['max']:>6.1f}ms {s['stdev']:>5.1f}")
            label = ""
        print("-" * 72)

    # Aggregate
    print("\nAGGREGATE mean-of-means (warm, isolated):")
    for method in ["vector", "bm25", "hybrid"]:
        vals = [r[method]["mean"] for r in all_results if r.get(method)]
        if vals:
            print(f"  {method:<8}: {statistics.mean(vals):.1f}ms avg across {len(vals)} queries")

    with open("benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved to benchmark_results.json")


if __name__ == "__main__":
    main()
