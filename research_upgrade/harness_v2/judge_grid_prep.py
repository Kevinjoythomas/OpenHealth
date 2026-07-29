"""Blind a v2 grid output file into judging batches + a de-blinding map.
Reusable for grid_clean / grid_orig / pair2 / mitigation. Usage:
  python judge_grid_prep.py <grid.json> <tag>
Writes results_v2/judge_batches_<tag>/batch_XX.json and results_v2/blindmap_<tag>.json
Prints the batch count (for the judging workflow's N).
"""
import hashlib
import json
import os
import sys

RESULTS = r"C:\OpenHealth\research_upgrade\results_v2"
B = 14


def main(grid_path, tag):
    grid = json.load(open(grid_path, encoding="utf-8"))
    cells = []
    for model, entries in grid.items():
        if not isinstance(entries, dict):
            continue
        for key, c in entries.items():
            if not isinstance(c, dict) or "answer" not in c:
                continue
            cells.append({
                "uid": f"{tag}|{model}|{c['case_id']}|{c['condition']}",
                "model": model, "case_id": c["case_id"], "condition": c["condition"],
                "safety_required": c.get("safety_required"), "category": c.get("category"),
                "question": c.get("question", ""), "answer": c["answer"],
            })
    # deterministic shuffle to decorrelate model/condition adjacency
    cells.sort(key=lambda r: hashlib.md5(r["uid"].encode()).hexdigest())
    bdir = os.path.join(RESULTS, f"judge_batches_{tag}")
    os.makedirs(bdir, exist_ok=True)
    blindmap = {}
    nb = 0
    for bi in range(0, len(cells), B):
        batch = []
        for j, r in enumerate(cells[bi:bi + B]):
            bid = f"B{bi//B:03d}{j:02d}"
            blindmap[bid] = {k: r[k] for k in ("uid", "model", "case_id", "condition",
                                               "safety_required", "category")}
            batch.append({"blind_id": bid, "safety_required": r["safety_required"],
                          "category": r["category"], "question": r["question"], "answer": r["answer"]})
        json.dump(batch, open(os.path.join(bdir, f"batch_{bi//B:02d}.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        nb += 1
    json.dump(blindmap, open(os.path.join(RESULTS, f"blindmap_{tag}.json"), "w"), indent=0)
    print(f"tag={tag}: {len(cells)} cells -> {nb} batches in {bdir}")
    print(f"N_BATCHES={nb}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
