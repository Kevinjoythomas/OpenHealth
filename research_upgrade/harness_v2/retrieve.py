"""Standalone hybrid retriever — faithful reimplementation of the v1 service:
dense (cosine, MMR lambda=0.5) + BM25Okapi + RRF (k=60). See
services/retrieval-service/app/{vector_store,bm25,hybrid}.py for the original."""
import os
import sys

import numpy as np
from rank_bm25 import BM25Okapi

sys.path.insert(0, os.path.dirname(__file__))
from common import RU, embed, read_json

DOCS = os.path.join(RU, "results_v2", "corpus_docs.json")
VECS = os.path.join(RU, "results_v2", "corpus_vecs.npy")
RRF_K = 60

_docs = _vecs = _bm25 = None


def _load():
    global _docs, _vecs, _bm25
    if _docs is None:
        _docs = read_json(DOCS, None)
        _vecs = np.load(VECS)
        _vecs = _vecs / (np.linalg.norm(_vecs, axis=1, keepdims=True) + 1e-9)
        _bm25 = BM25Okapi([d["content"].lower().split() for d in _docs])
        assert len(_docs) == len(_vecs), "docs/vecs mismatch"


def dense_mmr(query, top_k=10, fetch=20, lam=0.5):
    _load()
    q = np.array(embed(query), dtype=np.float32)
    q = q / (np.linalg.norm(q) + 1e-9)
    sims = _vecs @ q
    cand = list(np.argsort(-sims)[:fetch])
    sel = []
    while cand and len(sel) < top_k:
        if not sel:
            sel.append(cand.pop(0))
            continue
        best, best_val = None, -1e9
        for c in cand:
            red = max(float(_vecs[c] @ _vecs[s]) for s in sel)
            val = lam * float(sims[c]) - (1 - lam) * red
            if val > best_val:
                best, best_val = c, val
        sel.append(best)
        cand.remove(best)
    return [(i, float(sims[i])) for i in sel]


def bm25(query, top_k=10):
    _load()
    scores = _bm25.get_scores(query.lower().split())
    idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(i, float(scores[i])) for i in idx if scores[i] > 0]


def hybrid(query, top_k=5):
    """RRF fuse of dense-MMR and BM25 (each fetches top_k*2, per v1 service)."""
    d = dense_mmr(query, top_k=top_k * 2)
    b = bm25(query, top_k=top_k * 2)
    scores = {}
    for rank, (i, _) in enumerate(d):
        scores[i] = scores.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
    for rank, (i, _) in enumerate(b):
        scores[i] = scores.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    _load()
    return [{"doc_index": int(i), "rrf": float(s), "content": _docs[i]["content"],
             "source": _docs[i]["source"],
             "page": (int(_docs[i]["page"]) if _docs[i]["page"] is not None else None)}
            for i, s in ranked]
