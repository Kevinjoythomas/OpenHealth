"""R0: Extract the 1,823 corpus chunks from website/chroma sqlite and embed with
nomic-embed-text via the BATCH endpoint (reliable on this box). Saves docs.json +
vecs.npy. Resume-safe (batches of 64, saved after each batch)."""
import os
import sqlite3
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import RU, ROOT, batch_embed, log, write_json

DOCS = os.path.join(RU, "results_v2", "corpus_docs.json")
VECS = os.path.join(RU, "results_v2", "corpus_vecs.npy")
BATCH = 64


def main():
    con = sqlite3.connect(os.path.join(ROOT, "website", "chroma", "chroma.sqlite3"))
    rows = con.execute(
        "select e.id, "
        " max(case when m.key='chroma:document' then m.string_value end) as doc, "
        " max(case when m.key='source' then m.string_value end) as source, "
        " max(case when m.key='page' then m.int_value end) as page "
        "from embeddings e join embedding_metadata m on m.id=e.id "
        "group by e.id order by e.id").fetchall()
    docs = [{"i": i, "content": r[1], "source": r[2] or "", "page": r[3]}
            for i, r in enumerate(rows) if r[1]]
    write_json(DOCS, docs)
    log(f"extracted {len(docs)} chunks")

    vecs = list(np.load(VECS)) if os.path.exists(VECS) else []
    done = len(vecs)
    log(f"resuming embeddings at {done}")
    i = done
    while i < len(docs):
        chunk = [d["content"] for d in docs[i:i + BATCH]]
        embs = batch_embed(chunk)
        vecs.extend(np.array(e, dtype=np.float32) for e in embs)
        i += len(chunk)
        np.save(VECS, np.array(vecs))
        log(f"embedded {len(vecs)}/{len(docs)}")
    np.save(VECS, np.array(vecs))
    log(f"DONE corpus_build: {len(vecs)} vectors")


if __name__ == "__main__":
    main()
