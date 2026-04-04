# Retrieval Evaluation

## RAG Corpus

| Property | Value |
|---|---|
| Total PDFs | 27 (in `data/` folder) |
| Initially ingested | 21 |
| Added later | 6 |
| New chunks added (full ingestion) | ~1,262 |

Topics covered include: BPPV/vertigo guidelines, chest pain protocols, pediatric care, diabetic management, and related clinical areas.

---

## Hybrid Retrieval Strategy

The retrieval service combines two complementary search methods:

| Component | Description |
|---|---|
| BM25 | Keyword-based search using TF-IDF in-memory index |
| ChromaDB vector search | Semantic search using nomic-embed-text embeddings |
| Fusion method | Reciprocal Rank Fusion (RRF) |
| RRF formula | score = Σ 1/(60 + rank_i) |
| Execution | Both retrievers run concurrently via ThreadPoolExecutor |
| Top-K | 5 results returned after fusion |
| Relevance threshold | MIN_SCORE = 0.020 |

The relevance threshold ensures that sources are only cited when both retrievers agreed on a document. Documents with an RRF score below 0.020 are filtered out and the model answers from its training knowledge instead.

---

## Latency Benchmark

**Setup:** 4 queries, 5 runs each.

| Strategy | Avg Latency |
|---|---|
| BM25 only | 2,058 ms |
| Vector only | 2,161 ms |
| Hybrid (concurrent) | 2,155 ms |

Hybrid latency is near-identical to vector-only because both retrievers run concurrently. Total time is approximately max(BM25, vector) rather than the sum of both. BM25 is faster when run alone because it uses an in-memory TF-IDF index with no embedding call required.

**Key finding:** Hybrid retrieval improves result quality over either strategy alone (RRF requires relevance consensus across both retrievers) while adding no meaningful latency cost compared to vector search alone.

---

## Corpus Gap Issue and Fix

**Problem:** When tested with queries about aplastic anemia and MRI, the retrieval service was citing cardiac PDFs in response to a blood disorder query. The corpus had no relevant coverage for these topics, but RRF scores were still marginally above the old relevance threshold, causing irrelevant documents to be returned as sources.

**Root cause:** Near-zero RRF scores (due to corpus gap) were still crossing the original threshold.

**Fix:** The relevance threshold (MIN_SCORE) was raised so that documents with weak cross-retriever agreement are filtered out entirely. When no relevant documents are found, the model responds from its training knowledge without citing sources, which is the correct behaviour for out-of-corpus queries.
