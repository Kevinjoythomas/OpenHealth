# OpenHealth

A medical question-answering platform that serves a QLoRA-fine-tuned Llama 3 8B model through hybrid retrieval over clinical reference PDFs, built as Flask microservices with a web chat frontend.

![OpenHealth chat interface](website/static/repo_images/chatbot.png)

## Overview

General-purpose chat models give generic medical answers and cannot point to a source. OpenHealth fine-tunes Llama 3 8B Instruct on patient and doctor conversations so it answers in a short consultation style, then grounds each answer in reference PDFs that you supply and cites the file and page it used. Retrieval combines keyword search (BM25) with dense vector search, fuses the two rankings with reciprocal rank fusion (RRF), and drops weak matches, so the model falls back to its own training when no document fits. The backend is split into five services (gateway, auth, chat orchestration, retrieval, ingestion) and runs locally with Docker Compose and Ollama. It started as a college project at MVSR Engineering College, Hyderabad; the original single-file Flask app is kept in `website/`.

## Highlights

- **Fine-tuning:** QLoRA with Unsloth on `unsloth/llama-3-8b-Instruct-bnb-4bit` (4-bit base, LoRA rank 8, alpha 16, all attention and MLP projections). 20,971,520 trainable parameters, 1,000 steps at an effective batch size of 8 on one Tesla T4, over the 112,165 examples in [`kevinjoythomas/Doctor-Dataset`](https://huggingface.co/datasets/kevinjoythomas/Doctor-Dataset), a prompt-formatted copy of the ChatDoctor HealthCareMagic conversations (logged output in `models/train.ipynb`). The model is exported to [GGUF](https://huggingface.co/kevinjoythomas/medical-loratuned-chatbot-GGUF) and served by Ollama.
- **Hybrid retrieval:** BM25 (`rank-bm25`) and Chroma MMR search with `nomic-embed-text` embeddings run in parallel threads and are fused with RRF (k = 60). The orchestrator keeps fused scores of at least 0.015, so a chunk ranked first by either retriever (1/61 = 0.0164) passes while weaker matches are dropped (`services/retrieval-service/app/hybrid.py`, `services/chat-orchestrator/app/orchestrator.py`). The floor used to be 0.020, which only passed chunks found by both retrievers; in a 30-question test that left every answer without a citation, and 0.015 restored citations on 10 of 10.
- **Cited answers:** each answer ends with the source file and page of the chunks used, or a note that no document passed the threshold.
- **Service plumbing:** JWT access and refresh tokens with bcrypt password hashing and refresh-token revocation in Redis, a Redis sliding-window rate limiter at the gateway, and a Celery ingestion worker on RabbitMQ with a dead-letter queue and SHA-256 deduplication of uploaded PDFs.
- **Tests:** 27 pytest tests across the five services (`services/*/tests/`).

## Architecture

```mermaid
flowchart LR
    UI["Web frontend<br/>website/ or deploy/"] --> GW["api-gateway :5000<br/>JWT check, rate limit"]
    GW --> AUTH["auth-service :5001"]
    GW --> CHAT["chat-orchestrator :5002"]
    CHAT --> RET["retrieval-service :5003<br/>BM25 + vector, RRF"]
    RET --> VDB[("Chroma index")]
    CHAT --> OLL["Ollama on host<br/>Llama 3 8B GGUF, nomic-embed-text"]
    RET --> OLL
    CHAT -- "ingest task" --> MQ[("RabbitMQ")]
    MQ --> ING["ingestion-worker<br/>chunk, embed, upsert"]
    S3[("S3 bucket")] --> ING
    ING --> VDB
    AUTH --> PG[("Postgres")]
    CHAT --> PG
    GW & AUTH & CHAT --> RD[("Redis")]
```

For each message, the gateway validates the JWT and forwards the request to the chat orchestrator. The orchestrator loads recent history from Postgres through a Redis cache, asks the retrieval service for the top 3 chunks, builds the prompt, calls the model through Ollama, and stores both turns.

## Safety gate and evaluation

The chat path above answers every question. For emergencies that is the wrong behavior, so a companion research harness adds a **triage gate** that runs before the model: 40 red-flag patterns plus a zero-shot Gemma 3 4B check. On held-out, physician-labeled HealthBench conversations the gate catches 92% of emergent cases at 92% specificity (AUROC 0.95), and the gated pipeline runs at a 2.9 s median per message on a 4 GB laptop GPU. The same study found that the keyword safety rubric commonly used to score medical chatbots stops working after fine-tuning: its agreement with blinded judges fell from Cohen's kappa 0.61 to 0.03, replicated on 273 physician-labeled conversations. The gate code and the full write-up will be released with the paper, which is under review; the harness is not part of this repository yet.

Measured on this repository (October 2026, same Q4_K_M model):

| What | Result |
|---|---|
| Hybrid retrieval, warm | about 0.1 s per query (81 ms p50 server-side); 3.5 s cold while the BM25 index builds |
| End-to-end chat, RTX 3050 Laptop 4 GB (model split across CPU and GPU) | p50 14.8 s, p95 23.5 s at 1 user; p95 58 s at 4 users; 0 errors |
| Model alone, Tesla T4 | 4.2 s per answer, 32 tokens/s, fully on GPU |
| Fine-tune vs base, ROUGE against the doctors' answers on 100 held-out questions | ROUGE-1 0.258 to 0.327, ROUGE-2 0.032 to 0.061; answers 2.3x shorter |

ROUGE measures overlap with the corpus answers, not clinical quality; that is what the gate and the paper are for.

## Tech stack

Python, Flask, Celery, LangChain, Chroma, rank-bm25, Ollama, Unsloth, PostgreSQL, Redis, RabbitMQ, Docker Compose; static frontend deployable to Vercel.

## Quickstart

You need Docker with Compose v2, [Ollama](https://ollama.com) running on the host, and Python 3.10+ to build the index.

```bash
ollama pull hf.co/kevinjoythomas/medical-loratuned-chatbot-GGUF
ollama pull nomic-embed-text
```

### 1. Add your own reference documents

The clinical guideline PDFs and the vector index used during development are **not included** in this repository, because the documents are copyrighted. Retrieval needs a corpus, so supply PDFs you are allowed to use (for example open-access guidelines or reviews) and build the index with `website/populate_db.py`:

```bash
cd website
mkdir -p data                      # copy your PDFs into website/data/
pip install -r ../services/retrieval-service/requirements.txt pypdf
python populate_db.py --reset      # reads ./data, writes ./chroma
cd ..
```

The script loads every PDF in `data/`, splits pages into 800-character chunks with 80 characters of overlap, embeds them with `nomic-embed-text` through Ollama, and stores them in Chroma with IDs of the form `file:page:chunk`. Without `--reset` it only adds chunks it has not seen before. Both `data/` and `chroma/` are git-ignored, so your documents stay local.

### 2. Start the stack

`docker-compose.yml` needs `DATABASE_URL` and `JWT_SECRET`. There is no root `.env.example`, so create `.env` directly:

```bash
printf 'DATABASE_URL=postgresql://openhealth:openhealth@postgres:5432/openhealth\nJWT_SECRET=%s\n' "$(openssl rand -hex 32)" > .env
make build && make up
docker compose cp website/chroma/. retrieval-service:/app/chroma   # load your index into the shared volume
docker compose restart retrieval-service                          # BM25 is built on first query and cached
curl http://localhost:5000/health
```

To use the chat page, run `python website/serve.py` and open `http://localhost:3000/login`. The pages call the API at `http://localhost:5000` by default; to point them elsewhere, set `window.OPENHEALTH_API` before the page script runs.

Documents can also be added while the stack runs: upload a PDF to the S3 bucket named by `S3_BUCKET` (AWS credentials go in `.env`), then call `POST /v1/ingest/document` with `{"s3_key": "...", "filename": "..."}`. The worker applies the same chunking into the shared Chroma volume; restart `retrieval-service` afterwards so BM25 picks up the new chunks.

`make test` runs the service test suites inside the containers.

### API (base URL `http://localhost:5000/v1`)

| Area | Endpoints |
|---|---|
| Auth | `POST /auth/signup`, `POST /auth/login`, `POST /auth/refresh`, `GET /auth/me` |
| Chat | `POST /chat/sessions`, `GET /chat/sessions`, `POST /chat/sessions/{id}/messages`, `GET /chat/sessions/{id}/messages`, `DELETE /chat/sessions/{id}` |
| Ingestion | `POST /ingest/document` |

## Project structure

```text
services/
  api-gateway/          JWT validation, rate limiting, reverse proxy
  auth-service/         signup, login, token refresh and revocation
  chat-orchestrator/    chat sessions, prompt building, RAG flow
  retrieval-service/    BM25 + vector search, RRF fusion
  ingestion-worker/     Celery worker: S3 -> chunks -> Chroma
website/                original Flask app, local frontend (serve.py), populate_db.py
deploy/                 static frontend for Vercel (routes in vercel.json)
models/                 train.py, train.ipynb, Ollama Modelfile, early diagnostic prototypes
eval_model.py           rubric evaluation: base vs fine-tuned vs fine-tuned + RAG
rouge_eval.py           ROUGE evaluation against held-out doctor answers
benchmark_retrieval.py  latency benchmark for BM25, vector and hybrid search
docker-compose.yml      five services plus Postgres, Redis and RabbitMQ
```

The diagnostic prototypes in `models/` (brain MRI tumor segmentation, a lung cancer survey classifier, a breast cancer stub) come from the first version of the project. Their trained weights are not in the repository.

## Limitations and next steps

- No reference corpus or index ships with the repository. Until you add PDFs, answers come from the fine-tuned model alone, with a note saying so.
- The adapter was trained for 1,000 steps at batch size 8, about 8,000 of the 112,165 examples (7%). A full epoch is roughly 14,000 steps.
- The demo scripts (`run_local.ps1`, `start_demo.ps1`) are PowerShell; set `OPENHEALTH_PUBLIC_URL` to your own tunnel before `start_demo.ps1`. Point the frontend pages at your API address as described in the Quickstart.
- The BM25 index is rebuilt only when the retrieval service restarts.

OpenHealth is a research project, not a medical device. Do not use its output for diagnosis or treatment.

## License

No license specified yet.
