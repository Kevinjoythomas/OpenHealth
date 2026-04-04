# OpenHealth

An end-to-end medical AI platform: LoRA fine-tuned Llama 3 8B served through a hybrid RAG pipeline, wrapped in a microservices architecture with a web frontend.

> Live demo: [openhealth-two.vercel.app](https://openhealth-two.vercel.app) · Backend via ngrok
>
> The previous monolith lives in `website/` for reference. The active backend codebase is in `services/`.

---

## Results

### Model Quality (8 Clinical Test Cases)

| Condition | Avg Score | Description |
|---|---|---|
| Base Llama 3 8B | 6.12 / 8 | No fine-tuning, no RAG |
| Fine-tuned (no RAG) | 6.75 / 8 | LoRA adapter only |
| Fine-tuned + RAG | **7.25 / 8** | Full pipeline |

Rubric: factual accuracy (2), tone (2), conciseness (2), safety referral (2).

The full pipeline improved every failed safety escalation case. The base model consistently failed to refer emergency cases (chest pain + cardiac history, suspected TIAs, polypharmacy exhaustion).

### ROUGE Evaluation (N=50 per condition)

| Split | Base R1 | Fine-tuned R1 | Δ |
|---|---|---|---|
| Memorisation (seen) | 0.2517 | 0.3122 | +24.0% |
| Generalisation (unseen) | 0.2582 | 0.3271 | **+26.7%** |

ROUGE-2 improvement on unseen data: **+94.3%**. The generalisation gain matching the memorisation gain shows the model learned medical communication patterns, not just memorised answers.

### Retrieval Latency (4 queries, 5 runs each)

| Strategy | Avg Latency |
|---|---|
| BM25 only | 2,058 ms |
| Vector only | 2,161 ms |
| Hybrid (concurrent) | 2,155 ms |

Hybrid runs both strategies in parallel — total latency ≈ max(BM25, vector), not their sum. Quality improves because RRF requires consensus across both retrievers.

---

## Models and Data

- Chat model: [kevinjoythomas/medical-loratuned-chatbot-GGUF](https://huggingface.co/kevinjoythomas/medical-loratuned-chatbot-GGUF)
- Training dataset: [lavita/ChatDoctor-HealthCareMagic-100k](https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k)
- Embeddings: nomic-embed-text (via Ollama)

**Training details:** LoRA rank=8, alpha=16, 20.97M trainable params (0.26% of 8.03B). Trained for 1,000 steps (7.1% of dataset), loss 3.74→2.19. Model was undertrained at cutoff — full convergence requires ~14,000 steps. Full training on a T4 GPU takes ~2–3 hours.

---

## Key Features

### AI Chatbot

The chatbot is the core OpenHealth interface for doctors and patients. It is LoRA fine-tuned with Unsloth and integrated into a retrieval pipeline for medical question answering.

- Retrieval-augmented generation over curated medical PDFs
- Context-aware conversations with stored chat history
- Hybrid BM25 + vector search fused via Reciprocal Rank Fusion (RRF)
- Microservice-based chat orchestration with Redis, Postgres, RabbitMQ, and ChromaDB
- Relevance threshold filtering — only cites sources when both retrievers agree (RRF score ≥ 0.020)

### Diagnostic Models

OpenHealth also includes supporting models for:

- Brain tumor detection
- Lung disease detection
- Breast cancer detection
- Diabetes risk prediction

### Collaboration

The platform also includes doctor-to-doctor collaboration and notification flows from the earlier web application.

---

## Architecture

```text
api-gateway (:5000) -> auth-service (:5001)
                    -> chat-orchestrator (:5002)
                         -> retrieval-service (:5003)
                              -> ChromaDB (vector)
                              -> BM25 index (keyword)
                         -> Ollama (host)
ingestion-worker <- RabbitMQ
```

| Service | Port | Stack | Responsibility |
|---|---|---|---|
| api-gateway | 5000 | Flask | JWT validation, rate limiting, reverse proxy |
| auth-service | 5001 | Flask + SQLite/Postgres | Signup/login, JWT issue/refresh |
| chat-orchestrator | 5002 | Flask + SQLite/Postgres | Chat sessions, RAG orchestration |
| retrieval-service | 5003 | Flask + ChromaDB + BM25 | Hybrid search, RRF fusion |
| ingestion-worker | — | Celery + RabbitMQ | Chunking, embedding, ingestion jobs |
| Ollama | host | GGUF | Llama 3 8B fine-tuned model inference |

See [docs/architecture.md](docs/architecture.md) for full chat flow and deployment details.

---

## Prerequisites

- Docker 24+ and Docker Compose v2 (for Docker setup)
- Python 3.11+ (for local no-Docker setup)
- Ollama running on the host with:

```bash
ollama pull hf.co/kevinjoythomas/medical-loratuned-chatbot-GGUF
ollama pull nomic-embed-text
```

## Quick Start (Docker)

```bash
# 1. Copy and configure environment variables
cp .env.example .env

# 2. Build and start services
make build
make up

# 3. Run database migrations
make migrate

# 4. Check service health
curl http://localhost:5000/health
```

## Quick Start (Local, No Docker)

```bash
# Start each service in a separate terminal (or use run_local.ps1)
cd services/auth-service        && pip install -r requirements.txt && python -m flask run --port 5001
cd services/chat-orchestrator   && pip install -r requirements.txt && python -m flask run --port 5002
cd services/retrieval-service   && pip install -r requirements.txt && python -m flask run --port 5003
cd services/api-gateway         && pip install -r requirements.txt && python -m flask run --port 5000
cd website                      && python serve.py

# Ingest PDFs into ChromaDB
cd services/retrieval-service && python app/populate_db.py

# Pre-warm Ollama (prevents cold-start timeout on first request)
ollama run openhealth-doctor "hi"
```

Local dev uses SQLite instead of Postgres and fakeredis instead of Redis automatically.

See [docs/running-locally.md](docs/running-locally.md) for the full guide.

---

## API

Base URL: `http://localhost:5000/v1`

OpenAPI spec: [docs/api-spec.yaml](docs/api-spec.yaml)

### Auth

- `POST /v1/auth/signup`
- `POST /v1/auth/login`
- `POST /v1/auth/refresh`
- `GET /v1/auth/me`

### Chat

- `POST /v1/chat/sessions`
- `GET /v1/chat/sessions`
- `POST /v1/chat/sessions/{id}/messages`
- `GET /v1/chat/sessions/{id}/messages`
- `DELETE /v1/chat/sessions/{id}`

### Ingestion

- `POST /v1/ingest/document`

---

## Useful Commands

```bash
make logs           # tail all service logs
make logs-chat      # tail chat-orchestrator only
make test           # run test suite
make migrate        # run DB migrations
make shell-auth     # open shell in auth-service container
make clean          # stop and remove containers + volumes
```

---

## Project Structure

```
services/
  api-gateway/          # JWT auth, rate limiting, reverse proxy
  auth-service/         # User accounts, JWT issue/refresh
  chat-orchestrator/    # Session management, RAG orchestration
  retrieval-service/    # Hybrid BM25+vector search, RRF fusion
website/
  templates/            # Jinja2 HTML (main_chatbot.html, index.html, login.html)
  serve.py              # Dev Flask server for frontend
deploy/                 # Static build for Vercel deployment
data/                   # Medical PDFs for RAG corpus
docs/                   # Architecture, evaluation writeups, API spec
eval_model.py           # 8-case rubric evaluation
rouge_eval.py           # ROUGE N=50 evaluation
benchmark_retrieval.py  # Retrieval latency benchmark
model_eval_report.html  # Full model evaluation report
retrieval_report.html   # Retrieval strategy comparison report
```

---

## Evaluation Reports

- [Model Evaluation Report](model_eval_report.html) — ROUGE scores, rubric scoring, training analysis
- [Retrieval Report](retrieval_report.html) — BM25 vs vector vs hybrid comparison
- [docs/model-evaluation.md](docs/model-evaluation.md) — Summary of all results
- [docs/retrieval-evaluation.md](docs/retrieval-evaluation.md) — Retrieval strategy details

---

## Deployment

Frontend is deployed statically on Vercel from the `deploy/` folder.
Backend runs locally and is exposed via ngrok (`https://nonsignificantly-untippled-mikaela.ngrok-free.dev`).
All frontend API calls include `ngrok-skip-browser-warning: true` header.

See [docs/architecture.md](docs/architecture.md) for full deployment details.
