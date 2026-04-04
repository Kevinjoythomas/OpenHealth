# OpenHealth

An end-to-end medical AI platform: LoRA fine-tuned Llama 3 8B served through a hybrid RAG pipeline, wrapped in a microservices architecture with a web frontend.

> Live demo: [openhealth-two.vercel.app](https://openhealth-two.vercel.app) · Backend via ngrok

## What It Does

OpenHealth is a medical chatbot that answers patient health questions by combining:
1. **A fine-tuned language model** — Llama 3 8B LoRA-adapted on 112K real clinical consultations
2. **Hybrid RAG retrieval** — BM25 + vector search fused with Reciprocal Rank Fusion over curated medical PDFs
3. **Microservices backend** — auth, chat orchestration, retrieval, and API gateway as separate services

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

## Architecture

```
Client (browser)
     │
     ▼
API Gateway :5000  ──── JWT validation, rate limiting
     │
     ├── Auth Service :5001        (signup / login / refresh)
     │
     └── Chat Orchestrator :5002   (sessions, RAG prompt builder)
              │
              └── Retrieval Service :5003   (BM25 + ChromaDB + RRF)
              │
              └── Ollama (host)             (Llama 3 8B GGUF inference)
```

See [docs/architecture.md](docs/architecture.md) for full details.

## Models

| Model | Link |
|---|---|
| Fine-tuned chatbot (GGUF) | [kevinjoythomas/medical-loratuned-chatbot-GGUF](https://huggingface.co/kevinjoythomas/medical-loratuned-chatbot-GGUF) |
| Training dataset | [lavita/ChatDoctor-HealthCareMagic-100k](https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k) |
| Embeddings | nomic-embed-text (via Ollama) |

**Training details:** LoRA rank=8, alpha=16, 20.97M trainable params. Trained for 1,000 steps (7.1% of dataset), loss 3.74→2.19. Model was undertrained at cutoff — full convergence would require ~14,000 steps on a T4 GPU.

## Quick Start (Local, No Docker)

```bash
# 1. Pull models
ollama pull hf.co/kevinjoythomas/medical-loratuned-chatbot-GGUF
ollama pull nomic-embed-text

# 2. Configure environment
cp .env.example .env   # edit as needed

# 3. Start services (separate terminals or use run_local.ps1)
cd services/auth-service        && pip install -r requirements.txt && python -m flask run --port 5001
cd services/chat-orchestrator   && pip install -r requirements.txt && python -m flask run --port 5002
cd services/retrieval-service   && pip install -r requirements.txt && python -m flask run --port 5003
cd services/api-gateway         && pip install -r requirements.txt && python -m flask run --port 5000
cd website                      && python serve.py

# 4. Ingest PDFs
cd services/retrieval-service && python app/populate_db.py

# 5. Pre-warm model (prevents cold-start timeout)
ollama run openhealth-doctor "hi"
```

See [docs/running-locally.md](docs/running-locally.md) for full setup guide.

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

## API Reference

Base URL: `http://localhost:5000/v1`  
Full spec: [docs/api-spec.yaml](docs/api-spec.yaml)

| Method | Endpoint | Description |
|---|---|---|
| POST | /v1/auth/signup | Create account |
| POST | /v1/auth/login | Login, returns JWT |
| POST | /v1/auth/refresh | Refresh access token |
| GET | /v1/auth/me | Current user |
| POST | /v1/chat/sessions | New chat session |
| GET | /v1/chat/sessions | List sessions |
| POST | /v1/chat/sessions/{id}/messages | Send message |
| GET | /v1/chat/sessions/{id}/messages | Message history |
| DELETE | /v1/chat/sessions/{id} | Delete session |
| POST | /v1/ingest/document | Ingest PDF into RAG |

## Evaluation Reports

- [Model Evaluation Report](model_eval_report.html) — ROUGE scores, rubric scoring, training analysis
- [Retrieval Report](retrieval_report.html) — BM25 vs vector vs hybrid comparison
- [docs/model-evaluation.md](docs/model-evaluation.md) — Summary of all results
- [docs/retrieval-evaluation.md](docs/retrieval-evaluation.md) — Retrieval strategy details

## Deployment

Frontend is deployed statically on Vercel from the `deploy/` folder.  
Backend runs locally and is exposed via ngrok.  
See [docs/architecture.md](docs/architecture.md) for deployment details.

## Diagnostic Models (Legacy)

The earlier web application also includes supporting models for:
- Brain tumor detection
- Lung disease detection
- Breast cancer detection
- Diabetes risk prediction

These live in `backend/` and are not part of the active microservices architecture.
