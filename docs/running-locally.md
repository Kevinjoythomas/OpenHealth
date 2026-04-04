# Running Locally

## Prerequisites

- Python 3.11+
- Ollama installed and running
- Models pulled (see below)

## Pull Models

```bash
ollama pull hf.co/kevinjoythomas/medical-loratuned-chatbot-GGUF
ollama pull nomic-embed-text
```

## Start All Services

Run each command in a separate terminal:

```bash
# Auth service
cd services/auth-service && pip install -r requirements.txt && python -m flask run --port 5001

# Chat orchestrator
cd services/chat-orchestrator && pip install -r requirements.txt && python -m flask run --port 5002

# Retrieval service
cd services/retrieval-service && pip install -r requirements.txt && python -m flask run --port 5003

# API gateway
cd services/api-gateway && pip install -r requirements.txt && python -m flask run --port 5000

# Frontend
cd website && python serve.py
```

Alternatively, use the PowerShell script from the project root:

```powershell
./run_local.ps1
```

## Ingest PDFs into ChromaDB

```bash
cd services/retrieval-service
python app/populate_db.py
```

## Environment Configuration

Copy `.env` to the project root. Services auto-detect dev mode and use SQLite (instead of Postgres) and fakeredis (instead of Redis) automatically.

## Pre-warm Ollama

Run this before making chat requests to avoid a cold-start timeout on the first request:

```bash
ollama run openhealth-doctor "hi"
```

## Test the API

```bash
# Health check
curl http://localhost:5000/health

# Signup
curl -X POST http://localhost:5000/v1/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"name":"Test","email":"test@test.com","password":"password123","role":"doctor"}'

# Login
curl -X POST http://localhost:5000/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@test.com","password":"password123"}'
```

## Run Evaluations

```bash
# 8-case rubric evaluation (fast, ~5 min)
python eval_model.py

# ROUGE evaluation (slow, ~3 hours for 50 samples per condition)
python rouge_eval.py

# Retrieval latency benchmark
python benchmark_retrieval.py
```
