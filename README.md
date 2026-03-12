# OpenHealth

OpenHealth is a medical AI platform that combines a chatbot, retrieval-augmented generation, and domain-specific diagnostic models inside a microservices monorepo.

> The previous monolith lives in `website/` for reference. The active backend codebase is in `services/`.

## Models and Data

- Chat model: [kevinjoythomas/medical-loratuned-chatbot-GGUF](https://huggingface.co/kevinjoythomas/medical-loratuned-chatbot-GGUF)
- Training dataset: [lavita/ChatDoctor-HealthCareMagic-100k](https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k)

## Key Features

### AI Chatbot

The chatbot is the core OpenHealth interface for doctors and patients. It is LoRA fine-tuned with Unsloth and integrated into a retrieval pipeline for medical question answering.

- Retrieval-augmented generation over curated medical PDFs
- Context-aware conversations with stored chat history
- Microservice-based chat orchestration with Redis, Postgres, RabbitMQ, and ChromaDB

### Diagnostic Models

OpenHealth also includes supporting models for:

- Brain tumor detection
- Lung disease detection
- Breast cancer detection
- Diabetes risk prediction

### Collaboration

The platform also includes doctor-to-doctor collaboration and notification flows from the earlier web application.

## Architecture

```text
api-gateway (:5000) -> auth-service (:5001)
                    -> chat-orchestrator (:5002)
                         -> retrieval-service (:5003)
                         -> Ollama (host)
ingestion-worker <- RabbitMQ
```

| Service | Port | Responsibility |
|---|---|---|
| api-gateway | 5000 | JWT validation, rate limiting, reverse proxy |
| auth-service | 5001 | Signup/login, JWT issue/refresh, password auth |
| chat-orchestrator | 5002 | Chat sessions, orchestration, RAG pipeline |
| retrieval-service | 5003 | ChromaDB hybrid retrieval |
| ingestion-worker | - | Chunking, embedding, ingestion jobs |

## Prerequisites

- Docker 24+ and Docker Compose v2
- Ollama running on the host with:

```bash
ollama pull hf.co/kevinjoythomas/medical-loratuned-chatbot-GGUF
ollama pull nomic-embed-text
```

## Quick Start

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

## Useful Commands

```bash
make logs
make logs-chat
make test
make migrate
make shell-auth
make clean
```
