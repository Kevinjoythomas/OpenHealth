# Architecture

## Services

| Service | Port | Stack | Responsibility |
|---|---|---|---|
| api-gateway | 5000 | Flask | JWT validation, rate limiting, reverse proxy |
| auth-service | 5001 | Flask + SQLite/Postgres | Signup/login, JWT issue/refresh |
| chat-orchestrator | 5002 | Flask + SQLite/Postgres | Sessions, RAG orchestration |
| retrieval-service | 5003 | Flask + ChromaDB + BM25 | Hybrid search, RRF fusion |
| Ollama | host | GGUF | Model inference (Llama 3 fine-tuned) |

---

## Local Development

In local development mode, external infrastructure dependencies are replaced with lightweight alternatives:

- **fakeredis** replaces Redis for rate limiting
- **SQLite** replaces Postgres for auth and session storage
- Each service reads from a single shared `.env` file at the project root

---

## Deployment

| Component | Platform | Details |
|---|---|---|
| Frontend | Vercel | Static files from `deploy/` folder |
| Backend | Local + ngrok | Exposed via ngrok static tunnel |

**ngrok URL:** `https://nonsignificantly-untippled-mikaela.ngrok-free.dev`

All frontend API calls must include the header `ngrok-skip-browser-warning: true` to bypass the ngrok browser interstitial page.

---

## Chat Flow

1. User sends a message → api-gateway validates the JWT
2. api-gateway forwards the request to chat-orchestrator
3. chat-orchestrator calls retrieval-service with the user's query
4. retrieval-service runs hybrid BM25 + vector search concurrently, applies RRF and the relevance threshold
5. Top-K chunks are returned to the orchestrator
6. Orchestrator builds the prompt: system prompt + conversation history + context chunks + user message
7. Prompt is sent to Ollama (running locally) → response is streamed back
8. Response and sources are saved to the session history

---

## Model Serving

Pull the required models with:

```bash
ollama pull hf.co/kevinjoythomas/medical-loratuned-chatbot-GGUF
ollama pull nomic-embed-text
```

The fine-tuned chatbot model is served in GGUF Q4_K_M quantization format. The nomic-embed-text model is used by the retrieval service to generate vector embeddings for semantic search.
