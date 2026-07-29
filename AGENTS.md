OpenHealth Backend Redesign — Full Execution Plan

Current State Analysis

Your app.py is a single-file Flask monolith with everything coupled together:

ProblemWhereImpactStateful globalssession\_histories = {}, count, session\_idDies on restart, can't scale horizontallyIn-memory chat historyInMemoryChatMessageHistory()Lost on every restartPlaintext passwordsuser\['password']==password (line 381)Critical security vulnerabilityFirebase as primary DBFirestore for users, notificationsNo relational integrity, no transactionsSynchronous ingestionpopulate\_db.py blocks until doneCan't handle concurrent uploadsLRU cache on LLM calls@lru\_cache(maxsize=128) on run\_conversationBreaks chat history (same input ≠ same context)Mixed concernsAuth + Chat + ML + Notifications in one fileUntestable, undeployable independentlyNo error handlingLLM failures crash the requestNo retries, no fallbackExposed secretsopenhealth.json has private keys in repoCredential leak



Target Architecture

┌─────────────────────────────────────────────────────────────────┐

│                        CLIENTS (React SPA)                       │

└──────────────────────────────┬──────────────────────────────────┘

&nbsp;                              │

&nbsp;                        ┌─────▼─────┐

&nbsp;                        │ API Gateway│  (Kong / AWS ALB)

&nbsp;                        │  + Auth    │  JWT validation, rate limiting

&nbsp;                        └─────┬─────┘

&nbsp;                              │

&nbsp;         ┌────────────────────┼────────────────────┐

&nbsp;         │                    │                     │

&nbsp;  ┌──────▼──────┐     ┌──────▼──────┐      ┌──────▼──────┐

&nbsp;  │ Auth Service │     │    Chat     │      │  Ingestion  │

&nbsp;  │  (FastAPI)   │     │ Orchestrator│      │   Worker    │

&nbsp;  │              │     │  (FastAPI)  │      │  (Celery)   │

&nbsp;  └──────┬──────┘     └──────┬──────┘      └──────┬──────┘

&nbsp;         │                   │                     │

&nbsp;         │            ┌──────▼──────┐              │

&nbsp;         │            │  Retrieval  │              │

&nbsp;         │            │   Service   │              │

&nbsp;         │            │  (FastAPI)  │              │

&nbsp;         │            └──────┬──────┘              │

&nbsp;         │                   │                     │

&nbsp;  ┌──────▼───────────────────▼─────────────────────▼──────┐

&nbsp;  │                    DATA LAYER                          │

&nbsp;  │  ┌──────────┐  ┌──────────┐  ┌──────┐  ┌──────────┐  │

&nbsp;  │  │ Postgres │  │  Redis   │  │  S3  │  │ Pgvector │  │

&nbsp;  │  │ (users,  │  │ (cache,  │  │(docs)│  │ /ChromaDB│  │

&nbsp;  │  │ sessions)│  │  rate    │  │      │  │ (vectors)│  │

&nbsp;  │  └──────────┘  │  limit)  │  └──────┘  └──────────┘  │

&nbsp;  │                └──────────┘                           │

&nbsp;  └───────────────────────────────────────────────────────┘

