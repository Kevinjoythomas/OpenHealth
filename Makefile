.PHONY: up down build test logs ps shell-auth shell-chat shell-retrieval shell-worker migrate

# ── Environment ───────────────────────────────────────────────────────────────
COMPOSE = docker compose
ENV_FILE = .env

$(ENV_FILE):
	cp .env.example $(ENV_FILE)
	@echo "Created .env from .env.example — edit it before starting services."

# ── Lifecycle ─────────────────────────────────────────────────────────────────

up: $(ENV_FILE)
	$(COMPOSE) up -d

down:
	$(COMPOSE) down

build:
	$(COMPOSE) build --no-cache

restart:
	$(COMPOSE) restart

ps:
	$(COMPOSE) ps

# ── Logs ──────────────────────────────────────────────────────────────────────

logs:
	$(COMPOSE) logs -f

logs-gateway:
	$(COMPOSE) logs -f api-gateway

logs-auth:
	$(COMPOSE) logs -f auth-service

logs-chat:
	$(COMPOSE) logs -f chat-orchestrator

logs-retrieval:
	$(COMPOSE) logs -f retrieval-service

logs-worker:
	$(COMPOSE) logs -f ingestion-worker

# ── Database migrations ───────────────────────────────────────────────────────

migrate:
	$(COMPOSE) exec auth-service alembic upgrade head

migrate-chat:
	$(COMPOSE) exec chat-orchestrator alembic upgrade head

# ── Tests ─────────────────────────────────────────────────────────────────────

test:
	$(COMPOSE) run --rm auth-service python -m pytest tests/ -v
	$(COMPOSE) run --rm chat-orchestrator python -m pytest tests/ -v
	$(COMPOSE) run --rm retrieval-service python -m pytest tests/ -v
	$(COMPOSE) run --rm api-gateway python -m pytest tests/ -v

test-auth:
	$(COMPOSE) run --rm auth-service python -m pytest tests/ -v

test-chat:
	$(COMPOSE) run --rm chat-orchestrator python -m pytest tests/ -v

test-retrieval:
	$(COMPOSE) run --rm retrieval-service python -m pytest tests/ -v

test-gateway:
	$(COMPOSE) run --rm api-gateway python -m pytest tests/ -v

# ── Shell access ──────────────────────────────────────────────────────────────

shell-auth:
	$(COMPOSE) exec auth-service bash

shell-chat:
	$(COMPOSE) exec chat-orchestrator bash

shell-retrieval:
	$(COMPOSE) exec retrieval-service bash

shell-worker:
	$(COMPOSE) exec ingestion-worker bash

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	$(COMPOSE) down -v --remove-orphans
	docker image prune -f
