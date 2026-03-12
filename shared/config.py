"""Shared configuration helpers.

Each service instantiates its own Config subclass, but common env-var
loading and constants live here to avoid duplication.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class BaseConfig:
    # ── Database ──────────────────────────────────────────────────────────────
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://openhealth:openhealth@postgres:5432/openhealth",
    )
    SQLALCHEMY_TRACK_MODIFICATIONS: bool = False

    # ── Redis ─────────────────────────────────────────────────────────────────
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis:6379/0")

    # ── JWT ───────────────────────────────────────────────────────────────────
    JWT_SECRET: str = os.getenv("JWT_SECRET", "CHANGE-ME-in-production")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")

    # ── Service discovery ─────────────────────────────────────────────────────
    AUTH_SERVICE_URL: str = os.getenv("AUTH_SERVICE_URL", "http://auth-service:5001")
    CHAT_SERVICE_URL: str = os.getenv("CHAT_SERVICE_URL", "http://chat-orchestrator:5002")
    RETRIEVAL_SERVICE_URL: str = os.getenv(
        "RETRIEVAL_SERVICE_URL", "http://retrieval-service:5003"
    )

    # ── Ollama / LLM ──────────────────────────────────────────────────────────
    OLLAMA_BASE_URL: str = os.getenv(
        "OLLAMA_BASE_URL", "http://host.docker.internal:11434"
    )
    LLM_MODEL: str = os.getenv(
        "LLM_MODEL", "hf.co/kevinjoythomas/medical-loratuned-chatbot-GGUF"
    )
    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "nomic-embed-text")

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    CHROMA_PATH: str = os.getenv("CHROMA_PATH", "./chroma")

    # ── Celery ────────────────────────────────────────────────────────────────
    CELERY_BROKER_URL: str = os.getenv(
        "CELERY_BROKER_URL", "amqp://openhealth:openhealth@rabbitmq:5672//"
    )
    CELERY_RESULT_BACKEND: str = os.getenv(
        "CELERY_RESULT_BACKEND", "redis://redis:6379/1"
    )
