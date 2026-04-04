import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=r"C:\OpenHealth\.env", override=True)


class Config:
    SQLALCHEMY_DATABASE_URI: str = os.getenv(
        "DATABASE_URL",
        "postgresql://openhealth:openhealth@postgres:5432/openhealth",
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY: str = os.getenv("JWT_SECRET", "CHANGE-ME-in-production")

    REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis:6379/0")
    REDIS_CACHE_TTL: int = int(os.getenv("REDIS_CACHE_TTL", "3600"))  # 1 hour

    JWT_SECRET: str = os.getenv("JWT_SECRET", "CHANGE-ME-in-production")
    JWT_ALGORITHM: str = "HS256"

    RETRIEVAL_SERVICE_URL: str = os.getenv(
        "RETRIEVAL_SERVICE_URL", "http://retrieval-service:5003"
    )
    OLLAMA_BASE_URL: str = os.getenv(
        "OLLAMA_BASE_URL", "http://host.docker.internal:11434"
    )
    LLM_MODEL: str = os.getenv(
        "LLM_MODEL", "hf.co/kevinjoythomas/medical-loratuned-chatbot-GGUF"
    )
    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "nomic-embed-text")

    # How many recent messages to cache in Redis per session
    RECENT_MESSAGES_LIMIT: int = int(os.getenv("RECENT_MESSAGES_LIMIT", "20"))

    # Celery (for enqueuing ingestion tasks)
    CELERY_BROKER_URL: str = os.getenv(
        "CELERY_BROKER_URL", "amqp://openhealth:openhealth@rabbitmq:5672//"
    )
