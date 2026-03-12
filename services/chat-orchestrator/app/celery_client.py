"""Thin client to send tasks to the ingestion-worker via Celery/RabbitMQ."""
import os
from celery import Celery

_celery: Celery | None = None


def _get_celery() -> Celery:
    global _celery
    if _celery is None:
        broker = os.getenv(
            "CELERY_BROKER_URL", "amqp://openhealth:openhealth@rabbitmq:5672//"
        )
        backend = os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/1")
        _celery = Celery("openhealth", broker=broker, backend=backend)
    return _celery


def enqueue_ingest(s3_key: str, filename: str) -> str:
    """Send an ingest_document task to the ingestion-worker.

    Returns the Celery task ID.
    """
    celery = _get_celery()
    result = celery.send_task(
        "ingestion_worker.tasks.ingest_document",
        kwargs={"s3_key": s3_key, "filename": filename},
    )
    return result.id
