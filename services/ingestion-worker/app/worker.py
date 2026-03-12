"""Celery application for the ingestion worker.

Consume from the 'ingestion' queue on RabbitMQ. Tasks are defined in tasks.py.

Dead-letter queue (DLQ) setup:
- Tasks that exhaust all retries are rejected and routed to 'ingestion.failed'
  via the RabbitMQ dead-letter exchange 'ingestion.dlx'.
- acks_late + reject_on_worker_lost ensures a task is re-queued (not lost) if
  the worker crashes mid-execution.
"""
import os
from celery import Celery
from kombu import Queue, Exchange

from dotenv import load_dotenv

load_dotenv()

BROKER_URL = os.getenv(
    "CELERY_BROKER_URL", "amqp://openhealth:openhealth@rabbitmq:5672//"
)
RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/1")

# Dead-letter exchange and queues
_dlx = Exchange("ingestion.dlx", type="direct", durable=True)
_ingestion_queue = Queue(
    "ingestion",
    Exchange("ingestion", type="direct", durable=True),
    routing_key="ingestion",
    durable=True,
    queue_arguments={
        "x-dead-letter-exchange": "ingestion.dlx",
        "x-dead-letter-routing-key": "ingestion.failed",
    },
)
_dlq = Queue(
    "ingestion.failed",
    _dlx,
    routing_key="ingestion.failed",
    durable=True,
)

celery_app = Celery(
    "ingestion_worker",
    broker=BROKER_URL,
    backend=RESULT_BACKEND,
    include=["app.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_default_queue="ingestion",
    task_routes={"app.tasks.*": {"queue": "ingestion"}},
    task_queues=[_ingestion_queue, _dlq],
    worker_prefetch_multiplier=1,  # process one task at a time (LLM-heavy)
    task_acks_late=True,           # ack only after successful completion
    task_reject_on_worker_lost=True,  # re-queue if worker crashes mid-task
)
