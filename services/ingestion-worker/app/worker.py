"""Celery application for the ingestion worker.

Consume from the 'ingestion' queue on RabbitMQ. Tasks are defined in tasks.py.
"""
import os
from celery import Celery
from celery.utils.log import get_task_logger

from dotenv import load_dotenv

load_dotenv()

BROKER_URL = os.getenv(
    "CELERY_BROKER_URL", "amqp://openhealth:openhealth@rabbitmq:5672//"
)
RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/1")

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
    worker_prefetch_multiplier=1,  # process one task at a time (LLM-heavy)
)
