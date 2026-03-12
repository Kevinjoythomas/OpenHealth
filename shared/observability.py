"""Structured JSON logging setup used by all services.

Usage in a Flask app factory::

    from shared.observability import configure_logging, get_logger
    configure_logging(service_name="auth-service")
    log = get_logger(__name__)
    log.info("Server starting", port=5001)
"""
import logging
import os
import sys

from pythonjsonlogger import jsonlogger


def configure_logging(service_name: str = "openhealth", level: str | None = None) -> None:
    """Configure root logger with JSON output and a service_name field."""
    log_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()

    handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(log_level)

    # Inject service name into every record via a filter
    class ServiceFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            record.service = service_name
            return True

    handler.addFilter(ServiceFilter())


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
