"""Postgres helpers for ingestion metadata.

Tracks every successfully ingested document with its content hash so that
re-uploading the same file is a no-op (idempotent by content_hash).

Table is created on first use (CREATE TABLE IF NOT EXISTS) — no migration
runner needed in the worker context.
"""
import os
import logging
from contextlib import contextmanager

import psycopg2
import psycopg2.extras

log = logging.getLogger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS ingestion_metadata (
    id              SERIAL PRIMARY KEY,
    s3_key          TEXT NOT NULL,
    filename        TEXT,
    content_hash    TEXT NOT NULL,
    chunks_added    INTEGER NOT NULL DEFAULT 0,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (content_hash)
);
"""


@contextmanager
def _get_conn():
    url = os.getenv("DATABASE_URL")
    conn = psycopg2.connect(url)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def ensure_table() -> None:
    """Create ingestion_metadata table if it does not exist."""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(_DDL)
    log.debug("ingestion_metadata table ready")


def is_already_ingested(content_hash: str) -> bool:
    """Return True if a document with this content_hash was already ingested."""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM ingestion_metadata WHERE content_hash = %s",
                (content_hash,),
            )
            return cur.fetchone() is not None


def record_ingestion(
    s3_key: str,
    filename: str,
    content_hash: str,
    chunks_added: int,
) -> None:
    """Upsert an ingestion record keyed by content_hash."""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ingestion_metadata (s3_key, filename, content_hash, chunks_added)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (content_hash) DO UPDATE SET
                    s3_key       = EXCLUDED.s3_key,
                    filename     = EXCLUDED.filename,
                    chunks_added = EXCLUDED.chunks_added,
                    ingested_at  = NOW()
                """,
                (s3_key, filename, content_hash, chunks_added),
            )
