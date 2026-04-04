"""Chat session CRUD with Redis caching.

Read path: check Redis first (recent N messages), fall back to Postgres.
Write path: write to Postgres, then invalidate / update Redis cache.
"""
import json
import logging
import re
import uuid

import redis
from flask import current_app

from app.db import db
from app.models import ChatSession, ChatMessage, MessageRole

log = logging.getLogger(__name__)

_redis_client: redis.Redis | None = None

_GENERIC_SESSION_TITLES = {
    "chat",
    "consultation",
    "new chat",
    "new consultation",
    "untitled",
}
_LEADING_TITLE_PATTERNS = [
    r"^(?:hi|hello|hey)\b[\s,!:.-]*",
    r"^(?:can|could|would)\s+you\s+(?:please\s+)?(?:help(?: me)? with|help|tell me about|explain|advise(?: me)? on)\s+",
    r"^(?:please\s+)?(?:help(?: me)? with|tell me about|explain|advise(?: me)? on|i need help with)\s+",
    r"^(?:what should i do about|should i worry about|is it normal to have|is it normal for)\s+",
    r"^(?:i have|i'm having|i am having|i've been having|i have been having|i feel|my)\s+",
]


def get_redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        import os
        if os.getenv("USE_FAKEREDIS", "0") == "1":
            import fakeredis
            _redis_client = fakeredis.FakeRedis(decode_responses=True)
        else:
            _redis_client = redis.from_url(
                current_app.config["REDIS_URL"], decode_responses=True
            )
    return _redis_client


def _session_cache_key(session_id: str) -> str:
    return f"chat:session:{session_id}:messages"


def _normalise_text(value: str | None) -> str:
    return " ".join((value or "").split())


def _needs_generated_title(title: str | None) -> bool:
    cleaned = _normalise_text(title).strip(" -:;,.!?").lower()
    return not cleaned or cleaned in _GENERIC_SESSION_TITLES


def _truncate_title(text: str, limit: int = 60) -> str:
    if len(text) <= limit:
        return text

    trimmed = text[: limit - 3].rstrip()
    if " " in trimmed:
        trimmed = trimmed.rsplit(" ", 1)[0]
    trimmed = trimmed.rstrip(" ,;:-")
    return (trimmed or text[: limit - 3].rstrip()) + "..."


def _generate_session_title(message: str) -> str:
    base = _normalise_text(message)
    if not base:
        return "New consultation"

    candidate = re.split(r"[\r\n]+", base, maxsplit=1)[0]
    sentence = re.split(r"(?<=[.!?])\s+", candidate, maxsplit=1)[0]
    candidate = sentence.strip(" -:;,.!?") or candidate.strip(" -:;,.!?")

    for pattern in _LEADING_TITLE_PATTERNS:
        updated = re.sub(pattern, "", candidate, count=1, flags=re.IGNORECASE)
        if updated != candidate:
            candidate = updated.strip(" -:;,.!?")
            break

    if not candidate:
        candidate = base.strip(" -:;,.!?")
    if not candidate:
        return "New consultation"

    candidate = candidate[0].upper() + candidate[1:]
    return _truncate_title(candidate, limit=60)


def _prepare_initial_title(title: str | None) -> str | None:
    cleaned = _normalise_text(title)
    if _needs_generated_title(cleaned):
        return None
    return cleaned[:255]


def _populate_missing_title(session: ChatSession | None) -> bool:
    if not session or not _needs_generated_title(session.title):
        return False

    first_user_message = (
        ChatMessage.query
        .filter_by(session_id=session.id, role=MessageRole.USER)
        .order_by(ChatMessage.created_at.asc())
        .first()
    )
    if not first_user_message:
        return False

    session.title = _generate_session_title(first_user_message.content)
    return True


# ── Session CRUD ──────────────────────────────────────────────────────────────

def create_session(user_id: str, title: str | None = None) -> ChatSession:
    session = ChatSession(user_id=user_id, title=_prepare_initial_title(title))
    db.session.add(session)
    db.session.commit()
    return session


def get_session(session_id: str, user_id: str) -> ChatSession | None:
    session = ChatSession.query.filter_by(
        id=uuid.UUID(session_id), user_id=user_id
    ).first()
    if _populate_missing_title(session):
        db.session.commit()
    return session


def list_sessions(user_id: str) -> list[ChatSession]:
    sessions = (
        ChatSession.query
        .filter_by(user_id=user_id)
        .order_by(ChatSession.updated_at.desc())
        .all()
    )
    changed = False
    for session in sessions:
        changed = _populate_missing_title(session) or changed
    if changed:
        db.session.commit()
    return sessions


def delete_session(session_id: str, user_id: str) -> bool:
    session = get_session(session_id, user_id)
    if not session:
        return False
    db.session.delete(session)
    db.session.commit()
    _invalidate_cache(session_id)
    return True


# ── Message CRUD ──────────────────────────────────────────────────────────────

def add_message(session_id: str, role: MessageRole, content: str) -> ChatMessage:
    session_uuid = uuid.UUID(session_id)
    msg = ChatMessage(
        session_id=session_uuid,
        role=role,
        content=content,
    )
    db.session.add(msg)

    # Bump session updated_at
    session = db.session.get(ChatSession, session_uuid)
    if session:
        from datetime import datetime, timezone
        session.updated_at = datetime.now(timezone.utc)
        if role == MessageRole.USER:
            db.session.flush()
            _populate_missing_title(session)

    db.session.commit()
    _append_to_cache(session_id, msg)
    return msg


def get_messages(session_id: str) -> list[ChatMessage]:
    """Return all messages for a session, using Redis cache when available."""
    cached = _load_from_cache(session_id)
    if cached is not None:
        return cached
    messages = (
        ChatMessage.query
        .filter_by(session_id=uuid.UUID(session_id))
        .order_by(ChatMessage.created_at)
        .all()
    )
    _populate_cache(session_id, messages)
    return messages


def get_recent_messages_for_llm(session_id: str, limit: int = 20) -> list[dict]:
    """Return the last `limit` messages as plain dicts for LangChain history."""
    messages = get_messages(session_id)
    recent = messages[-limit:] if len(messages) > limit else messages
    return [{"role": m.role.value, "content": m.content} for m in recent]


# ── Redis cache helpers ───────────────────────────────────────────────────────

def _populate_cache(session_id: str, messages: list[ChatMessage]) -> None:
    try:
        r = get_redis()
        key = _session_cache_key(session_id)
        limit = current_app.config.get("RECENT_MESSAGES_LIMIT", 20)
        recent = messages[-limit:]
        serialised = json.dumps([m.to_dict() for m in recent])
        r.set(key, serialised, ex=current_app.config.get("REDIS_CACHE_TTL", 3600))
    except Exception as exc:
        log.warning("Redis cache write failed: %s", exc)


def _append_to_cache(session_id: str, msg: ChatMessage) -> None:
    """Invalidate cache on new message — simpler than maintaining append-only log."""
    _invalidate_cache(session_id)


def _load_from_cache(session_id: str) -> list[ChatMessage] | None:
    """Returns list of ChatMessage-like objects from cache, or None on miss."""
    try:
        r = get_redis()
        raw = r.get(_session_cache_key(session_id))
        if raw is None:
            return None
        data = json.loads(raw)
        # Reconstruct lightweight objects for callers that only read .role/.content
        return [_dict_to_message(d) for d in data]
    except Exception as exc:
        log.warning("Redis cache read failed: %s", exc)
        return None


def _invalidate_cache(session_id: str) -> None:
    try:
        get_redis().delete(_session_cache_key(session_id))
    except Exception as exc:
        log.warning("Redis cache invalidation failed: %s", exc)


def _dict_to_message(d: dict) -> ChatMessage:
    """Reconstruct a ChatMessage from a cached dict (not a DB-backed object)."""
    msg = ChatMessage.__new__(ChatMessage)
    msg.id = d["id"]
    msg.session_id = d["session_id"]
    msg.role = MessageRole(d["role"])
    msg.content = d["content"]
    msg.created_at = d["created_at"]
    return msg
