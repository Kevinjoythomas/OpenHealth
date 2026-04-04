"""Password hashing, JWT generation/verification, and Redis refresh-token store.

Uses standard redis-py (not upstash) so it works with the Dockerised Redis
container as well as any Redis-compatible service.
"""
import os
import uuid
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
import redis

from app.config import Config

_redis_client: redis.Redis | None = None


def get_redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        if os.getenv("USE_FAKEREDIS", "0") == "1":
            import fakeredis
            _redis_client = fakeredis.FakeRedis(decode_responses=True)
        else:
            _redis_client = redis.from_url(Config.REDIS_URL, decode_responses=True)
    return _redis_client


# ── Password ──────────────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(
        plain.encode(), bcrypt.gensalt(rounds=Config.BCRYPT_COST_FACTOR)
    ).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


# ── JWT ───────────────────────────────────────────────────────────────────────

def create_access_token(user_id: uuid.UUID, role: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "role": role,
        "type": "access",
        "iat": now,
        "exp": now + timedelta(minutes=Config.ACCESS_TOKEN_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, Config.JWT_SECRET, algorithm=Config.JWT_ALGORITHM)


def create_refresh_token(user_id: uuid.UUID) -> str:
    now = datetime.now(timezone.utc)
    jti = uuid.uuid4().hex
    payload = {
        "sub": str(user_id),
        "type": "refresh",
        "jti": jti,
        "iat": now,
        "exp": now + timedelta(days=Config.REFRESH_TOKEN_EXPIRE_DAYS),
    }
    return jwt.encode(payload, Config.JWT_SECRET, algorithm=Config.JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    """Decode and verify a JWT. Raises jwt.InvalidTokenError on failure."""
    return jwt.decode(token, Config.JWT_SECRET, algorithms=[Config.JWT_ALGORITHM])


# ── Refresh token store (Redis) ───────────────────────────────────────────────

def store_refresh_token(user_id: uuid.UUID, jti: str) -> None:
    r = get_redis()
    key = f"refresh:{user_id}:{jti}"
    r.set(key, "1", ex=Config.REFRESH_TOKEN_EXPIRE_DAYS * 86400)


def revoke_refresh_token(user_id: uuid.UUID, jti: str) -> None:
    r = get_redis()
    r.delete(f"refresh:{user_id}:{jti}")


def is_refresh_token_valid(user_id: uuid.UUID, jti: str) -> bool:
    r = get_redis()
    return r.exists(f"refresh:{user_id}:{jti}") >= 1
