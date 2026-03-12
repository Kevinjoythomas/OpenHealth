import uuid
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from upstash_redis import Redis

from app.config import Config

_redis: Redis | None = None


def get_redis() -> Redis:
    global _redis
    if _redis is None:
        _redis = Redis(url=Config.UPSTASH_REDIS_REST_URL, token=Config.UPSTASH_REDIS_REST_TOKEN)
    return _redis


# --- Password hashing ---

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(
        plain.encode(), bcrypt.gensalt(rounds=Config.BCRYPT_COST_FACTOR)
    ).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


# --- JWT tokens ---

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


def store_refresh_token(user_id: uuid.UUID, jti: str) -> None:
    """Store refresh token JTI in Redis with TTL matching token expiry."""
    r = get_redis()
    key = f"refresh:{user_id}:{jti}"
    r.set(key, "1", ex=Config.REFRESH_TOKEN_EXPIRE_DAYS * 86400)


def revoke_refresh_token(user_id: uuid.UUID, jti: str) -> None:
    """Delete a refresh token from Redis (rotation: old token becomes invalid)."""
    r = get_redis()
    r.delete(f"refresh:{user_id}:{jti}")


def is_refresh_token_valid(user_id: uuid.UUID, jti: str) -> bool:
    """Check if the refresh token JTI still exists in Redis."""
    r = get_redis()
    return r.exists(f"refresh:{user_id}:{jti}") >= 1
