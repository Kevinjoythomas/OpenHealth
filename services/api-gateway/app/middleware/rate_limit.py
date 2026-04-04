"""Redis-based sliding-window rate limiter.

Limits each IP (or authenticated user) to RATE_LIMIT_PER_MINUTE requests
per 60-second window using a Redis sorted set.
"""
import os
import time
import logging

import redis
from flask import Flask, request, jsonify, g

log = logging.getLogger(__name__)


class RateLimiter:
    def __init__(self, app: Flask | None = None):
        self._redis: redis.Redis | None = None
        self.limit = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
        self.window = 60  # seconds
        if app is not None:
            self.init_app(app)

    def init_app(self, app: Flask) -> None:
        app.before_request(self._check_rate_limit)
        app.extensions["rate_limiter"] = self

    def _get_redis(self) -> redis.Redis:
        if self._redis is None:
            if os.getenv("USE_FAKEREDIS", "0") == "1":
                import fakeredis
                self._redis = fakeredis.FakeRedis(decode_responses=True)
            else:
                self._redis = redis.from_url(
                    os.getenv("REDIS_URL", "redis://redis:6379/0"),
                    decode_responses=True,
                )
        return self._redis

    def _check_rate_limit(self):
        # Skip health probes
        if request.path in ("/health", "/health/ready"):
            return None

        identifier = _get_identifier()
        key = f"rl:{identifier}"
        now = time.time()
        window_start = now - self.window

        try:
            r = self._get_redis()
            pipe = r.pipeline()
            pipe.zremrangebyscore(key, "-inf", window_start)
            pipe.zadd(key, {str(now): now})
            pipe.zcard(key)
            pipe.expire(key, self.window)
            _, _, count, _ = pipe.execute()
        except Exception as exc:
            log.warning("Rate limiter Redis error: %s — allowing request", exc)
            return None

        if count > self.limit:
            return jsonify({
                "error": "Rate limit exceeded",
                "retry_after": self.window,
            }), 429
        return None


def _get_identifier() -> str:
    """Use authenticated user ID when available, fall back to IP."""
    if hasattr(g, "user_id"):
        return f"user:{g.user_id}"
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return f"ip:{forwarded.split(',')[0].strip()}"
    return f"ip:{request.remote_addr}"
