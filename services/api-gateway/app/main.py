import os
import logging

from flask import Flask, jsonify
from flask_cors import CORS

from app.middleware.rate_limit import RateLimiter
from app.routes.auth import auth_proxy_bp
from app.routes.chat import chat_proxy_bp
from app.routes.ingest import ingest_proxy_bp


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    # ── Structured logging ────────────────────────────────────────────────────
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

    # ── Rate limiter (attaches to app) ────────────────────────────────────────
    RateLimiter(app)

    # ── Blueprints ────────────────────────────────────────────────────────────
    app.register_blueprint(auth_proxy_bp)
    app.register_blueprint(chat_proxy_bp)
    app.register_blueprint(ingest_proxy_bp)

    # ── Health endpoints ──────────────────────────────────────────────────────
    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "service": "api-gateway"}), 200

    @app.get("/health/ready")
    def ready():
        """Liveness check: verify downstream services are reachable."""
        import requests
        auth_url = os.getenv("AUTH_SERVICE_URL", "http://auth-service:5001")
        chat_url = os.getenv("CHAT_SERVICE_URL", "http://chat-orchestrator:5002")
        issues = []
        for name, url in [("auth-service", auth_url), ("chat-orchestrator", chat_url)]:
            try:
                r = requests.get(f"{url}/health", timeout=3)
                if r.status_code != 200:
                    issues.append(f"{name}: HTTP {r.status_code}")
            except Exception as exc:
                issues.append(f"{name}: {exc}")
        if issues:
            return jsonify({"status": "degraded", "issues": issues}), 503
        return jsonify({"status": "ready"}), 200

    return app


if __name__ == "__main__":
    application = create_app()
    application.run(host="0.0.0.0", port=int(os.getenv("SERVICE_PORT", 5000)))
