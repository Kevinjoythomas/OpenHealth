import os

from flask import Flask, jsonify
from flask_cors import CORS

from app.config import Config
from app.db import db
from app.routes import chat_bp, ingest_bp


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)

    CORS(app)
    db.init_app(app)
    app.register_blueprint(chat_bp)
    app.register_blueprint(ingest_bp)

    with app.app_context():
        import app.models  # noqa: F401
        db.create_all()

    # ── Health endpoints ──────────────────────────────────────────────────────
    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "service": "chat-orchestrator"}), 200

    @app.get("/health/ready")
    def ready():
        try:
            db.session.execute(db.text("SELECT 1"))
            return jsonify({"status": "ready"}), 200
        except Exception as exc:
            return jsonify({"status": "not ready", "error": str(exc)}), 503

    return app


if __name__ == "__main__":
    application = create_app()
    application.run(host="0.0.0.0", port=int(os.getenv("SERVICE_PORT", 5002)))
