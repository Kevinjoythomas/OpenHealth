import os

from flask import Flask, jsonify
from flask_cors import CORS

from app.config import Config
from app.routes import retrieve_bp


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)

    CORS(app)
    app.register_blueprint(retrieve_bp)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "service": "retrieval-service"}), 200

    @app.get("/health/ready")
    def ready():
        """Check ChromaDB is accessible."""
        try:
            from app.vector_store import get_chroma
            get_chroma()
            return jsonify({"status": "ready"}), 200
        except Exception as exc:
            return jsonify({"status": "not ready", "error": str(exc)}), 503

    return app


if __name__ == "__main__":
    application = create_app()
    application.run(host="0.0.0.0", port=int(os.getenv("SERVICE_PORT", 5003)))
