from flask import Flask
from flask_cors import CORS

from app.config import Config
from app.database import db
from app.routes import auth_bp


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)

    CORS(app)
    db.init_app(app)
    app.register_blueprint(auth_bp)

    with app.app_context():
        import app.models  # noqa: F401 — ensure models are registered
        db.create_all()

    return app


if __name__ == "__main__":
    application = create_app()
    application.run(host="0.0.0.0", port=5001, debug=True)
