import uuid
from functools import wraps

from flask import request, jsonify, g

from app.auth import decode_token
from app.database import db
from app.models import User, Role


def require_auth(f):
    """Decorator that verifies the Bearer JWT and attaches the user to g.current_user."""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or malformed Authorization header"}), 401

        token = auth_header.split(" ", 1)[1]
        try:
            claims = decode_token(token)
        except Exception:
            return jsonify({"error": "Invalid or expired token"}), 401

        if claims.get("type") != "access":
            return jsonify({"error": "Token is not an access token"}), 401

        user = db.session.get(User, uuid.UUID(claims["sub"]))
        if not user:
            return jsonify({"error": "User not found"}), 401

        g.current_user = user
        return f(*args, **kwargs)
    return decorated


def require_role(*allowed: Role):
    """Decorator factory for RBAC. Usage: @require_role(Role.ADMIN)"""
    def decorator(f):
        @wraps(f)
        @require_auth
        def decorated(*args, **kwargs):
            if g.current_user.role not in allowed:
                return jsonify({"error": "Insufficient permissions"}), 403
            return f(*args, **kwargs)
        return decorated
    return decorator
