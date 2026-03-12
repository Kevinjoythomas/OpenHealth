"""JWT validation middleware for the API gateway.

Validates the Bearer token in incoming requests using the shared JWT_SECRET.
Attaches the decoded claims to Flask's g object so downstream route handlers
can access user identity without re-decoding.
"""
import os
from functools import wraps

import jwt
from flask import request, jsonify, g


JWT_SECRET = os.getenv("JWT_SECRET", "CHANGE-ME-in-production")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")


def require_auth(f):
    """Decorator: validates Bearer JWT, populates g.claims."""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or malformed Authorization header"}), 401
        token = auth_header.split(" ", 1)[1]
        try:
            claims = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token has expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401
        if claims.get("type") != "access":
            return jsonify({"error": "Token is not an access token"}), 401
        g.claims = claims
        g.user_id = claims["sub"]
        return f(*args, **kwargs)
    return decorated
