from flask import Blueprint, request, jsonify, g

from app.db import db
from app.models import User, Role
from app.auth import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_token,
    store_refresh_token,
    revoke_refresh_token,
    is_refresh_token_valid,
)
from app.middleware import require_auth

auth_bp = Blueprint("auth", __name__, url_prefix="/v1/auth")


@auth_bp.post("/signup")
def signup():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body required"}), 400

    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    name = data.get("name", "").strip()

    if not all([email, password, name]):
        return jsonify({"error": "email, password, and name are required"}), 400
    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters"}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({"error": "Email already registered"}), 409

    role_str = data.get("role", "doctor")
    try:
        role = Role(role_str)
    except ValueError:
        return jsonify({"error": f"Invalid role: {role_str}"}), 400

    user = User(
        email=email,
        password_hash=hash_password(password),
        name=name,
        role=role,
        specialization=data.get("specialization"),
    )
    db.session.add(user)
    db.session.commit()

    access = create_access_token(user.id, user.role.value)
    refresh = create_refresh_token(user.id)
    claims = decode_token(refresh)
    store_refresh_token(user.id, claims["jti"])

    return jsonify({
        "access_token": access,
        "refresh_token": refresh,
        "token_type": "bearer",
    }), 201


@auth_bp.post("/login")
def login():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body required"}), 400

    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not all([email, password]):
        return jsonify({"error": "email and password are required"}), 400

    user = User.query.filter_by(email=email).first()
    if not user or not verify_password(password, user.password_hash):
        return jsonify({"error": "Invalid email or password"}), 401

    access = create_access_token(user.id, user.role.value)
    refresh = create_refresh_token(user.id)
    claims = decode_token(refresh)
    store_refresh_token(user.id, claims["jti"])

    return jsonify({
        "access_token": access,
        "refresh_token": refresh,
        "token_type": "bearer",
    }), 200


@auth_bp.post("/refresh")
def refresh():
    data = request.get_json()
    if not data or not data.get("refresh_token"):
        return jsonify({"error": "refresh_token is required"}), 400

    try:
        claims = decode_token(data["refresh_token"])
    except Exception:
        return jsonify({"error": "Invalid refresh token"}), 401

    if claims.get("type") != "refresh":
        return jsonify({"error": "Token is not a refresh token"}), 401

    user_id = claims["sub"]
    jti = claims["jti"]

    if not is_refresh_token_valid(user_id, jti):
        return jsonify({"error": "Refresh token revoked or expired"}), 401

    # Token rotation: revoke old, issue new pair
    revoke_refresh_token(user_id, jti)

    user = db.session.get(User, user_id)
    if not user:
        return jsonify({"error": "User not found"}), 401

    new_access = create_access_token(user.id, user.role.value)
    new_refresh = create_refresh_token(user.id)
    new_claims = decode_token(new_refresh)
    store_refresh_token(user.id, new_claims["jti"])

    return jsonify({
        "access_token": new_access,
        "refresh_token": new_refresh,
        "token_type": "bearer",
    }), 200


@auth_bp.get("/me")
@require_auth
def me():
    return jsonify(g.current_user.to_dict()), 200
