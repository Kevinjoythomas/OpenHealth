"""Shared exception hierarchy for all OpenHealth services."""


class OpenHealthError(Exception):
    """Base class for all application errors."""
    http_status: int = 500
    default_message: str = "An unexpected error occurred."

    def __init__(self, message: str | None = None):
        self.message = message or self.default_message
        super().__init__(self.message)

    def to_dict(self) -> dict:
        return {"error": self.message}


# ── Auth / Identity ───────────────────────────────────────────────────────────

class AuthenticationError(OpenHealthError):
    http_status = 401
    default_message = "Authentication failed."


class AuthorizationError(OpenHealthError):
    http_status = 403
    default_message = "You do not have permission to perform this action."


class TokenExpiredError(AuthenticationError):
    default_message = "Token has expired."


class InvalidTokenError(AuthenticationError):
    default_message = "Invalid token."


# ── Resource ──────────────────────────────────────────────────────────────────

class NotFoundError(OpenHealthError):
    http_status = 404
    default_message = "Resource not found."


class ConflictError(OpenHealthError):
    http_status = 409
    default_message = "Resource already exists."


# ── Validation ────────────────────────────────────────────────────────────────

class ValidationError(OpenHealthError):
    http_status = 422
    default_message = "Validation error."


# ── Downstream services ───────────────────────────────────────────────────────

class ServiceUnavailableError(OpenHealthError):
    http_status = 503
    default_message = "An upstream service is unavailable."


class LLMError(OpenHealthError):
    http_status = 502
    default_message = "LLM service error."


class RetrievalError(OpenHealthError):
    http_status = 502
    default_message = "Retrieval service error."
