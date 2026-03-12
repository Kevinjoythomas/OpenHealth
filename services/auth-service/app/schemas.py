"""Request/response dataclasses for auth-service routes."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class SignupRequest:
    email: str
    password: str
    name: str
    role: str = "doctor"
    specialization: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "SignupRequest":
        return cls(
            email=data.get("email", ""),
            password=data.get("password", ""),
            name=data.get("name", ""),
            role=data.get("role", "doctor"),
            specialization=data.get("specialization"),
        )


@dataclass
class LoginRequest:
    email: str
    password: str

    @classmethod
    def from_dict(cls, data: dict) -> "LoginRequest":
        return cls(email=data.get("email", ""), password=data.get("password", ""))


@dataclass
class TokenResponse:
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

    def to_dict(self) -> dict:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "token_type": self.token_type,
        }
