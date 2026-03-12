"""Request/response dataclasses for chat-orchestrator routes."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class CreateSessionRequest:
    title: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "CreateSessionRequest":
        return cls(title=data.get("title"))


@dataclass
class SendMessageRequest:
    message: str

    @classmethod
    def from_dict(cls, data: dict) -> "SendMessageRequest":
        return cls(message=data.get("message", "").strip())


@dataclass
class IngestDocumentRequest:
    s3_key: str
    filename: str

    @classmethod
    def from_dict(cls, data: dict) -> "IngestDocumentRequest":
        return cls(
            s3_key=data.get("s3_key", ""),
            filename=data.get("filename", ""),
        )
