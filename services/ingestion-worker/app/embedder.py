"""OllamaEmbeddings wrapper — same model config as original app."""
import logging
import os

from langchain_ollama import OllamaEmbeddings

log = logging.getLogger(__name__)

_embeddings: OllamaEmbeddings | None = None


def get_embeddings() -> OllamaEmbeddings:
    global _embeddings
    if _embeddings is None:
        model = os.getenv("EMBED_MODEL", "nomic-embed-text")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
        log.info("Initialising OllamaEmbeddings model=%s base_url=%s", model, base_url)
        _embeddings = OllamaEmbeddings(model=model, base_url=base_url)
    return _embeddings
