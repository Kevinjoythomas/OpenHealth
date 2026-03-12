"""Thin wrapper around OllamaLLM.

Centralises model initialisation and provides a placeholder for retry /
circuit-breaker logic (Phase 4).
"""
import logging
import os

from langchain_ollama import OllamaLLM

log = logging.getLogger(__name__)

_llm: OllamaLLM | None = None


def get_llm() -> OllamaLLM:
    """Return a cached OllamaLLM instance (created once per process)."""
    global _llm
    if _llm is None:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
        model = os.getenv(
            "LLM_MODEL", "hf.co/kevinjoythomas/medical-loratuned-chatbot-GGUF"
        )
        log.info("Initialising OllamaLLM model=%s base_url=%s", model, base_url)
        _llm = OllamaLLM(model=model, base_url=base_url)
    return _llm


def invoke(prompt: str) -> str:
    """Invoke the LLM with a plain string prompt.

    Phase 4: add tenacity retry + circuit breaker here.
    """
    try:
        result = get_llm().invoke(prompt)
        return result if isinstance(result, str) else str(result)
    except Exception as exc:
        log.error("LLM invocation failed: %s", exc)
        raise
