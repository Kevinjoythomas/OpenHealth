import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=r"C:\OpenHealth\.env", override=True)


class Config:
    CHROMA_PATH: str = os.getenv("CHROMA_PATH", "./chroma")
    OLLAMA_BASE_URL: str = os.getenv(
        "OLLAMA_BASE_URL", "http://host.docker.internal:11434"
    )
    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "nomic-embed-text")
    DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "5"))
