import os
from dataclasses import dataclass, asdict, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    chunk_size: int = 500
    chunk_overlap: int = 100
    embedding_model: str = field(default_factory=lambda: os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"))
    dimension: int = 768
    generation_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen3:4b"))
    ollama_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))

    def __post_init__(self):
        if self.chunk_size < 100 or not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("Overlap must be smaller than chunk size (minimum 100).")
        if self.dimension < 1:
            raise ValueError("Embedding dimension must be positive.")

    def index_config(self):
        config = asdict(self)
        config.pop("generation_model")
        config.pop("ollama_url")
        return {"schema": 2, "provider": "ollama", "embedding_prefix": "nomic-search-v1",
                "splitter": "recursive-v1", **config}
