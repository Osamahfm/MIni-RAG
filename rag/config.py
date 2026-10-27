from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    chunk_size: int = 500
    chunk_overlap: int = 100
    embedding_model: str = "gemini-embedding-001"
    dimension: int = 768
    generation_model: str = "gemini-3.5-flash-lite"

    def __post_init__(self):
        if self.chunk_size < 100 or not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("Overlap must be smaller than chunk size (minimum 100).")
        if self.dimension < 1:
            raise ValueError("Embedding dimension must be positive.")

    def index_config(self):
        config = asdict(self)
        config.pop("generation_model")
        return {"schema": 1, "splitter": "recursive-v1", **config}
