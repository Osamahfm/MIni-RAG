"""Explicit optional model availability check; never runs on import."""
from rag.config import Settings
from rag.embeddings import OllamaService, RAGError


def main():
    try:
        service = OllamaService(Settings())
        names = service.list_models()
        for name in names:
            print(name)
        if not names:
            print("No local models installed.")
        print(f"Configured answer model: {service.settings.generation_model}")
        print(f"Configured embedding model: {service.settings.embedding_model}")
    except RAGError as exc:
        print(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
