"""Explicit optional model availability check; never runs on import."""
from rag.config import Settings
from rag.embeddings import GeminiService, RAGError, api_call


def main():
    try:
        service = GeminiService(Settings())
        names = api_call(lambda: [m.name for m in service.client.models.list()])
        for name in names:
            print(name)
    except RAGError as exc:
        print(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
