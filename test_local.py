"""Explicit live smoke test. Run manually; pytest never makes API calls."""
from rag.config import Settings
from rag.embeddings import OllamaService, RAGError


def main():
    try:
        service = OllamaService(Settings())
        values = service.embed(["MapReduce processes data in parallel."], "RETRIEVAL_DOCUMENT")
        print(f"Embedding successful: {values.shape}")
        answer = service.generate("Reply with the word ready.", "Follow the short request.")
        if not answer:
            raise RAGError("The generation model returned no text.")
        print("Generation successful:", answer)
    except RAGError as exc:
        print(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
