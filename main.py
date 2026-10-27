"""Optional terminal interface sharing the same RAG pipeline."""
import argparse
from pathlib import Path

from rag.config import ROOT, Settings
from rag.embeddings import OllamaService, RAGError
from rag.generator import ask
from rag.vector_store import VectorStore


def main():
    parser = argparse.ArgumentParser(description="Chat with local PDFs using Ollama and FAISS.")
    parser.add_argument("pdfs", nargs="*", type=Path, default=[ROOT / "document.pdf"])
    args = parser.parse_args()
    try:
        settings = Settings()
        service = OllamaService(settings)
        db = VectorStore(ROOT / "cache", settings).build([(p.name, p.read_bytes()) for p in args.pdfs], service)
        print(f"Ready: {len(db.documents)} documents, {len(db.chunks)} chunks. Type /clear or /quit.")
        for warning in db.warnings:
            print(warning)
        history = []
        while True:
            question = input("\nYou: ").strip()
            if question == "/quit":
                break
            if question == "/clear":
                history.clear()
                continue
            if not question:
                continue
            try:
                answer = ask(question, history, db, service)
                print(f"Search: {answer['query']}\n\n{answer['content']}")
                history.extend([{"role": "user", "content": question}, answer])
            except RAGError as exc:
                print(str(exc))
    except RAGError as exc:
        print(str(exc))
        return 1
    except OSError:
        print("Could not read the PDFs or write the cache. Check file paths and permissions.")
        return 1
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
