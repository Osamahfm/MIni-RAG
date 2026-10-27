import time

import httpx
import numpy as np


class RAGError(Exception):
    """An error whose message is safe to display to the user."""


def api_call(operation, sleep=time.sleep):
    for attempt in range(3):
        try:
            return operation()
        except Exception as exc:
            code = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
            transient = code in (429, 500, 502, 503, 504) or isinstance(exc, (httpx.TransportError, ConnectionError, TimeoutError))
            if transient and attempt < 2:
                sleep(2 ** attempt)
                continue
            if code == 404:
                message = "Ollama model or endpoint not found. Run python check_models.py and pull the configured models."
            elif code in (400, 401, 403):
                message = "Ollama rejected the request. Check your local models and configuration."
            elif code == 429:
                message = "Ollama is busy. Wait before trying again."
            elif transient:
                message = "Cannot reach Ollama or the request timed out. Start Ollama (ollama serve) and retry."
            else:
                message = "Ollama could not complete the request. Check that the configured local models are installed."
            raise RAGError(message) from None


class OllamaService:
    """Local embeddings and generation through Ollama's native HTTP API."""

    def __init__(self, settings, client=None):
        self.settings = settings
        self.client = client
        if not settings.embedding_model.strip() or not settings.generation_model.strip():
            raise RAGError("OLLAMA_MODEL and OLLAMA_EMBEDDING_MODEL must not be empty.")
        try:
            url = httpx.URL(settings.ollama_url)
            if url.scheme not in ("http", "https") or not url.host or url.query or url.fragment:
                raise ValueError
        except (httpx.InvalidURL, ValueError):
            raise RAGError("OLLAMA_BASE_URL must be a valid HTTP URL, for example http://localhost:11434.") from None

    def _request(self, method, path, payload=None):
        def operation():
            kwargs = {"timeout": 300.0}
            if payload is not None:
                kwargs["json"] = payload
            url = self.settings.ollama_url.rstrip("/") + path
            if self.client is None:
                response = httpx.request(method, url, trust_env=False, **kwargs)
            else:
                response = self.client.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        return api_call(operation)

    def list_models(self):
        result = self._request("GET", "/api/tags")
        try:
            names = [model["name"] for model in result["models"]]
            if not all(isinstance(name, str) for name in names):
                raise ValueError
            return names
        except (KeyError, TypeError, ValueError):
            raise RAGError("Ollama returned an invalid model list.") from None

    def embed(self, texts, task_type):
        if task_type not in ("RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"):
            raise RAGError("Unsupported embedding task type.")
        vectors = []
        prefix = "search_document: " if task_type == "RETRIEVAL_DOCUMENT" else "search_query: "
        for start in range(0, len(texts), 32):
            batch = texts[start:start + 32]
            inputs = [prefix + text for text in batch] if self.settings.embedding_model.split(":")[0] == "nomic-embed-text" else batch
            result = self._request("POST", "/api/embed", {
                "model": self.settings.embedding_model, "input": inputs, "truncate": False,
            })
            try:
                array = np.asarray(result["embeddings"], dtype=np.float32)
                if array.shape != (len(batch), self.settings.dimension) or not np.isfinite(array).all() or np.any(np.linalg.norm(array, axis=1) == 0):
                    raise ValueError
            except (KeyError, ValueError, TypeError, OverflowError):
                raise RAGError("Ollama returned invalid embeddings. Check the embedding model and dimension, then rebuild the library.") from None
            vectors.extend(array)
        return np.asarray(vectors, dtype=np.float32).reshape(-1, self.settings.dimension)

    def generate(self, prompt, instruction):
        result = self._request("POST", "/api/chat", {
            "model": self.settings.generation_model,
            "messages": [{"role": "system", "content": instruction},
                         {"role": "user", "content": prompt}],
            "stream": False, "think": False,
            "options": {"temperature": 0.1, "num_ctx": 8192},
        })
        try:
            return result["message"]["content"].strip()
        except (KeyError, TypeError, AttributeError):
            raise RAGError("Ollama returned an invalid answer.") from None
