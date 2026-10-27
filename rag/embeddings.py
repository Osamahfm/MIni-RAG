import os
import time

import httpx
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

from rag.config import ROOT


class RAGError(Exception):
    """An error whose message is safe to display to the user."""


def api_call(operation, sleep=time.sleep):
    for attempt in range(3):
        try:
            return operation()
        except Exception as exc:
            code = getattr(exc, "code", None)
            transient = code in (429, 500, 502, 503, 504) or isinstance(exc, (httpx.TransportError, ConnectionError, TimeoutError))
            if transient and attempt < 2:
                sleep(2 ** attempt)
                continue
            if code in (400, 401, 403):
                message = "Gemini rejected the request. Check your API key, permissions, and model configuration."
            elif code == 404:
                message = "The configured Gemini model is unavailable for this API key."
            elif code == 429:
                message = "Gemini rate limit reached. Wait before trying again and check your quota."
            elif transient:
                message = "Gemini is temporarily unavailable or the network failed. Please try again later."
            else:
                message = "Gemini could not complete the request. Check your connection and model configuration."
            raise RAGError(message) from None


class GeminiService:
    def __init__(self, settings, client=None, api_key=None):
        self.settings = settings
        if client is None:
            load_dotenv(ROOT / ".env")
            key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY")
            if not key or not key.strip():
                raise RAGError("GEMINI_API_KEY is missing. Add it to the project's .env file and restart.")
            client = genai.Client(api_key=key, http_options=types.HttpOptions(
                timeout=60000, retry_options=types.HttpRetryOptions(attempts=1)))
        self.client = client

    def embed(self, texts, task_type):
        vectors = []
        for start in range(0, len(texts), 32):
            batch = texts[start:start + 32]
            result = api_call(lambda: self.client.models.embed_content(
                model=self.settings.embedding_model, contents=batch,
                config=types.EmbedContentConfig(task_type=task_type,
                                                output_dimensionality=self.settings.dimension)))
            values = [item.values for item in (result.embeddings or [])]
            try:
                array = np.asarray(values, dtype=np.float32)
            except (ValueError, TypeError):
                raise RAGError("Gemini returned invalid embeddings. Please retry indexing.") from None
            if array.shape != (len(batch), self.settings.dimension) or not np.isfinite(array).all() or np.any(np.linalg.norm(array, axis=1) == 0):
                raise RAGError("Gemini returned invalid embeddings. Please retry indexing.")
            vectors.extend(values)
        return np.asarray(vectors, dtype=np.float32)

    def generate(self, prompt, instruction):
        response = api_call(lambda: self.client.models.generate_content(
            model=self.settings.generation_model, contents=prompt,
            config=types.GenerateContentConfig(system_instruction=instruction, temperature=0.1)))
        return (response.text or "").strip()
