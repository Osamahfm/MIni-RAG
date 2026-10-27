import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from rag.embeddings import RAGError
from rag.pdf_processor import document_hash, process_pdf


def stable_key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@dataclass
class Database:
    index: object
    chunks: list
    documents: list
    warnings: list
    key: str
    cached: bool = False


class VectorStore:
    def __init__(self, cache_dir, settings):
        self.cache_dir = Path(cache_dir)
        self.settings = settings

    def _load(self, kind, key, config):
        folder = self.cache_dir / kind / key
        try:
            manifest = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
            raw = (folder / "index.faiss").read_bytes()
            if manifest["config"] != config or manifest["sha256"] != document_hash(raw):
                return None
            chunks = manifest["chunks"]
            if not isinstance(chunks, list) or not chunks:
                return None
            for chunk in chunks:
                if not isinstance(chunk.get("text"), str) or not chunk["text"].strip() or not isinstance(chunk.get("source"), str) or type(chunk.get("page")) is not int or chunk["page"] < 1:
                    return None
            documents = manifest["documents"]
            warnings = manifest["warnings"]
            if not isinstance(documents, list) or not documents or not isinstance(warnings, list) or not all(isinstance(w, str) for w in warnings):
                return None
            for doc in documents:
                if not isinstance(doc, dict) or not isinstance(doc.get("source"), str) or type(doc.get("pages")) is not int or doc["pages"] < 0:
                    return None
                doc_chunks = [c for c in chunks if c["document_hash"] == doc.get("hash") and c["source"] == doc["source"]]
                if len(doc_chunks) != doc.get("chunks", 0) or any(c["page"] > doc["pages"] for c in doc_chunks):
                    return None
            if sum(d.get("chunks", 0) for d in documents) != len(chunks):
                return None
            index = faiss.deserialize_index(np.frombuffer(raw, dtype=np.uint8))
            if not isinstance(index, faiss.IndexFlatIP) or index.d != self.settings.dimension or index.ntotal != len(chunks):
                return None
            vectors = index.reconstruct_n(0, index.ntotal)
            if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
                return None
            return Database(index, chunks, documents, warnings, key, True)
        except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError):
            return None

    def _save(self, kind, db, config):
        folder = self.cache_dir / kind / db.key
        folder.mkdir(parents=True, exist_ok=True)
        raw = faiss.serialize_index(db.index).tobytes()
        manifest = {"config": config, "sha256": document_hash(raw), "chunks": db.chunks,
                    "documents": db.documents, "warnings": db.warnings}
        # Unique temporary files + atomic replacements; torn pairs fail checksum validation.
        for name, data in (("index.faiss", raw), ("metadata.json", json.dumps(manifest, ensure_ascii=False, indent=2).encode())):
            temp = folder / f"{name}.{uuid.uuid4().hex}.tmp"
            try:
                temp.write_bytes(data)
                os.replace(temp, folder / name)
            finally:
                temp.unlink(missing_ok=True)

    def build(self, documents, service, progress=None):
        unique, warnings, duplicate_warnings = {}, [], []
        for name, data in documents:
            digest = document_hash(data)
            if digest in unique:
                duplicate_warnings.append(f"{name}: duplicate file content skipped.")
            else:
                unique[digest] = (Path(name.replace("\\", "/")).name, data)
        ordered = sorted(unique.items())
        config = {"settings": self.settings.index_config(),
                  "files": [{"hash": h, "source": n} for h, (n, _) in ordered]}
        key = stable_key(config)
        cached = self._load("collections", key, config)
        if cached:
            cached.warnings = list(cached.warnings) + duplicate_warnings
            return cached
        chunks, stats, arrays = [], [], []
        for position, (digest, (name, data)) in enumerate(ordered):
            doc_config = {"settings": self.settings.index_config(), "hash": digest}
            doc_key = stable_key(doc_config)
            doc_db = self._load("documents", doc_key, doc_config)
            if doc_db is None:
                doc_chunks, stat, issues = process_pdf(name, data, self.settings)
                warnings.extend(issues)
                if not doc_chunks:
                    stats.append(stat)
                    continue
                vectors = service.embed([c["text"] for c in doc_chunks], "RETRIEVAL_DOCUMENT")
                index = self._index(vectors, len(doc_chunks))
                doc_db = Database(index, doc_chunks, [stat], issues, doc_key)
                self._save("documents", doc_db, doc_config)
            else:
                warnings.extend(issue.replace(doc_db.documents[0]["source"], name) for issue in doc_db.warnings)
            chunks.extend({**c, "source": name} for c in doc_db.chunks)
            stats.extend({**s, "source": name} for s in doc_db.documents)
            arrays.append(doc_db.index.reconstruct_n(0, doc_db.index.ntotal))
            if progress:
                progress((position + 1) / len(ordered))
        if not chunks:
            raise RAGError("No readable PDF text was found. Upload text PDFs or apply OCR/unlock them first. " + " ".join(warnings))
        db = Database(self._index(np.concatenate(arrays), len(chunks)), chunks, stats,
                      list(dict.fromkeys(warnings)), key)
        self._save("collections", db, config)
        db.warnings = db.warnings + duplicate_warnings
        return db

    def _index(self, vectors, count):
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.shape != (count, self.settings.dimension) or not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) == 0):
            raise RAGError("Invalid vector dimensions or values. Rebuild the collection.")
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(self.settings.dimension)
        index.add(vectors)
        return index
