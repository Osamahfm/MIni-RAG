import faiss
import numpy as np

from rag.embeddings import RAGError


def retrieve(database, query, service, top_k=4):
    if not query.strip():
        raise RAGError("Enter a question before searching.")
    if database.index.ntotal == 0:
        return []
    if database.index.ntotal != len(database.chunks):
        raise RAGError("Index and metadata do not match. Rebuild the collection.")
    vectors = np.ascontiguousarray(service.embed([query], "RETRIEVAL_QUERY"), dtype=np.float32)
    if vectors.shape != (1, database.index.d) or not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) == 0):
        raise RAGError("Query embedding values or dimensions do not match the index.")
    faiss.normalize_L2(vectors)
    try:
        scores, indices = database.index.search(vectors, min(max(1, top_k), database.index.ntotal))
    except RuntimeError:
        raise RAGError("FAISS search failed. Rebuild the collection.") from None
    return [{**database.chunks[int(i)], "score": float(score)}
            for score, i in zip(scores[0], indices[0]) if 0 <= i < len(database.chunks)]
