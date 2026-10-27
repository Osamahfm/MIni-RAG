import json
from dataclasses import replace
from unittest.mock import Mock

import faiss
import httpx
import numpy as np
import pytest

from conftest import make_pdf
from rag.config import ROOT, Settings
from rag.embeddings import OllamaService, RAGError, api_call
from rag.generator import UNSUPPORTED, ask, generate_answer, rewrite_query
from rag.pdf_processor import process_pdf
from rag.retriever import retrieve
from rag.vector_store import Database, VectorStore
from rag.pdf_processor import document_hash


def test_extraction_page_numbers_and_chunks(documents):
    chunks, stat, warnings = process_pdf(*documents[0], Settings())
    assert {c["page"] for c in chunks} == {1, 3}
    assert all(c["source"] == "lecture.pdf" and len(c["text"]) <= 500 for c in chunks)
    assert stat["pages"] == 3 and stat["text_pages"] == 2 and warnings


@pytest.mark.parametrize("data", [b"not a PDF", make_pdf([""]), make_pdf(["secret"], "password")])
def test_bad_empty_encrypted_pdf(data):
    chunks, _, warnings = process_pdf("bad.pdf", data, Settings())
    assert not chunks and warnings


def test_real_project_pdf():
    chunks, stat, _ = process_pdf("document.pdf", (ROOT / "document.pdf").read_bytes(), Settings())
    assert chunks and stat["pages"] > 0
    assert all(c["page"] <= stat["pages"] for c in chunks)


def test_persistence_dedupe_multi_document(tmp_path, service, documents):
    store = VectorStore(tmp_path, Settings())
    db = store.build(documents + [("duplicate.pdf", documents[0][1])], service)
    assert db.index.ntotal == len(db.chunks) == 3
    assert len(db.documents) == 2
    assert any("duplicate" in w for w in db.warnings)
    assert len(service.calls) == 2
    loaded = VectorStore(tmp_path, Settings()).build(list(reversed(documents)), service)
    assert loaded.cached and len(service.calls) == 2
    found = retrieve(loaded, "What is MapReduce?", service, 3)
    assert found[0]["source"] == "lecture.pdf"
    assert found[0]["score"] == pytest.approx(1)
    assert {s["source"] for s in found} == {"lecture.pdf", "biology.pdf"}
    assert service.calls[-1][1] == "RETRIEVAL_QUERY"
    assert np.allclose(np.linalg.norm(db.index.reconstruct_n(0, db.index.ntotal), axis=1), 1)


def test_added_and_renamed_files_reuse_embeddings(tmp_path, service, documents):
    store = VectorStore(tmp_path, Settings())
    store.build(documents[:1], service)
    store.build(documents, service)
    assert len(service.calls) == 2
    renamed = store.build([("renamed.pdf", documents[0][1])], service)
    assert len(service.calls) == 2
    assert all(c["source"] == "renamed.pdf" for c in renamed.chunks)


@pytest.mark.parametrize("setting", [{"chunk_size": 600}, {"chunk_overlap": 50}, {"embedding_model": "changed"}, {"dimension": 16}])
def test_cache_invalidates_settings(tmp_path, service, documents, setting):
    original = VectorStore(tmp_path, Settings()).build(documents, service)
    settings = replace(Settings(), **setting)
    service.dimension = settings.dimension
    changed = VectorStore(tmp_path, settings).build(documents, service)
    assert changed.key != original.key and len(service.calls) == 4


def test_cache_invalidates_content(tmp_path, service, documents):
    store = VectorStore(tmp_path, Settings())
    original = store.build(documents, service)
    changed = store.build([(documents[0][0], make_pdf(["New MapReduce lecture."]))] + documents[1:], service)
    assert changed.key != original.key and len(service.calls) == 3


@pytest.mark.parametrize("damage", ["missing", "checksum", "metadata", "dimensions"])
def test_damaged_collection_rebuilt_without_embeddings(tmp_path, service, documents, damage):
    store = VectorStore(tmp_path, Settings())
    db = store.build(documents, service)
    folder = tmp_path / "collections" / db.key
    path = folder / "metadata.json"
    manifest = json.loads(path.read_text())
    if damage == "missing":
        (folder / "index.faiss").unlink()
    elif damage == "checksum":
        (folder / "index.faiss").write_bytes(b"broken")
    elif damage == "metadata":
        manifest["chunks"].pop()
    else:
        manifest["config"]["settings"]["dimension"] = 12
    path.write_text(json.dumps(manifest))
    rebuilt = store.build(documents, service)
    assert not rebuilt.cached and rebuilt.index.ntotal == 3 and len(service.calls) == 2


def test_invalid_document_and_valid_document(tmp_path, service, documents):
    db = VectorStore(tmp_path, Settings()).build(documents[:1] + [("bad.pdf", b"bad")], service)
    assert len(db.documents) == 2 and db.index.ntotal == 2 and db.warnings
    with pytest.raises(RAGError, match="No readable"):
        VectorStore(tmp_path, Settings()).build([], service)


@pytest.mark.parametrize("question, expected", [("What are its advantages?", "MapReduce's"), ("How does it work?", "MapReduce"), ("Which lecture explains it?", "MapReduce")])
def test_rewrite_followups(service, question, expected):
    history = [{"role": "user", "content": "What is MapReduce?"}]
    assert expected in rewrite_query(question, history, service)
    assert rewrite_query("What is MapReduce?", [], service) == "What is MapReduce?"
    assert history == [{"role": "user", "content": "What is MapReduce?"}]


def test_rewrite_failure_fallback():
    service = Mock()
    service.generate.side_effect = RAGError("unavailable")
    assert rewrite_query("How does it work?", [{"role": "user", "content": "MapReduce"}], service) == "How does it work?"


def test_conversation_and_grounding(tmp_path, service, documents):
    db = VectorStore(tmp_path, Settings()).build(documents, service)
    history = [{"role": "user", "content": "What is MapReduce?"}]
    answer = ask("How does it work?", history, db, service)
    assert answer["query"] == "How does MapReduce work?"
    assert answer["cited_sources"] and len(answer["sources"]) > len(answer["cited_sources"])
    assert "[Source: lecture.pdf, Page " in answer["content"]
    payload, instruction = service.prompts[-1]
    assert "history" not in payload and "untrusted" in instruction
    assert len(history) == 1
    with pytest.raises(RAGError):
        ask("  ", history, db, service)


@pytest.mark.parametrize("answer", ["", "Unsupported assertion.", "A fact [Source: invented.pdf, Page 1]", "A fact [Source: lecture.pdf, Page 99]", "A fact [Source: lecture.pdf, Page: 1]"])
def test_invalid_citations_fail_closed(answer):
    service = Mock()
    service.generate.return_value = answer
    result, cited = generate_answer("Question", "Query", [{"source": "lecture.pdf", "page": 1, "text": "Evidence"}], service)
    assert result == UNSUPPORTED and not cited


def test_empty_index(service):
    db = Database(faiss.IndexFlatIP(768), [], [], [], "empty")
    assert retrieve(db, "question", service) == []
    assert not service.calls


def test_no_api_key_required():
    assert OllamaService(Settings()).settings.generation_model == Settings().generation_model


@pytest.mark.parametrize("url", ["", "localhost:11434", "ftp://localhost", "http://localhost:11434?x=1"])
def test_invalid_ollama_url(url):
    with pytest.raises(RAGError, match="OLLAMA_BASE_URL"):
        OllamaService(Settings(ollama_url=url))


@pytest.mark.parametrize("code, calls", [(429, 3), (503, 3), (401, 1), (403, 1), (404, 1)])
def test_retry_and_redacted_errors(code, calls):
    request = httpx.Request("POST", "http://localhost:11434/api/chat")
    error = httpx.HTTPStatusError("PRIVATE_TEXT_MUST_NOT_APPEAR", request=request,
                                  response=httpx.Response(code, request=request))
    operation, sleep = Mock(side_effect=error), Mock()
    with pytest.raises(RAGError) as exc:
        api_call(operation, sleep)
    assert operation.call_count == calls and sleep.call_count == calls - 1
    assert "PRIVATE_TEXT" not in str(exc.value)


def test_network_retry_success():
    operation = Mock(side_effect=[httpx.ConnectError("offline"), "ok"])
    sleep = Mock()
    assert api_call(operation, sleep) == "ok"
    sleep.assert_called_once_with(1)


def test_embedding_batching_and_validation():
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"embeddings": [[1., 0., 0.]] * len(requests[-1]["input"])})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        service = OllamaService(Settings(dimension=3), client=client)
        result = service.embed(["text"] * 33, "RETRIEVAL_DOCUMENT")
        assert result.shape == (33, 3) and result.dtype == np.float32
        assert len(requests) == 2 and len(requests[0]["input"]) == 32
        assert requests[0]["input"][0] == "search_document: text"
        assert requests[0]["truncate"] is False
        service.embed(["question"], "RETRIEVAL_QUERY")
        assert requests[-1]["input"] == ["search_query: question"]
        assert service.embed([], "RETRIEVAL_QUERY").shape == (0, 3)


@pytest.mark.parametrize("values", [[], [[1, 0]], [[0, 0, 0]], [[float("inf"), 0, 0]], [["bad", 0, 0]]])
def test_invalid_embeddings(values):
    client = Mock()
    client.request.return_value.json.return_value = {"embeddings": values}
    with pytest.raises(RAGError, match="invalid embeddings"):
        OllamaService(Settings(dimension=3), client=client).embed(["text"], "RETRIEVAL_QUERY")


def test_generation_and_model_list():
    requests = []

    def handler(request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen3:4b"}]})
        assert request.url.path == "/api/chat"
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": " ready "}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        service = OllamaService(Settings(), client=client)
        assert service.list_models() == ["qwen3:4b"]
        assert service.generate("question", "instruction") == "ready"
        assert requests[0]["messages"] == [{"role": "system", "content": "instruction"},
                                           {"role": "user", "content": "question"}]
        assert requests[0]["stream"] is False and requests[0]["think"] is False


def test_answer_model_changes_reuse_index_settings():
    settings = Settings()
    assert replace(settings, generation_model="other").index_config() == settings.index_config()
    assert settings.index_config()["provider"] == "ollama"


@pytest.mark.parametrize("damage", ["index_dimension", "page_range", "wrong_source", "wrong_hash", "invalid_statistics"])
def test_cache_consistency_validation(tmp_path, service, documents, damage):
    store = VectorStore(tmp_path, Settings())
    db = store.build(documents, service)
    folder = tmp_path / "collections" / db.key
    path = folder / "metadata.json"
    manifest = json.loads(path.read_text())
    if damage == "index_dimension":
        index = faiss.IndexFlatIP(3)
        index.add(np.array([[1, 0, 0]] * 3, dtype=np.float32))
        raw = faiss.serialize_index(index).tobytes()
        (folder / "index.faiss").write_bytes(raw)
        manifest["sha256"] = document_hash(raw)
    elif damage == "page_range":
        manifest["chunks"][0]["page"] = 999
    elif damage == "wrong_source":
        manifest["chunks"][0]["source"] = "unknown.pdf"
    elif damage == "wrong_hash":
        manifest["chunks"][0]["document_hash"] = "wrong"
    else:
        manifest["documents"][0]["chunks"] = 999
    path.write_text(json.dumps(manifest))
    rebuilt = store.build(documents, service)
    assert not rebuilt.cached and rebuilt.index.ntotal == 3 and len(service.calls) == 2


def test_unreadable_page_is_skipped():
    from unittest.mock import patch
    reader = Mock()
    bad, good = Mock(), Mock()
    bad.extract_text.side_effect = ValueError("Unreadable")
    good.extract_text.return_value = "Readable text"
    reader.is_encrypted = False
    reader.pages = [bad, good]
    with patch("rag.pdf_processor.PdfReader", return_value=reader):
        chunks, stat, warnings = process_pdf("pages.pdf", b"placeholder", Settings())
    assert chunks[0]["page"] == 2 and stat["pages"] == 2
    assert any("page 1 could not be read" in w for w in warnings)


def test_duplicate_warning_not_persisted(tmp_path, service, documents):
    store = VectorStore(tmp_path, Settings())
    store.build(documents + [documents[0]], service)
    db = store.build(documents, service)
    assert db.cached and not any("duplicate" in w for w in db.warnings)
