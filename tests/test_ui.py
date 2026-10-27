from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from conftest import FakeService
from rag.config import ROOT
from rag.embeddings import RAGError
from rag.vector_store import VectorStore


def test_ui_build_chat_followup_clear_and_selection(tmp_path):
    fake = FakeService()
    original = VectorStore.build

    def build(store, *args, **kwargs):
        store.cache_dir = tmp_path
        return original(store, *args, **kwargs)

    with patch("rag.embeddings.OllamaService", return_value=fake), patch.object(VectorStore, "build", build):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=20).run()
        assert not app.exception
        assert app.chat_input[0].disabled
        app.checkbox[0].check().run()
        app.button[0].click().run()
        assert not app.exception
        assert not app.chat_input[0].disabled
        assert len(app.metric) == 3
        app.chat_input[0].set_value("What is MapReduce?").run()
        assert not app.exception
        assert len(app.session_state["messages"]) == 2
        app.chat_input[0].set_value("What are its advantages?").run()
        assert len(app.session_state["messages"]) == 4
        assert "MapReduce" in app.session_state["messages"][-1]["query"]
        assert not app.exception
        app.button[1].click().run()
        assert not app.session_state["messages"]
        assert app.session_state["database"] is not None
        app.checkbox[0].uncheck().run()
        assert app.session_state["database"] is None and app.chat_input[0].disabled


def test_ui_invalid_local_configuration():
    with patch("rag.embeddings.OllamaService", side_effect=RAGError("OLLAMA_BASE_URL is invalid.")):
        app = AppTest.from_file(str(ROOT / "app.py")).run()
        assert not app.exception
        assert "OLLAMA_BASE_URL" in app.warning[0].value
        assert app.chat_input[0].disabled and app.button[0].disabled


def test_ui_quick_prompt_starts_a_grounded_chat(tmp_path):
    fake = FakeService()
    original = VectorStore.build

    def build(store, *args, **kwargs):
        store.cache_dir = tmp_path
        return original(store, *args, **kwargs)

    with patch("rag.embeddings.OllamaService", return_value=fake), patch.object(VectorStore, "build", build):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=20).run()
        app.checkbox[0].check().run()
        app.button[0].click().run()
        assert not app.exception
        prompt_button = next(button for button in app.button if button.label == "Summarize the key ideas")
        prompt_button.click().run()
        assert not app.exception
        assert len(app.session_state["messages"]) == 2
        assert app.session_state["messages"][0]["content"] == "Summarize the key ideas"
