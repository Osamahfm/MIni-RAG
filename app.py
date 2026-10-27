"""Streamlit entry point: python -m streamlit run app.py."""
import hashlib
import json

import streamlit as st

from rag.config import ROOT, Settings
from rag.embeddings import OllamaService, RAGError
from rag.generator import ask, citation
from rag.vector_store import VectorStore

QUICK_PROMPTS = (
    "Summarize the key ideas",
    "What are the main takeaways?",
    "Compare the documents",
)


def display_message(message):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("cited_sources"):
            st.caption("Cited references")
            for source in message["cited_sources"]:
                st.text(citation(source))
        if message.get("query"):
            with st.expander("Search query"):
                st.text(message["query"])
        if message.get("sources"):
            with st.expander(f"Retrieved passages ({len(message['sources'])})"):
                st.caption("Search results may include passages that were not cited in the answer. Scores measure similarity, not confidence.")
                for source in message["sources"]:
                    with st.container(border=True):
                        st.caption(f"{source['source']} · Page {source['page']} · Similarity {source['score']:.3f}")
                        st.write(source["text"])


def main():
    st.set_page_config(page_title="Mini RAG | Chat with PDFs", page_icon="📚", layout="wide")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("database", None)
    st.session_state.setdefault("selection", None)
    st.session_state.setdefault("pending_question", None)
    service = None
    try:
        service = OllamaService(Settings())
    except RAGError as exc:
        st.warning(str(exc))

    st.title("Mini RAG", icon=":material/menu_book:")
    st.caption("Ask questions across your PDFs and trace every answer back to its source.")

    with st.sidebar:
        st.header("Your library", icon=":material/folder_open:")
        files = st.file_uploader("Add PDF files", type="pdf", accept_multiple_files=True,
                                 help="Text and scanned PDFs are supported. Scanned pages use local CPU OCR and take longer to index.")
        use_sample = st.checkbox("Use included document.pdf", value=False) if (ROOT / "document.pdf").exists() else False
        with st.expander("Retrieval settings", icon=":material/tune:"):
            chunk_size = st.number_input("Chunk size (characters)", min_value=100, max_value=2000, value=500, step=100)
            overlap = st.number_input("Chunk overlap (characters)", min_value=0, max_value=int(chunk_size) - 1, value=min(100, int(chunk_size) - 1), step=10)
            top_k = st.slider("Retrieved passages", 3, 5, 4)
        settings = Settings(chunk_size=int(chunk_size), chunk_overlap=int(overlap))
        documents = [(f.name, f.getvalue()) for f in files]
        if use_sample:
            documents.append(("document.pdf", (ROOT / "document.pdf").read_bytes()))
        selection = hashlib.sha256(json.dumps({"files": sorted((n, hashlib.sha256(b).hexdigest()) for n, b in documents),
                                               "settings": settings.index_config()}, sort_keys=True).encode()).hexdigest()
        if selection != st.session_state.selection:
            st.session_state.database = None
            st.session_state.messages = []
            st.session_state.pending_question = None
            st.session_state.selection = selection
        if st.button("Build document library", type="primary", icon=":material/library_add:",
                     disabled=not documents or service is None, width="stretch"):
            progress = st.progress(0, text="Reading documents and preparing the index…")
            try:
                service.settings = settings
                with st.spinner("Preparing your library…"):
                    st.session_state.database = VectorStore(ROOT / "cache", settings).build(documents, service, progress.progress)
                st.session_state.messages = []
                st.session_state.pending_question = None
            except RAGError as exc:
                st.error(str(exc))
            except OSError:
                st.error("Could not read or save the library. Check available disk space and cache folder permissions.")
            except Exception:
                st.error("The library could not be built. Check the PDFs and try again.")
            finally:
                progress.empty()
        if st.button("Clear chat", icon=":material/delete_sweep:", disabled=not st.session_state.messages,
                     width="stretch"):
            st.session_state.messages = []
            st.session_state.pending_question = None
        st.caption("Embeddings and answers use Ollama. With the default localhost address, PDF text and questions stay on this computer.")

    database = st.session_state.database
    if database:
        columns = st.columns(3)
        with columns[0].container(border=True):
            st.metric("Documents", len(database.documents))
        with columns[1].container(border=True):
            st.metric("Pages", sum(d["pages"] for d in database.documents))
        with columns[2].container(border=True):
            st.metric("Indexed chunks", len(database.chunks))
        st.badge("Loaded from cache" if database.cached else "Library ready",
                 icon=":material/check_circle:", color="green")
        with st.sidebar:
            with st.expander(f"Indexed documents ({len(database.documents)})", icon=":material/description:"):
                for doc in database.documents:
                    with st.container(border=True):
                        st.text(doc["source"])
                        st.caption(f"{doc['pages']} pages · {doc.get('chunks', 0)} chunks")
                        if doc.get("ocr_pages"):
                            st.caption(f"Local OCR read {doc['ocr_pages']} scanned pages. Check passages for recognition errors.")
        for warning in database.warnings:
            st.warning(warning)
    else:
        with st.container(border=True):
            st.subheader("Start with your PDFs", icon=":material/auto_stories:")
            st.write("Build a searchable library, then ask questions grounded in the documents you choose.")
            steps = st.columns(3)
            for column, icon, title, detail in zip(
                steps,
                (":material/upload_file:", ":material/library_add:", ":material/forum:"),
                ("Add documents", "Build your library", "Ask with confidence"),
                ("Upload PDFs or use the sample.", "Create a searchable index.", "Inspect answers and page references."),
            ):
                with column.container(border=True):
                    st.markdown(f"{icon} **{title}**")
                    st.caption(detail)

    for message in st.session_state.messages:
        display_message(message)
    if database and not st.session_state.messages:
        st.caption("Try a question to get started")
        with st.container(horizontal=True):
            for index, prompt in enumerate(QUICK_PROMPTS):
                if st.button(prompt, key=f"quick_prompt_{index}", icon=":material/arrow_outward:",
                             width="stretch"):
                    st.session_state.pending_question = prompt
    question = st.chat_input("Ask about your documents…", disabled=database is None or service is None)
    question = question or st.session_state.pending_question
    st.session_state.pending_question = None
    if question:
        if not question.strip():
            st.warning("Please enter a non-empty question.")
            return
        display_message({"role": "user", "content": question})
        try:
            service.settings = settings
            with st.spinner("Searching your documents and preparing an answer…"):
                answer = ask(question, st.session_state.messages, database, service, top_k)
            st.session_state.messages.extend([{"role": "user", "content": question}, answer])
            display_message(answer)
        except RAGError as exc:
            st.error(str(exc))
        except Exception:
            st.error("The question could not be completed. Please retry or rebuild the library.")


if __name__ == "__main__":
    main()
